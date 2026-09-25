# SPDX-License-Identifier: Apache-2.0
"""L7 S3: the shared-expert gate (1 x 2560 BF16) as one small kernel.

Problem (kern-decode profile TABLE.md): each MoE layer runs
shared_expert_gate = ReplicatedLinear(2560, 1) as a cuBLAS dot_kernel plus
a reduce_1Block_kernel, then the sigmoid: about 21 us per layer, about
1.0 ms per verify step for 5 KB of weights. It runs on the shared-expert
side stream, so only a part of it is on the critical path.

Change. For M <= MAX_M rows the linear runs one Triton program: each lane
reads 16-byte slices of x and w, keeps FP32 products in a [BM, BK] tile,
and the program sums the tile in a fixed order at the end. The logit is
rounded once to BF16 (the cuBLAS output dtype), so the sigmoid and the
multiply that follow see the same kind of value. The sum order is not the
cuBLAS order: class B.

Knobs: VLLM_KERN_SGATE=1 swaps the quant method of every (1, 2560)
unquantized linear at model init (gen_kern L7 hook); _ON is the module flag
(kd_ext attr knob) and needs a recapture.
"""
import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

ENV = "VLLM_KERN_SGATE"
_ON = os.environ.get(ENV, "0") == "1"
MAX_M = 16
SHAPES = {(1, 2560)}
BK = 512

if triton is not None:

    @triton.jit
    def _rowdot_kernel(X, W, O, M, K, stride_xm, BM: tl.constexpr, BK: tl.constexpr):
        rm = tl.arange(0, BM)
        acc = tl.zeros((BM, BK), tl.float32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            km = rk < K
            x = tl.load(X + rm[:, None] * stride_xm + rk[None, :], (rm[:, None] < M) & km[None, :],
                        other=0.0).to(tl.float32)
            w = tl.load(W + rk, km, other=0.0).to(tl.float32)
            acc += x * w[None, :]
        y = tl.sum(acc, axis=1)
        tl.store(O + rm, y.to(tl.bfloat16), rm < M)


def rowdot(x2: torch.Tensor, w: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """y[M, 1] = x2[M, K] @ w[1, K]^T, FP32 sum, one BF16 rounding."""
    m, k = x2.shape
    if x2.stride(1) != 1:
        x2 = x2.contiguous()
    bm = max(2, triton.next_power_of_2(m))
    if out is None:
        out = torch.empty((m, 1), dtype=torch.bfloat16, device=x2.device)
    _rowdot_kernel[(1,)](x2, w, out, m, k, x2.stride(0), BM=bm, BK=BK, num_warps=4)
    return out


def rowdot_reference(x2: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """FP64 reference, rounded to BF16."""
    return (x2.double() @ w.double().t()).to(torch.bfloat16)


def _usable(x, weight, bias) -> bool:
    return (_ON and bias is None and x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
            and weight.is_cuda and tuple(weight.shape) in SHAPES
            and 0 < x.numel() // x.shape[-1] <= MAX_M)


try:
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod as _Base
except ImportError:  # pragma: no cover
    _Base = object


class SgateLinearMethod(_Base):
    def __init__(self, inner):
        self.inner = inner

    def create_weights(self, *args, **kwargs):
        return self.inner.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer) -> None:
        if hasattr(self.inner, "process_weights_after_loading"):
            self.inner.process_weights_after_loading(layer)

    def apply(self, layer, x, bias=None):
        w = layer.weight
        if not _usable(x, w, bias):
            return self.inner.apply(layer, x, bias)
        lead = x.shape[:-1]
        return rowdot(x.reshape(-1, x.shape[-1]), w).reshape(*lead, 1)


def enable_sgate(model: torch.nn.Module, logger=None) -> list:
    if os.environ.get(ENV, "0") != "1":
        return []
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    names = []
    for name, mod in model.named_modules():
        if not isinstance(mod, LinearBase) or type(getattr(mod, "quant_method", None)) is not UnquantizedLinearMethod:
            continue
        shape = (getattr(mod, "output_size_per_partition", None), getattr(mod, "input_size_per_partition", None))
        if shape not in SHAPES:
            continue
        mod.quant_method = SgateLinearMethod(mod.quant_method)
        names.append(name)
    if logger is not None:
        logger.info("kern L7 shared-expert gate kernel: %d linears", len(names))
    return names
