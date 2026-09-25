# SPDX-License-Identifier: Apache-2.0
"""R4: weight-only FP8 (W8A16) for the dense BF16 linears of the MTP drafter.

Problem (kern-decode profile, K=6): the MTP dense linears (q/k/v, o, the QSA
indexer, fc_embedding, fc_hidden, the 3 HC down/up pairs, the shared expert
and the router) read 181.5 MB of BF16 weights per draft pass, 1.09 GB per
step, in 6.5 ms. They are GEMVs (M = 1 in the 1-token passes, M = 7 or 28
in pass 0), so their time is the weight bytes.

Change. After the MTP weights load, each of these linears keeps its BF16
weight and gets a per-output-row E4M3 copy with an FP32 row scale (half the
bytes). For M <= MAX_M rows the linear runs this module's Triton kernel:
FP8 weights, BF16 activations, the dequantization in registers, exact
products and FP32 accumulation (tl.dot), the row scale in FP32, one BF16
rounding. More rows (the drafter prefill) and _ON = False use the BF16
weight (the image path). The drafter output is a draft token only: the
target verifies every draft token, so the served output does not depend on
this change beyond the verify-width effects of other draft tokens; the
acceptance rate can move.

Kernel. Adapted from the W8A16 draft-head kernel of MiaAI-Lab recipe PR #31
(files/spark_mtp_fp8_head.py, Raymond Lucke), which adapts Gabriel Olympie's
(@gabrielolympie) sglang-flashnext-sm120 patches 0004/0005, commit
67d2f9234fa45ae1339f0d53cd37cb695e9c6493, Apache-2.0. This version adds a
split-K form with a fixed-order reduction (deterministic, no atomics) for
the small-N shapes, and a tile table per (N, K).

Knobs:
  VLLM_MTP_DENSE_W8A16=1   build the FP8 copies at load (mtp.py calls
                           enable_mtp_w8a16 after load_weights).
  _ON (module flag)        use the FP8 path (default: the env value). The
                           kd_ext attr knob sets it at run time; graphs bake
                           the path in, so a change needs a recapture.
"""
import os

import torch

try:  # the CPU tests import this module without triton
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

ENV = "VLLM_MTP_DENSE_W8A16"
_ON = os.environ.get(ENV, "0") == "1"
# R17: VLLM_MTP_DENSE_W4A16=1 also builds a 4-bit copy (w4a16.py, 0.53 bytes
# per weight). _W4_ON picks it over the FP8 copy (kd_ext knob mtp_w4).
ENV4 = "VLLM_MTP_DENSE_W4A16"
_BUILD_W4 = os.environ.get(ENV4, "0") == "1"
_W4_ON = _BUILD_W4
MAX_M = 32
FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
MIN_N = 64  # smaller outputs (the shared-expert gate, 1 x 2560) stay BF16

# (N, K) -> (BN, BK, SPLIT, warps, stages). Filled by the tile sweep
# (tools/w8a16_bench.py); a shape that is not here uses default_tile().
# Best M=1 tiles from the kern-micro-1 sweep (spark2, 2026-09-24,
# runs/kern-micro-1-spark2-20260924T214504/w8a16.json, cold weights).
TILES: dict = {
    (320, 10240): (32, 256, 4, 4, 3),
    (336, 10240): (32, 256, 4, 4, 3),
    (512, 2560): (64, 128, 4, 8, 3),
    (640, 2560): (32, 256, 1, 4, 3),
    (1280, 2560): (32, 256, 1, 4, 3),
    (2560, 640): (32, 128, 1, 4, 3),
    (2560, 2560): (32, 256, 1, 4, 3),
    (2560, 6144): (64, 256, 1, 8, 3),
    (10240, 320): (16, 128, 1, 4, 3),
    (13312, 2560): (16, 256, 1, 4, 3),
    (47184, 2560): (64, 256, 1, 8, 3),
}


def default_tile(n: int, k: int, sms: int = 48):
    """BN 32 and BK 256 (the PR #31 tactic); split K until 2 CTAs per SM."""
    bn, bk = 32, 256
    ctas = -(-n // bn)
    split = 1
    while ctas * split < 2 * sms and k // (bk * split * 2) >= 1 and split < 16:
        split *= 2
    return bn, bk, split, 4, 3


if triton is not None:

    @triton.jit
    def _w8a16_kernel(X, W, S, O, P, M, N, K, stride_xm, K_PER,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                      SPLIT: tl.constexpr, STAGES: tl.constexpr):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        rn = pid_n * BN + tl.arange(0, BN)
        rm = tl.arange(0, BM)
        acc = tl.zeros((BM, BN), tl.float32)
        k_lo = pid_k * K_PER
        k_hi = tl.minimum(k_lo + K_PER, K)
        for k0 in tl.range(k_lo, k_hi, BK, num_stages=STAGES):
            rk = k0 + tl.arange(0, BK)
            km = rk < k_hi
            x = tl.load(X + rm[:, None] * stride_xm + rk[None, :],
                        (rm[:, None] < M) & km[None, :], other=0.0)
            w = tl.load(W + rn[:, None] * K + rk[None, :],
                        (rn[:, None] < N) & km[None, :], other=0.0).to(tl.bfloat16)
            acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        if SPLIT == 1:
            s = tl.load(S + rn, rn < N, other=0.0)
            out = (acc * s[None, :]).to(tl.bfloat16)
            tl.store(O + rm[:, None] * N + rn[None, :], out, mask)
        else:
            tl.store(P + pid_k * M * N + rm[:, None] * N + rn[None, :], acc, mask)

    @triton.jit
    def _splitk_reduce(P, S, O, M, N, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
        # Fixed-order sum of the SPLIT partials: the same bits on every run.
        pid = tl.program_id(0)
        idx = pid * BLOCK + tl.arange(0, BLOCK)
        mask = idx < M * N
        acc = tl.zeros((BLOCK,), tl.float32)
        for s in tl.static_range(SPLIT):
            acc += tl.load(P + s * M * N + idx, mask, other=0.0)
        n = idx % N
        sc = tl.load(S + n, mask, other=0.0)
        tl.store(O + idx, (acc * sc).to(tl.bfloat16), mask)


if triton is not None:

    @triton.jit
    def _gemma_rmsnorm_kernel(X, W, Y, N, stride, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        m = offs < N
        x = tl.load(X + row * stride + offs, m, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / N
        w = tl.load(W + offs, m, other=0.0).to(tl.float32) + 1.0
        y = (x * tl.math.rsqrt(var + eps)) * w
        tl.store(Y + row * N + offs, y.to(tl.bfloat16), m)


# R14 part: the two GemmaRMSNorms of the MTP input (pre_fc_norm_embedding,
# pre_fc_norm_hidden) run as about 10 torch kernels each per pass (the
# weight + 1 in FP32, then the native RMS norm). VLLM_MTP_FUSED_NORM=1 runs
# one Triton kernel: FP32 sum of squares (another order than torch), the
# same FP32 ops, one BF16 rounding. Draft-only.
NORM_ENV = "VLLM_MTP_FUSED_NORM"
_NORM_ON = os.environ.get(NORM_ENV, "0") == "1"


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.stride(1) != 1:
        x2 = x2.contiguous()
    n = x2.shape[1]
    y = torch.empty((x2.shape[0], n), dtype=torch.bfloat16, device=x.device)
    _gemma_rmsnorm_kernel[(x2.shape[0],)](x2, weight, y, n, x2.stride(0), eps,
                                          BLOCK=triton.next_power_of_2(n), num_warps=8)
    return y.reshape(shape)


def _wrap_norm(mod) -> None:
    orig = mod.forward

    def forward(x, residual=None, _orig=orig, _mod=mod):
        if (not _NORM_ON or residual is not None or x.dtype != torch.bfloat16 or not x.is_cuda
                or x.numel() // x.shape[-1] > MAX_M * 4):
            return _orig(x, residual) if residual is not None else _orig(x)
        return gemma_rmsnorm(x, _mod.weight, _mod.variance_epsilon)

    mod.forward = forward


def w8a16_linear(x: torch.Tensor, w8: torch.Tensor, scale: torch.Tensor,
                 tile=None) -> torch.Tensor:
    """y[M, N] = x[M, K] @ (w8 * scale)^T in FP32, rounded once to BF16."""
    m, k = x.shape
    n = w8.shape[0]
    if x.stride(1) != 1:
        x = x.contiguous()
    bn, bk, split, warps, stages = tile or TILES.get((n, k)) or default_tile(n, k)
    bm = max(16, triton.next_power_of_2(m))
    k_per = -(-k // split)
    k_per = -(-k_per // bk) * bk
    split = -(-k // k_per)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    part = torch.empty((split, m, n), dtype=torch.float32, device=x.device) if split > 1 else out
    grid = (triton.cdiv(n, bn), split)
    _w8a16_kernel[grid](x, w8, scale, out, part, m, n, k, x.stride(0), k_per,
                        BM=bm, BN=bn, BK=bk, SPLIT=split, STAGES=stages, num_warps=warps)
    if split > 1:
        block = 1024
        _splitk_reduce[(triton.cdiv(m * n, block),)](part, scale, out, m, n, SPLIT=split, BLOCK=block)
    return out


def w8a16_reference(x: torch.Tensor, w8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The same math in FP32 on any device (products are exact in FP32)."""
    acc = x.float() @ w8.float().t()
    return (acc * scale.float()[None, :]).to(torch.bfloat16)


@torch.no_grad()
def quantize_rows(weight: torch.Tensor, chunk_rows: int = 4096):
    """Per-output-row E4M3 weights and FP32 scales (amax / 448)."""
    n, k = weight.shape
    w8 = torch.empty((n, k), dtype=FP8, device=weight.device)
    scale = torch.empty((n,), dtype=torch.float32, device=weight.device)
    for r0 in range(0, n, chunk_rows):
        wf = weight[r0:r0 + chunk_rows].float()
        s = (wf.abs().amax(dim=1) / FP8_MAX).clamp_min(1e-12)
        w8[r0:r0 + chunk_rows] = (wf / s[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
        scale[r0:r0 + chunk_rows] = s
    return w8, scale


try:  # QuantizeMethodBase: the model loader post-processes only those
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod as _Base
except ImportError:  # pragma: no cover  (CPU tests without vllm)
    _Base = object


class MtpW8A16Method(_Base):
    """Quant method: FP8 kernel for M <= MAX_M, else the inner BF16 method."""

    def __init__(self, inner, name: str):
        self.inner = inner
        self.name = name
        self.w8 = None
        self.scale = None
        self.w4 = None

    def create_weights(self, *args, **kwargs):
        return self.inner.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer) -> None:
        if hasattr(self.inner, "process_weights_after_loading"):
            self.inner.process_weights_after_loading(layer)
        self.build(layer)

    def build(self, layer) -> None:
        w = getattr(layer, "weight", None)
        if w is None or w.dim() != 2 or w.dtype != torch.bfloat16 or w.device.type != "cuda":
            return
        self.w8, self.scale = quantize_rows(w.data)
        if _BUILD_W4 and w.shape[1] % 32 == 0:
            from .w4a16 import quantize_w4

            self.w4 = quantize_w4(w.data)

    def apply(self, layer, x: torch.Tensor, bias=None):
        if not _ON or self.w8 is None or bias is not None or x.dtype != torch.bfloat16:
            return self.inner.apply(layer, x, bias)
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        if x2.shape[0] == 0 or x2.shape[0] > MAX_M:
            return self.inner.apply(layer, x, bias)
        if _W4_ON and self.w4 is not None:
            from .w4a16 import w4a16_linear

            return w4a16_linear(x2, *self.w4).reshape(*lead, self.w8.shape[0])
        return w8a16_linear(x2, self.w8, self.scale).reshape(*lead, self.w8.shape[0])


def enable_mtp_w8a16(model: torch.nn.Module, logger=None) -> list:
    """Wrap the BF16 unquantized linears of the MTP model. Returns their names.

    Called at the end of Qwen3_8FlashNextMTP.load_weights. The model loader
    then calls process_weights_after_loading, which builds the FP8 copies.
    """
    if os.environ.get(ENV, "0") != "1":
        return []
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    names = []
    for name, mod in model.named_modules():
        if not isinstance(mod, LinearBase):
            continue
        qm = getattr(mod, "quant_method", None)
        if type(qm) is not UnquantizedLinearMethod:
            continue
        w = getattr(mod, "weight", None)
        if w is None or w.dim() != 2 or w.dtype != torch.bfloat16 or w.shape[0] < MIN_N:
            continue
        if getattr(mod, "tp_size", 1) != 1:
            continue
        mod.quant_method = MtpW8A16Method(qm, name)
        names.append(f"{name}{tuple(w.shape)}")
    if os.environ.get(NORM_ENV, "0") == "1":
        inner = getattr(model, "model", model)
        for nm in ("pre_fc_norm_embedding", "pre_fc_norm_hidden"):
            if hasattr(inner, nm):
                _wrap_norm(getattr(inner, nm))
                names.append(nm)
    if logger is not None:
        logger.info("MTP W8A16: %d dense linears get FP8 row-scaled copies: %s",
                    len(names), ", ".join(names))
    return names
