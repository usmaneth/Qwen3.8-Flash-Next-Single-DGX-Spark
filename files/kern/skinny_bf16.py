# SPDX-License-Identifier: Apache-2.0
"""R6: a skinny BF16 GEMM for the small-N decode linears of the target.

Problem (kern-decode profile, verify graph at M = 7): cuBLAS runs these
linears with 16x16 wmma tiles and a split-K reduce kernel:
  router gate      512 x 2560   19.5 us (floor 12.2 us), on the critical path
  GDN in_proj_ba    96 x 2560   18.3 us (floor 2.3 us, grid 8x1, 27 GB/s)
  HC down+inject   336 x 10240  41.2 us (floor 32.0 us)
  HC mixer down    320 x 10240
  HC up          10240 x 320    35.5 us (floor 30.5 us)

Change. For M <= MAX_M rows these linears run the Triton GEMV of
mtp_w8a16.py on their BF16 weights with a row scale of 1.0 (the multiply by
1.0 is exact): BF16 products are exact in FP32, the sum is FP32 in a fixed
order (split-K with a fixed-order reduction), and the result is rounded once
to BF16, as cuBLAS does. The accumulation order differs from cuBLAS, so the
last bit of an output can differ (class B). More rows (prefill) and
_ON = False keep the image path. The GDN ba weight is MXFP8 in the
checkpoint; the recipe dequantizes it to BF16 at load (the MXFP8 emulation
kernel), and this module routes that kernel's BF16 linear.

Knobs: VLLM_KERN_SKINNY=1 (model.py calls enable_skinny); _ON (module
flag, kd_ext attr knob) with a recapture.
"""
import os

import torch

ENV = "VLLM_KERN_SKINNY"
_ON = os.environ.get(ENV, "0") == "1"
MAX_M = 16
SHAPES = {(512, 2560), (96, 2560), (336, 10240), (320, 10240), (10240, 320)}
# Tiles for the BF16 weights (2 bytes per weight); kern-micro sweep.
TILES: dict = {}
_ONES: dict = {}


def _ones(n: int, device) -> torch.Tensor:
    key = (n, str(device))
    t = _ONES.get(key)
    if t is None:
        t = torch.ones(n, dtype=torch.float32, device=device)
        _ONES[key] = t
    return t


def skinny_linear(x2: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from .mtp_w8a16 import default_tile, w8a16_linear

    n, k = weight.shape
    tile = TILES.get((n, k)) or default_tile(n, k)
    return w8a16_linear(x2, weight, _ones(n, weight.device), tile=tile)


def _usable(x: torch.Tensor, weight: torch.Tensor, bias) -> bool:
    return (_ON and bias is None and x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
            and weight.is_cuda and tuple(weight.shape) in SHAPES
            and 0 < x.numel() // x.shape[-1] <= MAX_M)


try:
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod as _Base
except ImportError:  # pragma: no cover
    _Base = object


class SkinnyLinearMethod(_Base):
    def __init__(self, inner):
        self.inner = inner

    def create_weights(self, *args, **kwargs):
        return self.inner.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer) -> None:
        if hasattr(self.inner, "process_weights_after_loading"):
            self.inner.process_weights_after_loading(layer)
        w = getattr(layer, "weight", None)
        if w is not None and w.is_cuda:
            _ones(w.shape[0], w.device)

    def apply(self, layer, x, bias=None):
        w = layer.weight
        if not _usable(x, w, bias):
            return self.inner.apply(layer, x, bias)
        lead = x.shape[:-1]
        return skinny_linear(x.reshape(-1, x.shape[-1]), w).reshape(*lead, w.shape[0])


def _patch_mxfp8_emulation(logger=None) -> bool:
    """Route the BF16 linear of the MXFP8 emulation kernel (GDN ba)."""
    try:
        from vllm.model_executor.kernels.linear.mxfp8 import emulation as emu
    except ImportError:
        return False
    cls = emu.EmulationMxfp8LinearKernel
    if getattr(cls, "_kern_skinny", False):
        return True
    orig = cls.apply_weights

    def apply_weights(self, layer, x, bias=None):
        w = layer.weight
        if w.element_size() >= 2 and _usable(x, w, bias):
            lead = x.shape[:-1]
            return skinny_linear(x.reshape(-1, x.shape[-1]), w).reshape(*lead, w.shape[0])
        return orig(self, layer, x, bias)

    cls.apply_weights = apply_weights
    cls._kern_skinny = True
    return True


def enable_skinny(model: torch.nn.Module, logger=None) -> list:
    """Swap the matching unquantized linears of the target model."""
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
        mod.quant_method = SkinnyLinearMethod(mod.quant_method)
        names.append(name)
    emu = _patch_mxfp8_emulation(logger)
    if logger is None:
        from vllm.logger import init_logger

        logger = init_logger(__name__)
    if logger is not None:
        logger.info("kern skinny BF16 GEMM: %d linears (%s ...), MXFP8 emulation routed: %s",
                    len(names), ", ".join(names[:4]), emu)
    return names
