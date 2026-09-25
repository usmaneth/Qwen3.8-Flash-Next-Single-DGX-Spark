# SPDX-License-Identifier: Apache-2.0
"""L7 S1: a skinny split-K GEMV for the MXFP8 decode linears of the target.

Problem (kern-decode profile TABLE.md, verify graph at M = 7; floors at the
238 GB/s read ceiling of the bw-ceiling study, weights + scales):
  GDN in_proj qkvz   16384 x 2560  204.2 us in the graph, 194.2 us micro, floor 181.7 us
  QSA q/k/v          13312 x 2560  160.5 us, micro 157.9, floor 147.6
  GDN out_proj        2560 x 6144  101.2 us (the R22 state writes inflate it), micro 74.6, floor 68.2
  QSA o_proj          2560 x 6144   80.3 us, micro 74.9, floor 68.2
  shared gate_up      1280 x 2560   23.0 us, micro 18.0, floor 14.2
  shared down         2560 x  640   11.8 us, micro  9.5, floor  7.1
  QSA indexer qk       640 x 2560   13.9 us, micro 11.5, floor  7.1
The image runs these through FlashInfer mm_mxfp8 (CUTLASS, tactic 1, 10-128
CTAs). For M <= MAX_M rows this module runs a Triton GEMV instead.

Kernel. One program reads a BN x BK tile of the FP8 E4M3 weight (16-byte
loads along K) and the matching E8M0 scales straight from the FlashInfer
128x4 swizzled scale tensor (no second copy of the scales). The weight and
the MXFP8 activation are dequantized in registers to BF16: an E4M3 value
times a power of two is exact in BF16. The BF16 products are exact in FP32,
the sum is FP32, split-K partials are summed in a fixed order, and the
result is rounded once to BF16. The rounding points are the ones of the
image path: the activation is the same MXFP8 tensor (the FlashInfer
quantizer, or the in-kernel twin of it when FUSED), and the output is one
BF16 rounding. The accumulation order is not the order of the CUTLASS
block-scaled MMA, so the class is B (the micro gate reports the share of
bitwise-equal outputs against FlashInfer).

Note (2026-09-25 CPU check): Triton 3.7.1 lowers tl.dot_scaled to the
native mxf8f6f4 block-scale MMA only for sm_120; for sm_121 (GB10) it
emits the BF16 emulation. This kernel does the BF16 dequant itself, so the
code path is the same on both.

FUSED = True (S1q) also moves the MXFP8 activation quantizer into the GEMV:
each program quantizes its [M, BK] slice of x with the FlashInfer formula
(cvt_warp_fp16_to_mxfp8: SF = amax * rcp(448), E8M0 rounded to +inf, x *
rcp(SF), E4M3 round-to-nearest with saturation). That removes one launch and
one dependent chain per linear. The FlashInfer rcp is rcp.approx.ftz; the
kernel takes rcp(448) as the argument RCP448 (from the GPU when there is
one) and takes rcp(SF) of a power of two, which is exact above 2^-126.
Exactness of the in-kernel quantizer is a GPU gate (bitwise against the
FlashInfer quantizer on real activations).

Knobs: VLLM_KERN_SKINNY_MX=1 patches the FlashInfer MXFP8 linear kernel at
model init (gen_kern L7 hook); _ON and _FUSED are module flags (kd_ext attr
knobs) and need a recapture.
"""
import os

import numpy as np
import torch

try:  # the CPU tests import this module without a GPU
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

ENV = "VLLM_KERN_SKINNY_MX"
_ON = os.environ.get(ENV, "0") == "1"
_FUSED = os.environ.get("VLLM_KERN_SKINNY_MX_FUSED", "0") == "1"
MAX_M = 16
BLOCK = 32  # MXFP8 block size along K
# (N, K) of the target MXFP8 decode linears (per verify step: calls).
SHAPES = {
    (16384, 2560): 36,  # GDN in_proj qkvz
    (13312, 2560): 12,  # QSA q/k/v
    (2560, 6144): 48,   # GDN out_proj (36) + QSA o_proj (12)
    (1280, 2560): 48,   # shared expert gate_up
    (2560, 640): 48,    # shared expert down
    (640, 2560): 12,    # QSA indexer qk
}
# (N, K) -> (BN, BK, SPLIT, warps, stages). The micro sweep (tools/l7/l7_micro.py)
# fills it; start points come from the bw-ceiling shapes arms (plain_rbX_sY:
# X rows per CTA, Y K splits), a1-spark2-20260925T005729/analysis.txt.
TILES: dict = {
    (16384, 2560): (16, 256, 1, 4, 3),  # plain_rb16_s1 235 GB/s
    (13312, 2560): (128, 256, 1, 8, 3),  # plain_rb128_s1 235 GB/s
    (2560, 6144): (64, 256, 4, 4, 3),   # plain_rb64_s4 230 GB/s
    (1280, 2560): (16, 256, 2, 4, 3),   # plain_rb16_s2 214 GB/s
    (2560, 640): (16, 128, 1, 4, 3),    # plain_rb16_s1 198 GB/s
    (640, 2560): (16, 256, 4, 4, 3),    # plain_rb16_s4 194 GB/s
}
RCP448_EXACT = float(np.float32(1.0) / np.float32(448.0))
_RCP448 = {"v": None}


def swizzled_sf_index(row, kb, num_k_blocks):
    """Byte offset of scale (row, kb) in the FlashInfer 128x4 layout.

    vllm mxfp8_utils.swizzle_mxfp8_scale: view (MT, 4, 32, KT, 4), transpose
    dims 1 and 3 -> (MT, KT, 32, 4, 4). Works on ints, numpy and Triton values.
    """
    kt = (num_k_blocks + 3) // 4
    return (((row // 128) * kt + kb // 4) * 512 + (row % 32) * 16
            + ((row // 32) % 4) * 4 + kb % 4)


if triton is not None:

    @triton.jit
    def _e8m0_to_f32(s):
        # 2^(s - 127) for s in 1..254; s = 0 is 2^-127 (a denormal); 255 is NaN.
        s = s.to(tl.int32)
        norm = (s << 23).to(tl.float32, bitcast=True)
        return tl.where(s == 0, 5.877471754111438e-39, norm)

    @triton.jit
    def _mx_quant(x, RCP448, BM: tl.constexpr, BK: tl.constexpr):
        """The FlashInfer MXFP8 quantizer on a [BM, BK] FP32 tile of BF16 values.

        Returns (q as FP32 values of the E4M3 codes, the E8M0 codes [BM, BK/32]).
        """
        xb = tl.reshape(x, (BM, BK // 32, 32))
        amax = tl.max(tl.abs(xb), axis=2)
        sf = amax * RCP448
        bits = sf.to(tl.int32, bitcast=True)
        e = (bits >> 23) & 0xFF
        mant = bits & 0x7FFFFF
        # __nv_cvt_float_to_e8m0(v, satfinite, round to +inf)
        code = tl.where(mant != 0, e + 1, e)
        code = tl.where((e == 0) & (mant != 0), tl.where(sf > 5.877471754111438e-39, 1, 0), code)
        code = tl.minimum(code, 254)
        code = tl.where(sf != sf, 255, code)
        sfv = _e8m0_to_f32(code)
        # rcp.approx.ftz: a denormal input flushes to 0 -> +inf; exact for 2^-126 .. 2^127
        inv = tl.where(code == 0, float("inf"), 1.0 / sfv)
        inv = tl.where(sfv == 0.0, 0.0, inv)
        q = xb * inv[:, :, None]
        return tl.reshape(_e4m3_rn(q), (BM, BK)), code

    @triton.jit
    def _e4m3_rn(v):
        """cvt.rn.satfinite.e4m3 as FP32 values, with integer and FP32 RN ops only.

        The Triton interpreter converts FP32 -> FP8 by truncation, so the
        kernel rounds by hand; the result is the same on the CPU and the GPU.
        Normal range (|v| >= 2^-6): round the mantissa to 3 bits, ties to even.
        Subnormal range: round to a multiple of 2^-9 with the 1.5 * 2^14 add.
        """
        a = tl.minimum(tl.abs(v), 448.0)
        b = a.to(tl.int32, bitcast=True)
        lsb = (b >> 20) & 1
        nb = (b + 0x7FFFF + lsb) & ~0xFFFFF
        norm = nb.to(tl.float32, bitcast=True)
        sub = (a + 24576.0) - 24576.0
        r = tl.where(a < 0.015625, sub, norm)
        r = tl.where(v < 0, -r, r)
        return tl.where(v != v, v, r)

    @triton.jit
    def _skinny_mx_kernel(X, XS, W, WS, O, P, M, N, K, stride_xm, K_PER, KB_X, KB_W, RCP448,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                          SPLIT: tl.constexpr, STAGES: tl.constexpr, FUSED: tl.constexpr,
                          DOT_F32: tl.constexpr = False):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        rn = pid_n * BN + tl.arange(0, BN)
        rm = tl.arange(0, BM)
        rb = tl.arange(0, BK // 32)
        acc = tl.zeros((BM, BN), tl.float32)
        k_lo = pid_k * K_PER
        k_hi = tl.minimum(k_lo + K_PER, K)
        for k0 in tl.range(k_lo, k_hi, BK, num_stages=STAGES):
            rk = k0 + tl.arange(0, BK)
            km = rk < k_hi
            kbm = (k0 + rb * 32) < k_hi
            kb = k0 // 32 + rb
            if FUSED:
                xv = tl.load(X + rm[:, None] * stride_xm + rk[None, :],
                             (rm[:, None] < M) & km[None, :], other=0.0).to(tl.float32)
                xq, xs = _mx_quant(xv, RCP448, BM, BK)
            else:
                xq = tl.load(X + rm[:, None] * stride_xm + rk[None, :],
                             (rm[:, None] < M) & km[None, :], other=0.0).to(tl.float32)
                xs_off = (((rm[:, None] // 128) * ((KB_X + 3) // 4) + kb[None, :] // 4) * 512
                          + (rm[:, None] % 32) * 16 + ((rm[:, None] // 32) % 4) * 4 + kb[None, :] % 4)
                xs = tl.load(XS + xs_off, (rm[:, None] < M) & kbm[None, :], other=127)
            xsf = _e8m0_to_f32(xs)
            xd = tl.reshape(tl.reshape(xq, (BM, BK // 32, 32)) * xsf[:, :, None], (BM, BK)).to(tl.bfloat16)
            w = tl.load(W + rn[:, None] * K + rk[None, :],
                        (rn[:, None] < N) & km[None, :], other=0.0).to(tl.float32)
            ws_off = (((rn[:, None] // 128) * ((KB_W + 3) // 4) + kb[None, :] // 4) * 512
                      + (rn[:, None] % 32) * 16 + ((rn[:, None] // 32) % 4) * 4 + kb[None, :] % 4)
            ws = tl.load(WS + ws_off, (rn[:, None] < N) & kbm[None, :], other=127)
            wsf = _e8m0_to_f32(ws)
            wd = tl.reshape(tl.reshape(w, (BN, BK // 32, 32)) * wsf[:, :, None], (BN, BK)).to(tl.bfloat16)
            if DOT_F32:
                # CPU tests only: the Triton interpreter computes a BF16 tl.dot
                # wrongly. The operand values are the same (BF16-exact in FP32).
                acc += tl.dot(xd.to(tl.float32), tl.trans(wd.to(tl.float32)), out_dtype=tl.float32)
            else:
                acc += tl.dot(xd, tl.trans(wd), out_dtype=tl.float32)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        if SPLIT == 1:
            tl.store(O + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask)
        else:
            tl.store(P + pid_k * M * N + rm[:, None] * N + rn[None, :], acc, mask)

    @triton.jit
    def _splitk_reduce(P, O, M, N, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
        # Fixed-order sum of the SPLIT partials: the same bits on every run.
        pid = tl.program_id(0)
        idx = pid * BLOCK + tl.arange(0, BLOCK)
        mask = idx < M * N
        acc = tl.zeros((BLOCK,), tl.float32)
        for s in tl.static_range(SPLIT):
            acc += tl.load(P + s * M * N + idx, mask, other=0.0)
        tl.store(O + idx, acc.to(tl.bfloat16), mask)

    @triton.jit
    def _mx_quant_debug_kernel(X, Q, S, M, K, stride_xm, RCP448, BM: tl.constexpr, BK: tl.constexpr):
        # Test helper: the in-kernel quantizer of one [BM, BK] slice per program.
        pid = tl.program_id(0)
        rm = tl.arange(0, BM)
        rk = pid * BK + tl.arange(0, BK)
        m = (rm[:, None] < M) & (rk[None, :] < K)
        x = tl.load(X + rm[:, None] * stride_xm + rk[None, :], m, other=0.0).to(tl.float32)
        q, code = _mx_quant(x, RCP448, BM, BK)
        tl.store(Q + rm[:, None] * K + rk[None, :], q, m)
        rb = pid * (BK // 32) + tl.arange(0, BK // 32)
        tl.store(S + rm[:, None] * (K // 32) + rb[None, :], code.to(tl.uint8),
                 (rm[:, None] < M) & (rb[None, :] < K // 32))

    @triton.jit
    def _rcp_approx_kernel(X, Y):
        x = tl.load(X)
        y = tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=r,r", [x],
                                      dtype=tl.float32, is_pure=True, pack=1)
        tl.store(Y, y)


def rcp448(device) -> float:
    """rcp.approx.ftz(448.0) as the GPU computes it (the FlashInfer constant)."""
    if _RCP448["v"] is None:
        if torch.device(device).type != "cuda" or os.environ.get("TRITON_INTERPRET") == "1":
            return RCP448_EXACT
        x = torch.full((1,), 448.0, dtype=torch.float32, device=device)
        y = torch.empty_like(x)
        _rcp_approx_kernel[(1,)](x, y)
        _RCP448["v"] = float(y.item())
    return _RCP448["v"]


def default_tile(n: int, k: int, sms: int = 48):
    """BN 32, BK 256; split K until 1-2 waves of 48 SMs (bw study: 20-48+ CTAs)."""
    bn, bk = 32, 256
    ctas = -(-n // bn)
    split = 1
    while ctas * split < sms and k // (bk * split * 2) >= 1 and split < 16:
        split *= 2
    return bn, bk, split, 4, 3


def skinny_mx_linear(x: torch.Tensor, w: torch.Tensor, w_sf: torch.Tensor,
                     x_q: torch.Tensor | None = None, x_sf: torch.Tensor | None = None,
                     tile=None, fused: bool | None = None, out: torch.Tensor | None = None,
                     _dot_f32: bool = False) -> torch.Tensor:
    """y[M, N] = MXFP8(x)[M, K] @ MXFP8(w)[N, K]^T, one BF16 rounding.

    w: [N, K] float8_e4m3fn; w_sf: the swizzled E8M0 scales (uint8, flat).
    Unfused: x_q [M, K] float8_e4m3fn and x_sf (swizzled, flat) from the
    FlashInfer quantizer. Fused: x is the BF16 activation.
    """
    fused = _FUSED if fused is None else fused
    m, k = x.shape
    n = w.shape[0]
    assert k % BLOCK == 0
    bn, bk, split, warps, stages = tile or TILES.get((n, k)) or default_tile(n, k)
    assert bk % BLOCK == 0
    bm = max(16, triton.next_power_of_2(m))
    k_per = -(-k // split)
    k_per = -(-k_per // bk) * bk
    split = -(-k // k_per)
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    part = torch.empty((split, m, n), dtype=torch.float32, device=x.device) if split > 1 else out
    if fused:
        xa, xs, sxm = x, w_sf, x.stride(0)
    else:
        assert x_q is not None and x_sf is not None
        xa, xs, sxm = x_q, x_sf, x_q.stride(0)
    grid = (triton.cdiv(n, bn), split)
    _skinny_mx_kernel[grid](xa, xs, w, w_sf, out, part, m, n, k, sxm, k_per, k // BLOCK, k // BLOCK,
                            rcp448(x.device), BM=bm, BN=bn, BK=bk, SPLIT=split, STAGES=stages,
                            FUSED=bool(fused), DOT_F32=_dot_f32, num_warps=warps)
    if split > 1:
        blk = 1024
        _splitk_reduce[(triton.cdiv(m * n, blk),)](part, out, m, n, SPLIT=split, BLOCK=blk)
    return out


def mx_quant_rows(x: torch.Tensor, bk: int = 256):
    """The in-kernel quantizer as a separate launch: (E4M3 values as FP32 [M, K], codes [M, K/32])."""
    m, k = x.shape
    q = torch.empty((m, k), dtype=torch.float32, device=x.device)
    s = torch.empty((m, k // BLOCK), dtype=torch.uint8, device=x.device)
    bm = max(16, triton.next_power_of_2(m))
    _mx_quant_debug_kernel[(triton.cdiv(k, bk),)](x, q, s, m, k, x.stride(0), rcp448(x.device), BM=bm, BK=bk)
    return q, s


# ------------------------------------------------------------------ CPU twins
def e8m0_to_f32_np(s):
    s = np.asarray(s).astype(np.int64)
    v = np.ldexp(np.float64(1.0), s - 127).astype(np.float32)
    return np.where(s == 255, np.float32(np.nan), v)


def mx_quant_twin(x_bf16_f32: np.ndarray, rcp=RCP448_EXACT):
    """numpy twin of _mx_quant (and of cvt_warp_fp16_to_mxfp8) on [M, K] FP32 values."""
    import ml_dtypes

    m, k = x_bf16_f32.shape
    xb = x_bf16_f32.reshape(m, k // 32, 32).astype(np.float32)
    amax = np.abs(xb).max(axis=2)
    sf = (amax * np.float32(rcp)).astype(np.float32)
    bits = sf.view(np.int32)
    e = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    code = np.where(mant != 0, e + 1, e)
    den = (e == 0) & (mant != 0)
    code = np.where(den, np.where(sf > np.float32(5.877471754111438e-39), 1, 0), code)
    code = np.minimum(code, 254)
    code = np.where(np.isnan(sf), 255, code)
    sfv = e8m0_to_f32_np(code)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inv = np.where(code == 0, np.float32(np.inf), (np.float32(1.0) / sfv).astype(np.float32))
        inv = np.where(sfv == 0, np.float32(0), inv).astype(np.float32)
        q = (xb * inv[:, :, None]).astype(np.float32)
    q = np.clip(q, -448.0, 448.0)
    q8 = q.astype(ml_dtypes.float8_e4m3fn)
    return q8.reshape(m, k), code.astype(np.uint8)


def swizzle_np(sf2d: np.ndarray) -> np.ndarray:
    """numpy copy of vllm swizzle_mxfp8_scale (row-major [R, K/32] -> flat 128x4)."""
    r, kb = sf2d.shape
    mt, kt = -(-r // 128), -(-kb // 4)
    pad = np.zeros((mt * 128, kt * 4), sf2d.dtype)
    pad[:r, :kb] = sf2d
    return pad.reshape(mt, 4, 32, kt, 4).transpose(0, 3, 2, 1, 4).reshape(-1)


def reference_np(x_q, x_sf2d, w_q, w_sf2d):
    """FP64 reference of the dequantized MXFP8 product, [M, N]."""
    xd = x_q.astype(np.float64) * np.repeat(e8m0_to_f32_np(x_sf2d).astype(np.float64), 32, axis=1)
    wd = w_q.astype(np.float64) * np.repeat(e8m0_to_f32_np(w_sf2d).astype(np.float64), 32, axis=1)
    return xd @ wd.T


# ------------------------------------------------------------------ the hook
def _usable(x, weight, bias) -> bool:
    return (_ON and bias is None and x.dtype == torch.bfloat16 and weight.is_cuda
            and tuple(weight.shape) in SHAPES and 0 < x.numel() // x.shape[-1] <= MAX_M)


def patch_flashinfer_mxfp8(logger=None) -> bool:
    """Route the FlashInfer CUTLASS MXFP8 linear kernel through the GEMV for M <= MAX_M."""
    try:
        from vllm.model_executor.kernels.linear.mxfp8 import flashinfer as fimod
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    except ImportError:
        return False
    cls = fimod.FlashInferCutlassMxfp8LinearKernel
    if getattr(cls, "_kern_skinny_mx", False):
        return True
    orig = cls.apply_weights

    def apply_weights(self, layer, x, bias=None):
        w = layer.weight
        if not _usable(x, w, bias):
            return orig(self, layer, x, bias)
        n, k = w.shape
        lead = x.shape[:-1]
        x2 = x.reshape(-1, k)
        if _FUSED:
            y = skinny_mx_linear(x2, w, layer.weight_scale, fused=True)
        else:
            xq, xs = mxfp8_e4m3_quantize(x2, is_sf_swizzled_layout=True)
            y = skinny_mx_linear(x2, w, layer.weight_scale, x_q=xq, x_sf=xs, fused=False)
        return y.reshape(*lead, n)

    cls.apply_weights = apply_weights
    cls._kern_skinny_mx = True
    if logger is not None:
        logger.info("kern L7 skinny MXFP8 GEMV: FlashInfer MXFP8 linear routed for M <= %d (fused quant %s)",
                    MAX_M, _FUSED)
    return True
