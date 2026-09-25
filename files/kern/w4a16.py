# SPDX-License-Identifier: Apache-2.0
"""R17 / R8: weight-only 4-bit (W4A16) GEMV for the MTP drafter.

Format (per output row n, groups of G = 32 along K):
  q[n, k]   4-bit integer in [-8, 7], stored as q + 8, two per byte
            (low nibble = even k, high nibble = odd k)
  gs[n, g]  E4M3 group scale
  rs[n]     FP32 row scale
  w[n, k] = rs[n] * gs[n, k // G] * q[n, k]
Bytes: 0.5 + 1/32 per weight (0.53), against 1.0 for the R4 FP8 copy and 2.0
for BF16.

Kernel: the dequantization runs in registers. (q) x (gs) is exact in BF16
(a 4-bit integer times a 4-bit-mantissa E4M3 value fits the 8-bit BF16
mantissa), so tl.dot gets exact BF16 weights, the products are exact and
the sum is FP32; the row scale is applied in FP32 and the result is rounded
once to BF16. Split-K uses a fixed-order reduction (no atomics), as in
mtp_w8a16.py.

The drafter output is a draft token only (the target verifies it). The gate
is the acceptance rate (tokens per step).
"""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

G = 32
FP8 = torch.float8_e4m3fn


@torch.no_grad()
def quantize_w4(weight: torch.Tensor, chunk_rows: int = 2048):
    """BF16 [N, K] -> (packed uint8 [N, K/2], gs e4m3 [N, K/G], rs fp32 [N])."""
    n, k = weight.shape
    assert k % G == 0, (n, k)
    packed = torch.empty((n, k // 2), dtype=torch.uint8, device=weight.device)
    gs_out = torch.empty((n, k // G), dtype=FP8, device=weight.device)
    rs_out = torch.empty((n,), dtype=torch.float32, device=weight.device)
    for r0 in range(0, n, chunk_rows):
        wf = weight[r0:r0 + chunk_rows].float()
        rows = wf.shape[0]
        wg = wf.view(rows, k // G, G)
        amax = wg.abs().amax(dim=-1)                       # [rows, K/G]
        want = (amax / 7.0).clamp_min(1e-20)               # ideal group step
        rs = (want.amax(dim=-1) / 448.0).clamp_min(1e-30)  # row scale
        gs = (want / rs[:, None]).clamp(max=448.0).to(FP8)
        # A group scale that rounds to 0 (a tiny group) gets the smallest
        # E4M3 value so the division below stays finite.
        tiny = torch.tensor(2.0 ** -9, device=weight.device).to(FP8)
        gs = torch.where(gs.float() == 0, tiny, gs)
        step = rs[:, None] * gs.float()                     # [rows, K/G]
        q = torch.round(wg / step[..., None]).clamp(-8, 7).to(torch.int16) + 8
        q = q.view(rows, k).to(torch.uint8)
        packed[r0:r0 + rows] = q[:, 0::2] | (q[:, 1::2] << 4)
        gs_out[r0:r0 + rows] = gs
        rs_out[r0:r0 + rows] = rs
    return packed, gs_out, rs_out


def dequant_w4(packed, gs, rs) -> torch.Tensor:
    """FP32 [N, K] weights of the format (the reference)."""
    n = packed.shape[0]
    lo = (packed & 0xF).to(torch.int16) - 8
    hi = (packed >> 4).to(torch.int16) - 8
    q = torch.stack((lo, hi), dim=-1).view(n, -1).float()
    k = q.shape[1]
    return q.view(n, k // G, G) * gs.float()[..., None] * rs[:, None, None].float()


def w4a16_reference(x, packed, gs, rs) -> torch.Tensor:
    """x @ w^T with exact BF16 weights (q * gs), FP32 sums, row scale, BF16."""
    n = packed.shape[0]
    lo = (packed & 0xF).to(torch.int16) - 8
    hi = (packed >> 4).to(torch.int16) - 8
    q = torch.stack((lo, hi), dim=-1).view(n, -1).float()
    k = q.shape[1]
    wq = (q.view(n, k // G, G) * gs.float()[..., None]).view(n, k)
    acc = x.float() @ wq.t()
    return (acc * rs.float()[None, :]).to(torch.bfloat16)


def default_tile(n: int, k: int, sms: int = 48):
    bn, bk = 32, 256
    ctas = -(-n // bn)
    split = 1
    while ctas * split < 2 * sms and k // (bk * split * 2) >= 1 and split < 16:
        split *= 2
    return bn, bk, split, 4, 3


TILES: dict = {}

if triton is not None:

    @triton.jit
    def _w4a16_kernel(X, Wp, GS, RS, O, P, M, N, K, stride_xm, K_PER,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                      GSZ: tl.constexpr, SPLIT: tl.constexpr, STAGES: tl.constexpr):
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
            rkp = k0 // 2 + tl.arange(0, BK // 2)
            p = tl.load(Wp + rn[:, None] * (K // 2) + rkp[None, :],
                        (rn[:, None] < N) & (rkp[None, :] < k_hi // 2), other=0x88)
            lo = (p & 0xF).to(tl.int32) - 8
            hi = (p >> 4).to(tl.int32) - 8
            q = tl.reshape(tl.join(lo, hi), (BN, BK))
            rg = k0 // GSZ + tl.arange(0, BK // GSZ)
            g = tl.load(GS + rn[:, None] * (K // GSZ) + rg[None, :],
                        (rn[:, None] < N) & (rg[None, :] < K // GSZ), other=0.0).to(tl.float32)
            w = tl.reshape(tl.reshape(q.to(tl.float32), (BN, BK // GSZ, GSZ)) * g[:, :, None], (BN, BK))
            acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)), out_dtype=tl.float32)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        if SPLIT == 1:
            s = tl.load(RS + rn, rn < N, other=0.0)
            tl.store(O + rm[:, None] * N + rn[None, :], (acc * s[None, :]).to(tl.bfloat16), mask)
        else:
            tl.store(P + pid_k * M * N + rm[:, None] * N + rn[None, :], acc, mask)

    @triton.jit
    def _w4_reduce(P, S, O, M, N, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        idx = pid * BLOCK + tl.arange(0, BLOCK)
        mask = idx < M * N
        acc = tl.zeros((BLOCK,), tl.float32)
        for s in tl.static_range(SPLIT):
            acc += tl.load(P + s * M * N + idx, mask, other=0.0)
        sc = tl.load(S + idx % N, mask, other=0.0)
        tl.store(O + idx, (acc * sc).to(tl.bfloat16), mask)


def w4a16_linear(x, packed, gs, rs, tile=None) -> torch.Tensor:
    m, k = x.shape
    n = packed.shape[0]
    if x.stride(1) != 1:
        x = x.contiguous()
    bn, bk, split, warps, stages = tile or TILES.get((n, k)) or default_tile(n, k)
    bk = min(bk, max(G, triton.next_power_of_2(k)))
    bm = max(16, triton.next_power_of_2(m))
    k_per = -(-k // split)
    k_per = -(-k_per // bk) * bk
    split = -(-k // k_per)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    part = torch.empty((split, m, n), dtype=torch.float32, device=x.device) if split > 1 else out
    _w4a16_kernel[(triton.cdiv(n, bn), split)](
        x, packed, gs, rs, out, part, m, n, k, x.stride(0), k_per,
        BM=bm, BN=bn, BK=bk, GSZ=G, SPLIT=split, STAGES=stages, num_warps=warps)
    if split > 1:
        _w4_reduce[(triton.cdiv(m * n, 1024),)](part, rs, out, m, n, SPLIT=split, BLOCK=1024)
    return out
