# SPDX-License-Identifier: Apache-2.0
"""PLE on the device (TECHNIQUES.md #7): the in-graph n-gram id kernel and the ATS row gather.

Status: draft for the micro lease. No GPU ran this file yet. The CPU twin
(ple_gpu_twin.py) copies each statement of the two kernels below, and
parity_real.py proves the twin bit-exact against the CPU worker
(forward_impl of files/ple_layer_patched.py) and the B3 numpy path.

Inputs are the persistent runner buffers that the CUDA graph already reads.
All of them are outside the graph pool:
  ids   int32 [T_pad]        input_ids (the sampled token and the drafts)
  qsl   int32 [R_pad + 1]    query_start_loc; the padded entries repeat the last
  nctx  int32 [R_pad, N-1]   ngram_context; column N-2 is the newest token
Constant buffers of the layer (int64): mult [N], sizes [H], offs [H].
Output: row ids int64 [T_pad, H], in the forward_impl order (the 2-gram heads,
then the 3-gram heads), and rows uint8 [T_pad, H * 90].

The rules that make the kernel equal to forward_impl (N = 3):
  1. Request of token t: r = (number of qsl entries <= t) - 1, clamped to
     R_pad - 1. This is searchsorted(right=True) - 1, so an empty request
     never owns a token.
  2. The valid tokens are t < qsl[R_pad]. A padding token (t >= qsl[R_pad])
     gets the ids of the all-EOS trigram. forward_impl gives a padding token
     the ids of a clamped real position; no real token reads a padding row
     (gate T-pad in PLAN.md checks this on the GPU).
  3. Column c = t - qsl[r]. The token s places back is ids[t - s] when
     c >= s, else nctx[r, N - 1 - (s - c)].
  4. EOS rule of _shift_precompute/_shift_apply: the token s places back
     becomes EOS when an EOS is at any of the positions 1..s-1 places back.
     (An EOS at s places back is EOS already.) For N = 3: g1 = w1,
     g2 = EOS if w1 == EOS else w2.
  5. All products and XORs are int64. ids load as int32: cast before the
     multiply. mult < (2^63 - 1) / vocab, so a product of a token in
     [0, vocab) cannot wrap, and the XOR of such products is >= 0.
  6. Triton's % on signed integers is C remainder (the sign follows the
     dividend). torch.remainder follows the divisor. The kernel adds the
     divisor when the remainder is negative. This only changes the result
     for a negative token id (for example a -1 placeholder).
  7. The byte offset of a row is id * 90 in int64 (ids go up to
     320,001,445, so id * 90 is above 2^31).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _ple_ids_kernel(ids_ptr, qsl_ptr, nctx_ptr, mult_ptr, sizes_ptr, offs_ptr,
                    out_ptr, R_PAD: tl.constexpr, EOS: tl.constexpr,
                    HPN: tl.constexpr, BLOCK_R: tl.constexpr):
    t = tl.program_id(0)
    # Rule 1: count the qsl entries <= t. BLOCK_R >= R_PAD + 1.
    rr = tl.arange(0, BLOCK_R)
    q = tl.load(qsl_ptr + rr, mask=rr <= R_PAD, other=2147483647)
    r = tl.sum((q <= t).to(tl.int32), axis=0) - 1
    r = tl.minimum(r, R_PAD - 1)
    q_end = tl.load(qsl_ptr + R_PAD)
    q_r = tl.load(qsl_ptr + r)
    c = t - q_r
    valid = t < q_end
    # Rule 3 (N = 3): the token itself and the two tokens before it.
    w0 = tl.load(ids_ptr + t).to(tl.int64)
    w1_in = tl.load(ids_ptr + t - 1, mask=c >= 1, other=0).to(tl.int64)
    w1_cx = tl.load(nctx_ptr + r * 2 + 1).to(tl.int64)
    w1 = tl.where(c >= 1, w1_in, w1_cx)
    w2_in = tl.load(ids_ptr + t - 2, mask=c >= 2, other=0).to(tl.int64)
    w2_cx = tl.load(nctx_ptr + r * 2 + tl.where(c == 1, 1, 0)).to(tl.int64)
    w2 = tl.where(c >= 2, w2_in, w2_cx)
    # Rule 2: padding tokens get the all-EOS trigram.
    w0 = tl.where(valid, w0, EOS)
    w1 = tl.where(valid, w1, EOS)
    w2 = tl.where(valid, w2, EOS)
    # Rule 4: the EOS rule.
    g2 = tl.where(w1 == EOS, EOS, w2)
    # Rule 5: int64 products.
    m0 = tl.load(mult_ptr + 0)
    m1 = tl.load(mult_ptr + 1)
    m2 = tl.load(mult_ptr + 2)
    mix2 = (w0 * m0) ^ (w1 * m1)
    mix3 = mix2 ^ (g2 * m2)
    hh = tl.arange(0, HPN)
    for g in tl.static_range(2):
        mix = mix2 if g == 0 else mix3
        size = tl.load(sizes_ptr + g * HPN + hh)
        off = tl.load(offs_ptr + g * HPN + hh)
        rem = mix % size                      # C remainder (rule 6)
        rem = tl.where(rem < 0, rem + size, rem)
        tl.store(out_ptr + t * (2 * HPN) + g * HPN + hh, rem + off)


@triton.jit
def _ple_gather_kernel(table_addr, rid_ptr, out_ptr, ROW: tl.constexpr,
                       BLOCK: tl.constexpr):
    """One program per (token, head): copy one ROW-byte row.

    table_addr is the int64 address of the read-only mmap of the packed table.
    The GPU reads it through ATS (no copy to device memory). A page that is
    not in the page cache faults to the host and stalls this program.
    The launcher does not check an integer argument. It checks a pointer
    argument with cuPointerGetAttribute, and that check fails for an
    unregistered host mmap.
    """
    table_ptr = table_addr.to(tl.pointer_type(tl.uint8))
    i = tl.program_id(0)
    rid = tl.load(rid_ptr + i)                 # int64
    src = rid * ROW                            # rule 7: int64 byte offset
    b = tl.arange(0, BLOCK)
    m = b < ROW
    v = tl.load(table_ptr + src + b, mask=m)
    tl.store(out_ptr + i * ROW + b, v, mask=m)


def ple_ids(ids, qsl, nctx, mult, sizes, offs, out, eos: int, hpn: int = 8):
    t_pad = ids.shape[0]
    r_pad = qsl.shape[0] - 1
    block_r = triton.next_power_of_2(r_pad + 1)
    _ple_ids_kernel[(t_pad,)](ids, qsl, nctx, mult, sizes, offs, out,
                              R_PAD=r_pad, EOS=eos, HPN=hpn, BLOCK_R=block_r)
    return out


def ple_gather(table_u8, rids, out_u8, row: int = 90):
    """table_u8 is a tensor over the table or its int address."""
    n = rids.numel()
    addr = table_u8 if isinstance(table_u8, int) else table_u8.data_ptr()
    assert addr >= 1 << 32, "the table address must be an int64 argument"
    _ple_gather_kernel[(n,)](addr, rids, out_u8, ROW=row, BLOCK=128)
    return out_u8


__all__ = ["ple_ids", "ple_gather", "torch"]
