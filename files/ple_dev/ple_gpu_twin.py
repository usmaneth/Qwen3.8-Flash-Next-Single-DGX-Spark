# SPDX-License-Identifier: Apache-2.0
"""CPU twin of the device PLE kernels in ple_ids_triton.py.

Each numpy statement below copies one statement of _ple_ids_kernel or
_ple_gather_kernel. One array lane is one Triton program (one token, or one
token-head row). The twin uses the Triton integer semantics:
  - int32 loads, then an explicit cast to int64 (rule 5),
  - int64 multiply and XOR that wrap (two's complement),
  - C remainder for % (np.fmod: the sign follows the dividend), then the
    sign fix of rule 6.
It does not use np.remainder, torch or any code of forward_impl.

Variants (mutations of one rule each) exist only for the negative controls
in parity_real.py. A parity gate that cannot see these mutations has no
value. VARIANTS lists them.
"""
import numpy as np

VARIANTS = {
    "exact": "the kernel as written",
    "no_sign_fix": "rule 6 dropped: C remainder only",
    "i32_mul": "rule 5 dropped: the multiply stays in int32 and wraps",
    "no_eos_rule": "rule 4 dropped: g2 = w2",
    "ctx_swap": "rule 3 with the two ngram_context columns swapped",
    "search_left": "rule 1 with searchsorted(left) - 1",
    "i32_offset": "rule 7 dropped: the byte offset wraps in int32",
}


class Consts:
    """The layer constants that the kernel reads (int64 buffers)."""

    def __init__(self, mult, sizes, offs, eos: int, hpn: int, n: int = 3) -> None:
        if n != 3:
            raise ValueError("the kernel is written for ngram_size 3")
        self.mult = np.asarray(mult, dtype=np.int64)
        self.sizes = np.asarray(sizes, dtype=np.int64)
        self.offs = np.asarray(offs, dtype=np.int64)
        self.eos = int(eos)
        self.hpn = int(hpn)
        self.heads = 2 * self.hpn

    @classmethod
    def from_layer(cls, layer) -> "Consts":
        return cls(layer.layer_multipliers.cpu().numpy(),
                   layer.ngram_heads_vocab_sizes.cpu().numpy(),
                   layer.ngram_heads_offsets.cpu().numpy(),
                   int(layer.eos_token_id), int(layer.heads_per_ngram),
                   int(layer.ngram_size))


def _c_rem(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.fmod(a, b)  # C semantics, as arith.remsi


def ids_twin(k: Consts, ids: np.ndarray, qsl: np.ndarray, nctx: np.ndarray,
             variant: str = "exact") -> np.ndarray:
    """Row ids int64 [T_pad, H], as _ple_ids_kernel writes them.

    ids int32 [T_pad], qsl int32 [R_pad + 1], nctx int32 [R_pad, 2].
    """
    assert ids.dtype == np.int32 and qsl.dtype == np.int32 and nctx.dtype == np.int32
    t_pad = ids.shape[0]
    r_pad = qsl.shape[0] - 1
    t = np.arange(t_pad, dtype=np.int32)
    # Rule 1.
    side = "left" if variant == "search_left" else "right"
    r = np.searchsorted(qsl, t, side=side).astype(np.int32) - 1
    r = np.minimum(r, r_pad - 1)
    r = np.maximum(r, 0)  # the kernel never sees r < 0 (qsl[0] = 0 <= t)
    q_end = qsl[r_pad]
    c = t - qsl[r]
    valid = t < q_end
    # Rule 3.
    w0 = ids.astype(np.int64)
    w1_in = np.where(c >= 1, ids[np.maximum(t - 1, 0)], 0).astype(np.int64)
    col_new, col_old = (0, 1) if variant == "ctx_swap" else (1, 0)
    w1_cx = nctx[r, col_new].astype(np.int64)
    w1 = np.where(c >= 1, w1_in, w1_cx)
    w2_in = np.where(c >= 2, ids[np.maximum(t - 2, 0)], 0).astype(np.int64)
    w2_cx = nctx[r, np.where(c == 1, col_new, col_old)].astype(np.int64)
    w2 = np.where(c >= 2, w2_in, w2_cx)
    # Rule 2.
    eos = np.int64(k.eos)
    w0 = np.where(valid, w0, eos)
    w1 = np.where(valid, w1, eos)
    w2 = np.where(valid, w2, eos)
    # Rule 4.
    g2 = w2 if variant == "no_eos_rule" else np.where(w1 == eos, eos, w2)
    # Rule 5.
    with np.errstate(over="ignore"):
        if variant == "i32_mul":
            m = k.mult.astype(np.int32)  # the low 32 bits
            p0 = (w0.astype(np.int32) * m[0]).astype(np.int64)
            p1 = (w1.astype(np.int32) * m[1]).astype(np.int64)
            p2 = (g2.astype(np.int32) * m[2]).astype(np.int64)
        else:
            p0, p1, p2 = w0 * k.mult[0], w1 * k.mult[1], g2 * k.mult[2]
    mix2 = p0 ^ p1
    mix3 = mix2 ^ p2
    out = np.empty((t_pad, k.heads), dtype=np.int64)
    for g, mix in ((0, mix2), (1, mix3)):
        size = k.sizes[g * k.hpn:(g + 1) * k.hpn]
        off = k.offs[g * k.hpn:(g + 1) * k.hpn]
        rem = _c_rem(mix[:, None], size[None, :])
        if variant != "no_sign_fix":
            rem = np.where(rem < 0, rem + size[None, :], rem)
        out[:, g * k.hpn:(g + 1) * k.hpn] = rem + off[None, :]
    return out


def gather_twin(table_flat: np.ndarray, rids: np.ndarray, row: int = 90,
                variant: str = "exact") -> np.ndarray:
    """Rows uint8 [n, row], as _ple_gather_kernel writes them.

    table_flat is a 1-D uint8 view of the packed table (an np.memmap).
    """
    rid = rids.reshape(-1).astype(np.int64)
    with np.errstate(over="ignore"):
        src = rid * np.int64(row)
        if variant == "i32_offset":
            src = src.astype(np.int32).astype(np.int64)
    b = np.arange(row, dtype=np.int64)
    idx = src[:, None] + b[None, :]
    if variant == "i32_offset":
        # A wrapped offset can be negative. The GPU would read out of bounds;
        # the twin reads the wrapped address modulo the table size.
        idx = np.mod(idx, table_flat.shape[0])
    return table_flat[idx]
