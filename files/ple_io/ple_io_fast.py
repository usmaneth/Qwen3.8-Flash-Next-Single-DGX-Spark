# SPDX-License-Identifier: Apache-2.0
"""B3: a numpy copy of the PLE n-gram hash for small batches (``fast=1``).

The container mounts this file as ``vllm.v1.ple_offload.ple_io_fast``.
files/ple_io/patch_fast.py adds one early return at the top of
``Qwen3_8FlashNextNGramEmbedding.forward_impl`` (ple_layer_patched.py).

The offload branch of forward_impl computes the row ids with about 50 small
torch ops. For a decode verify step (7 tokens per request at K=6) the op
overhead is about 0.7 ms. ``small_forward()`` computes the same ids with
numpy on the same inputs, then calls the same ``ple_io.gather()``. It keeps:
  - the packing of the requests, with the clamp of the CUDA-graph padding
    tokens (num_valid_tokens = min(query_start_loc[-1], num_tokens)) and the
    clamp of the request index and the column,
  - the EOS segment rule of _shift_precompute and _shift_apply (a shifted
    token that crosses an EOS, or the start of the context, becomes EOS),
  - int64 products that wrap, np.remainder (the sign follows the divisor,
    as torch.remainder), the per-head offsets and the row order.
It reads only the layer's own buffers (layer_multipliers,
ngram_heads_vocab_sizes, ngram_heads_offsets). Gate G2 (test_fast.py)
compares its ids and bytes with the torch forward_impl.

The fast path runs only in the offload process, with an output buffer and
a packed table, with fast=1, and for at most VLLM_PLE_IO_FAST_MAX tokens
(default 64). Else it returns None and forward_impl runs as before.
"""

import os

import numpy as np
import torch

from vllm.v1.ple_offload import ple_io as _ple_io

FAST_MAX = int(os.environ.get("VLLM_PLE_IO_FAST_MAX", "64") or 64)


def enabled(input_ids: torch.Tensor) -> bool:
    return bool(_ple_io.FAST) and input_ids.numel() <= FAST_MAX


class _Consts:
    """numpy copies of the hash buffers of one layer."""

    def __init__(self, layer) -> None:
        self.n = int(layer.ngram_size)
        self.h = int(layer.heads_per_ngram)
        self.c = self.n - 1
        self.eos = int(layer.eos_token_id)
        self.mult = layer.layer_multipliers.detach().cpu().numpy().astype(np.int64)
        sizes = layer.ngram_heads_vocab_sizes.detach().cpu().numpy().astype(np.int64)
        offs = layer.ngram_heads_offsets.detach().cpu().numpy().astype(np.int64)
        self.sizes = [sizes[(g - 2) * self.h:(g - 1) * self.h] for g in range(2, self.n + 1)]
        self.offs = [offs[(g - 2) * self.h:(g - 1) * self.h] for g in range(2, self.n + 1)]
        self.max_tokens = int(layer.positions_buffer.numel())
        self.max_reqs = int(layer.padded_buffer.shape[0])


def row_ids(k: _Consts, ids: np.ndarray, q: np.ndarray,
            nc: np.ndarray) -> np.ndarray | None:
    """The int64 [num_tokens, (n-1)*h] row ids, as forward_impl computes them."""
    t = ids.shape[0]
    r = q.shape[0] - 1
    if r <= 0 or t > k.max_tokens or r > k.max_reqs:
        return None
    max_len = max(1, int((q[1:] - q[:-1]).max()))
    n_valid = min(int(q[-1]), t)
    pos = np.arange(t, dtype=np.int64)
    ri = np.searchsorted(q, pos, side="right") - 1
    np.minimum(ri, r - 1, out=ri)
    cols = pos - q[ri]
    np.clip(cols, 0, max_len - 1, out=cols)
    packed = np.full((r, max_len), k.eos, dtype=np.int64)
    packed[ri[:n_valid], cols[:n_valid]] = ids[:n_valid]
    ctx = np.concatenate((nc[:r], packed), axis=1)
    width = ctx.shape[1]
    p = np.arange(width, dtype=np.int64)
    eos_pos = np.where(ctx == k.eos, p, -1)
    prev = np.empty_like(eos_pos)
    prev[:, 0] = -1
    if width > 1:
        prev[:, 1:] = np.maximum.accumulate(eos_pos, axis=1)[:, :-1]
    cc = cols + k.c
    pis = cc - prev[ri, cc] - 1  # position in the EOS segment at (ri, cc)
    toks = [ctx[ri, cc]]
    for s in range(1, k.n):
        # cc - s >= 0 always (cc >= n - 1), so only the segment test remains.
        toks.append(np.where(pis >= s, ctx[ri, cc - s], k.eos))
    blocks = []
    for g in range(2, k.n + 1):
        mixed = toks[0] * k.mult[0]
        for i in range(1, g):
            mixed = np.bitwise_xor(mixed, toks[i] * k.mult[i])
        blocks.append(np.remainder(mixed[:, None], k.sizes[g - 2][None, :])
                      + k.offs[g - 2][None, :])
    return np.concatenate(blocks, axis=1)


def small_forward(layer, input_ids: torch.Tensor, query_start_loc: torch.Tensor,
                  ngram_context: torch.Tensor | None,
                  output_buffer: torch.Tensor) -> torch.Tensor | None:
    """The offload branch of forward_impl for a small batch, or None."""
    emb = getattr(layer, "ngram_embedding", None)
    table = getattr(emb, "_packed_table", None)
    if table is None or ngram_context is None:
        return None
    k = layer.__dict__.get("_ple_io_fast_consts")
    if k is None:
        k = _Consts(layer)
        layer.__dict__["_ple_io_fast_consts"] = k
    ids = input_ids.reshape(-1).numpy().astype(np.int64)
    q = query_start_loc.numpy().astype(np.int64)
    r = q.shape[0] - 1
    nc = ngram_context[:max(r, 0)].numpy().astype(np.int64)
    ngram_ids = row_ids(k, ids, q, nc)
    if ngram_ids is None:
        return None  # forward_impl raises the same error as before
    row_width = table.shape[-1]
    total_width = ngram_ids.shape[-1] * row_width
    output = output_buffer[:ids.shape[0], :total_width]
    _ple_io.gather(
        getattr(emb, "_packed_table_fd", None), table,
        torch.from_numpy(ngram_ids.reshape(-1)),
        output.reshape(-1, row_width).view(torch.uint8),
    )
    return output
