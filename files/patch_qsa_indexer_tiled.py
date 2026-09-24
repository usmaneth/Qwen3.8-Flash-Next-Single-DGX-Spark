#!/usr/bin/env python3
"""Row-tiled QSA indexer score kernel (prefill-ttft B5).

The image kernel _qsa_mqa_paged_kernel scores one query row per program. Each
program reads all visible compressed keys of its request again, so a prefill
step of R rows reads the keys R times (about 1.6 TB/s of re-reads at long
context). This is the quadratic term of the cold prefill curve.

The tiled kernel scores ROWS query rows of one request against one key tile
with one tensor-core dot ([BLOCK_N, 128] x [128, ROWS * 4 heads]), so a key
tile is read once per ROWS rows. It uses the same scale, the same ReLU, the
same fp32 accumulation and the same division as the image kernel.

Rules:
  * A row tile is "uniform" when its first and last rows belong to the same
    request (rows are sorted by request). The tiled kernel handles uniform
    tiles. The image kernel, with MIXED_ROWS set, handles only the rows of the
    other tiles (request boundaries, padding rows).
  * Batches of fewer than 64 rows (decode) use the image kernel only.
  * Runtime flag qsa_indexer (pt_flags): "tiled" (default) or "old".

Input and output: files/qsa_ops_patched.py, after patch_qsa_fp8_kv.py. start.sh
runs this script after patch_qsa_fp8_kv.py on every launch.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "qsa_ops_patched.py")
MARK = "prefill-ttft B5"

OLD_SIG = """    COMPRESS_RATIO: tl.constexpr,
    KV_QUANT_MODE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    heads = tl.arange(0, MAX_N)
    request = tl.load(token_to_req_ptr + row)
"""
NEW_SIG = """    COMPRESS_RATIO: tl.constexpr,
    KV_QUANT_MODE: tl.constexpr,
    MIXED_ROWS: tl.constexpr = 0,
) -> None:
    row = tl.program_id(0)
    if MIXED_ROWS > 0:
        # prefill-ttft B5: the tiled kernel owns the rows of uniform tiles.
        tile_first = (row // MIXED_ROWS) * MIXED_ROWS
        tile_last = tl.minimum(tile_first + MIXED_ROWS, num_rows) - 1
        first_request = tl.load(token_to_req_ptr + tile_first)
        last_request = tl.load(token_to_req_ptr + tile_last)
        if (
            (first_request == last_request)
            & (first_request >= 0)
            & (first_request < num_requests)
        ):
            return
    dims = tl.arange(0, BLOCK_D)
    heads = tl.arange(0, MAX_N)
    request = tl.load(token_to_req_ptr + row)
"""

TILED_KERNEL = '''

# prefill-ttft B5: row-tiled score kernel.
_TILED_ROWS = 32
_TILED_MIN_ROWS = 64
try:
    from vllm import pt_flags as _pt_flags
except ImportError:
    _pt_flags = None


@triton.jit
def _qsa_mqa_paged_tiled_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    num_rows,
    num_columns,
    num_pages,
    num_requests,
    score_divisor,
    k_scale_ptr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROWS: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    STAGES: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    KV_QUANT_MODE: tl.constexpr,
) -> None:
    tile_first = tl.program_id(0) * ROWS
    tile_last = tl.minimum(tile_first + ROWS, num_rows) - 1
    request = tl.load(token_to_req_ptr + tile_first)
    last_request = tl.load(token_to_req_ptr + tile_last)
    # Mixed or padding tiles: the image kernel (MIXED_ROWS) scores them.
    if (request != last_request) | (request < 0) | (request >= num_requests):
        return
    row_offsets = tl.arange(0, ROWS)
    rows = tile_first + row_offsets
    row_ok = rows < num_rows
    sequence_length = tl.load(sequence_lengths_ptr + request)
    query_position = tl.load(query_positions_ptr + rows, mask=row_ok, other=0)
    visible = tl.minimum(
        (query_position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    visible = tl.where(row_ok, visible, 0)
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + rows, visible, mask=row_ok)
    visible_max = tl.max(visible, axis=0)
    tile_start = tl.program_id(1) * TILES_PER_PROG
    if tile_start * BLOCK_N >= visible_max:
        return
    tile_end = tl.minimum(tile_start + TILES_PER_PROG, tl.cdiv(visible_max, BLOCK_N))
    tile_end = tl.minimum(tile_end, tl.cdiv(num_columns, BLOCK_N))

    # Query tile [BLOCK_D, ROWS * NUM_HEADS]: column m is row m // NUM_HEADS,
    # head m % NUM_HEADS.
    dims = tl.arange(0, BLOCK_D)
    m = tl.arange(0, ROWS * NUM_HEADS)
    m_row = tile_first + m // NUM_HEADS
    m_head = m % NUM_HEADS
    query = tl.load(
        q_ptr
        + m_row[None, :] * stride_q_row
        + m_head[None, :] * stride_q_head
        + dims[:, None] * stride_q_dim,
        mask=(m_row[None, :] < num_rows) & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(tile_start, tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live_any = columns < visible_max
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr
            + request * stride_table_req
            + logical_page * stride_table_page,
            mask=live_any,
            other=-1,
        )
        page_valid = live_any & (physical_page >= 0) & (physical_page < num_pages)
        safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=page_valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
        )
        keys = keys.to(query.dtype)
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        if KV_QUANT_MODE:
            scores *= tl.load(k_scale_ptr)
        scores = tl.maximum(scores, 0.0)
        score = tl.sum(tl.reshape(scores, (BLOCK_N, ROWS, NUM_HEADS)), axis=2)
        score = score / score_divisor
        live = (columns[:, None] < visible[None, :]) & (columns[:, None] < num_columns)
        tl.store(
            logits_ptr + rows[None, :] * stride_logits_row + columns[:, None],
            tl.where(page_valid[:, None], score, -float("inf")),
            mask=live & row_ok[None, :],
        )


def _qsa_use_tiled(rows: int, num_heads: int) -> bool:
    if rows < _TILED_MIN_ROWS or _TILED_ROWS * num_heads > 256:
        return False
    if num_heads & (num_heads - 1):
        return False
    return _pt_flags is None or _pt_flags.get("qsa_indexer", "tiled") != "old"


@triton.jit
def _expand_qsa_indices_kernel('''

OLD_EXPAND_HEAD = '''

@triton.jit
def _expand_qsa_indices_kernel('''

OLD_LAUNCH_HEAD = """    # Tuned on GB300: larger row batches provide enough parallelism to reuse Q.
    tiles_per_program = 1 if q.shape[0] <= 32 else 8
    _qsa_mqa_paged_kernel[
        (q.shape[0], triton.cdiv(columns, BLOCK_N * tiles_per_program))
    ](
"""
NEW_LAUNCH_HEAD = """    # Tuned on GB300: larger row batches provide enough parallelism to reuse Q.
    tiles_per_program = 1 if q.shape[0] <= 32 else 8
    mixed_rows = 0
    if _qsa_use_tiled(q.shape[0], q.shape[1]):
        # prefill-ttft B5: uniform row tiles run the tiled kernel. The image
        # kernel then scores only the rows of mixed tiles, all columns in one
        # program per row.
        _qsa_mqa_paged_tiled_kernel[
            (triton.cdiv(q.shape[0], _TILED_ROWS), triton.cdiv(columns, BLOCK_N * 8))
        ](
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            sequence_lengths,
            visible_blocks,
            logits,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(3),
            page_table.stride(0),
            page_table.stride(1),
            logits.stride(0),
            q.shape[0],
            columns,
            k_cache.shape[0],
            page_table.shape[0],
            float(score_divisor),
            _qsa_scale_ptr(k_scale, q.device),
            PAGE_SIZE=k_cache.shape[1],
            PAGE_TABLE_WIDTH=page_table.shape[1],
            NUM_HEADS=q.shape[1],
            HEAD_DIM=q.shape[2],
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            ROWS=_TILED_ROWS,
            TILES_PER_PROG=8,
            STAGES=2,
            COMPRESS_RATIO=compress_ratio,
            KV_QUANT_MODE=kv_quant_mode,
            num_warps=4,
        )
        mixed_rows = _TILED_ROWS
        tiles_per_program = triton.cdiv(columns, BLOCK_N)
    _qsa_mqa_paged_kernel[
        (q.shape[0], triton.cdiv(columns, BLOCK_N * tiles_per_program))
    ](
"""

OLD_LAUNCH_TAIL = """        COMPRESS_RATIO=compress_ratio,
        KV_QUANT_MODE=kv_quant_mode,
        num_warps=2,
    )
    return logits, visible_blocks
"""
NEW_LAUNCH_TAIL = """        COMPRESS_RATIO=compress_ratio,
        KV_QUANT_MODE=kv_quant_mode,
        MIXED_ROWS=mixed_rows,
        num_warps=2,
    )
    return logits, visible_blocks
"""

EDITS = [
    (OLD_SIG, NEW_SIG),
    (OLD_EXPAND_HEAD, TILED_KERNEL),
    (OLD_LAUNCH_HEAD, NEW_LAUNCH_HEAD),
    (OLD_LAUNCH_TAIL, NEW_LAUNCH_TAIL),
]


def main() -> None:
    s = open(PATH).read()
    if MARK in s:
        sys.exit(f"{PATH}: already patched ({MARK})")
    for old, new in EDITS:
        if s.count(old) != 1:
            sys.exit(f"qsa_ops_patched.py: anchor found {s.count(old)} times: {old[:70]!r}")
        s = s.replace(old, new, 1)
    open(PATH, "w").write(s)
    print("patched qsa_ops_patched.py (row-tiled QSA indexer)")


if __name__ == "__main__":
    main()
