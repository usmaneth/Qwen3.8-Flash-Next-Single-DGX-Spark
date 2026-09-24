#!/usr/bin/env python3
"""Fuse the HC gate GEMM into the gated stream mean (prefill-ttft B6).

Each GatedResidual mix computes gate = lora @ W_up^T ([M, 320] x [320, 10240])
with cuBLAS, writes the [M, 10240] bf16 gate, and _hc_gate_mix reads it back to
form block_input[m, h] = mean_s(sigmoid(gate[m, s*H + h]) * xn[m, s*H + h]).
At M = 8K that is about 40 KB of gate traffic per row, 96 times per step.

The fused kernel computes each gate tile with tl.dot (fp32 accumulation over
K = 320 in order), rounds it to bf16 at the point where cuBLAS writes it,
applies the sigmoid in fp32 and accumulates the 4 streams in the same order
as _hc_gate_mix. The gate tensor is never written.

Rules:
  * M >= 256 rows only (prefill steps). Decode keeps the cuBLAS path.
  * Runtime flag hc_gate (pt_flags): "fused" (default) or "old".

Reads the image nvidia/hyperconnection.py from SRC and writes
hyperconnection.py next to this script. The .env mounts it over the image file.

    python3 patch_hc_gate_fused.py [/path/to/extracted/vllm]
"""
import os
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "/models/usman/qwen38-tune/vllm-src/vllm"
OUT = os.path.dirname(os.path.abspath(__file__))
REL = "models/qwen3_8_flash_next/nvidia/hyperconnection.py"

KERNEL = '''
from vllm.triton_utils import tl, triton

try:  # prefill-ttft B6: runtime A/B flags (files/ours/pt_flags.py)
    from vllm import pt_flags as _pt_flags
except ImportError:
    _pt_flags = None

_FUSED_MIN_ROWS = 256


@triton.jit
def _hc_gate_mix_fused_kernel(
    xn_ptr,
    lora_ptr,
    w_ptr,
    out_ptr,
    num_rows,
    stride_xn,
    stride_lora,
    stride_w,
    stride_out,
    HC: tl.constexpr,
    HC_DIM: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_ok = rm < num_rows
    col_ok = rn < HC_DIM
    out = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for stream in tl.static_range(HC):
        cols = stream * HC_DIM + rn
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k0 in tl.range(0, RANK, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            k_ok = rk < RANK
            a = tl.load(
                lora_ptr + rm[:, None] * stride_lora + rk[None, :],
                mask=row_ok[:, None] & k_ok[None, :],
                other=0.0,
            )
            b = tl.load(
                w_ptr + cols[None, :] * stride_w + rk[:, None],
                mask=k_ok[:, None] & col_ok[None, :],
                other=0.0,
            )
            acc = tl.dot(a, b, acc)
        # cuBLAS writes the gate as bf16; round at the same point.
        gate = acc.to(tl.bfloat16).to(tl.float32)
        x = tl.load(
            xn_ptr + rm[:, None] * stride_xn + cols[None, :],
            mask=row_ok[:, None] & col_ok[None, :],
            other=0.0,
        )
        out += tl.sigmoid(gate) * x.to(tl.float32)
    out /= HC
    tl.store(
        out_ptr + rm[:, None] * stride_out + rn[None, :],
        out,
        mask=row_ok[:, None] & col_ok[None, :],
    )


def hc_gate_mix_fused(
    xn: torch.Tensor, lora: torch.Tensor, w_up: torch.Tensor, hc_count: int
) -> torch.Tensor:
    """block_input = mean_s(sigmoid(lora @ w_up^T) * xn) without the gate tensor."""
    rows, dim = xn.shape
    rank = lora.shape[1]
    assert w_up.shape == (dim, rank) and xn.stride(1) == 1
    assert lora.stride(1) == 1 and w_up.stride(1) == 1
    hc_dim = dim // hc_count
    out = xn.new_empty(rows, hc_dim)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
    _hc_gate_mix_fused_kernel[(triton.cdiv(rows, BLOCK_M), triton.cdiv(hc_dim, BLOCK_N))](
        xn,
        lora,
        w_up,
        out,
        rows,
        xn.stride(0),
        lora.stride(0),
        w_up.stride(0),
        out.stride(0),
        HC=hc_count,
        HC_DIM=hc_dim,
        RANK=rank,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    return out


def _use_fused(rows: int, w_up: torch.Tensor) -> bool:
    if rows < _FUSED_MIN_ROWS or w_up.dtype != torch.bfloat16:
        return False
    return _pt_flags is None or _pt_flags.get("hc_gate", "fused") != "old"


class GatedResidual(nn.Module):'''

OLD_GATE = """        lora = hc_silu(lora, self.hc_count)
        gate = self.input_mix_weight_up(lora)  # [M, D]
        block_input = hc_gate_mix(xn, gate, self.hc_count)
"""
NEW_GATE = """        lora = hc_silu(lora, self.hc_count)
        w_up = self.input_mix_weight_up.weight
        if _use_fused(xn.shape[0], w_up):
            # prefill-ttft B6: the gate tensor is never written.
            block_input = hc_gate_mix_fused(xn, lora, w_up, self.hc_count)
        else:
            gate = self.input_mix_weight_up(lora)  # [M, D]
            block_input = hc_gate_mix(xn, gate, self.hc_count)
"""


def main() -> None:
    s = open(os.path.join(SRC, REL)).read()
    anchor = "\nclass GatedResidual(nn.Module):"
    if s.count(anchor) != 1 or s.count(OLD_GATE) != 2:
        sys.exit(f"{REL}: anchors changed (class {s.count(anchor)}, gate {s.count(OLD_GATE)})")
    s = s.replace(anchor, KERNEL.rstrip("\n").replace("class GatedResidual(nn.Module):", "")
                  + "\n\nclass GatedResidual(nn.Module):", 1)
    s = s.replace(OLD_GATE, NEW_GATE)
    open(os.path.join(OUT, "hyperconnection.py"), "w").write(s)
    print("wrote hyperconnection.py (fused HC gate)")


if __name__ == "__main__":
    main()
