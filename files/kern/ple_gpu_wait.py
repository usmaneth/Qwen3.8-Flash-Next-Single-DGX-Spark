# SPDX-License-Identifier: Apache-2.0
"""R3a: a GPU-side wait for the PLE rows, so the host does not block.

Problem (kern-decode profile, K=6): before each verify graph replay the
host waits about 2.1 ms for the PLE CPU worker (connector._wait_done), then
launches the graph (about 0.6 ms). The GPU is idle for both. Upstream vLLM
waits on the GPU with cuStreamWaitValue32, but GB10 has no CUDA stream
memory operations, so the recipe patch moved the wait to the host.

Change. The semaphore tensor of each PLE layer gets 4 int32 words:
  [0] done   the CPU worker writes the request seq with an H2D copy on its
             copy stream, after the H2D copy of the rows (stream order)
  [1] expect the connector writes the seq of the forward with an H2D copy
             on the model stream, before the forward
  [2] error  this kernel writes 1 when it times out
  [3] spare
The PLE placeholder (layer index 1) runs wait_kernel just before it returns
the output buffer: one program spins on a volatile load of [0] until it
reaches [1]. The loads go to L2, where the copy-engine writes land. The
kernels after it read the rows. The kernel is captured in the CUDA graphs.
For dummy and capture forwards the connector writes expect = 0, so the wait
passes at once. On a timeout (VLLM_PLE_OFFLOAD_STEP_TIMEOUT) the kernel
writes the error word and returns; the connector raises on the next step,
as the host wait raises today.
"""
import os

import torch
import triton
import triton.language as tl

TIMEOUT_NS = int(float(os.environ.get("VLLM_PLE_OFFLOAD_STEP_TIMEOUT", "300")) * 1e9)


@triton.jit
def _now(x):
    return tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [x],
                                     dtype=tl.int64, is_pure=False, pack=1)


@triton.jit
def _ple_wait_kernel(F, timeout_ns):
    expect = tl.load(F + 1, volatile=True)
    cur = tl.load(F, volatile=True)
    t0 = _now(cur)
    el = t0 - t0
    while (cur < expect) & (el < timeout_ns):
        cur = tl.load(F, volatile=True)
        el = _now(cur) - t0
    if cur < expect:
        tl.store(F + 2, 1)


def wait_kernel(flag: torch.Tensor, timeout_ns: int = TIMEOUT_NS) -> None:
    """Enqueue the wait on the current stream (flag: int32[4] on the GPU)."""
    _ple_wait_kernel[(1,)](flag, timeout_ns, num_warps=1)


# ---------------------------------------------------------------------------
# v2 (the R3a fix): the CPU worker does not DMA the rows. It writes them into
# a shared, host-registered staging buffer and publishes the seq with a
# release store into the shared host flag. This kernel polls the host flag
# with an acquire load at system scope and then copies the rows into the
# GPU output buffer itself. v1 (the worker's H2D copies from its own CUDA
# context, then a device flag) made the GPU wait for a context time slice
# while the spin kernel ran: kern-r4 measured the verify part +2.0 ms and
# the host tail only -1.1 ms.
# ---------------------------------------------------------------------------
class _Ptr:
    """A raw device address for a Triton pointer argument."""

    def __init__(self, ptr: int, dtype: torch.dtype):
        self.ptr = int(ptr)
        self.dtype = dtype

    def data_ptr(self) -> int:
        return self.ptr


# gpu_output_buffer.data_ptr() -> (host flag device address, staging device address)
_REG: dict = {}


def register(buffer_ptr: int, flag_dev_ptr: int, staging_dev_ptr: int) -> None:
    _REG[int(buffer_ptr)] = (int(flag_dev_ptr), int(staging_dev_ptr))


@triton.jit
def _fence_acq_rel_sys(x):
    # A relaxed (volatile) read followed by fence.acq_rel.sys is an acquire
    # at system scope (PTX memory model): the row loads after it see the CPU
    # stores that came before the worker's release store of the flag.
    return tl.inline_asm_elementwise("fence.acq_rel.sys;\n\tmov.b32 $0, $1;", "=r,r", [x],
                                     dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _ple_wait_copy_kernel(F, HFLAG, SRC, DST, n_elem, timeout_ns, BLOCK: tl.constexpr):
    expect = tl.load(F + 1, volatile=True).to(tl.int64)
    cur = tl.load(HFLAG, volatile=True)
    t0 = _now(expect.to(tl.int32))
    el = t0 - t0
    while (cur < expect) & (el < timeout_ns):
        cur = tl.load(HFLAG, volatile=True)
        el = _now(expect.to(tl.int32)) - t0
    ok = _fence_acq_rel_sys((cur >= expect).to(tl.int32))
    if ok == 0:
        tl.store(F + 2, 1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_elem
    v = tl.load(SRC + offs, m, volatile=True)
    tl.store(DST + offs, v, m)


def wait_for_rows(flag: torch.Tensor, out_buffer: torch.Tensor, rows: int,
                  timeout_ns: int = TIMEOUT_NS) -> None:
    """v2 when the connector registered this buffer, else the v1 device-flag wait."""
    reg = _REG.get(out_buffer.data_ptr())
    if reg is None:
        wait_kernel(flag, timeout_ns)
        return
    hflag, staging = reg
    n_elem = int(rows) * out_buffer.shape[1]
    block = 4096
    grid = (max(1, triton.cdiv(n_elem, block)),)
    _ple_wait_copy_kernel[grid](flag, _Ptr(hflag, torch.int64), _Ptr(staging, out_buffer.dtype),
                                out_buffer, n_elem, timeout_ns, BLOCK=block, num_warps=4)
