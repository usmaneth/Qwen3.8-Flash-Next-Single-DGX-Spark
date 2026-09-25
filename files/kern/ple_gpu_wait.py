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
