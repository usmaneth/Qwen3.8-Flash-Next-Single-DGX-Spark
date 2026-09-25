#!/usr/bin/env python3
"""R3a: add the GPU-side PLE wait to the files that patch_ple_offload.py
writes (files/ple_offload/{ple_offload_layer,connector,worker}.py).

start.sh runs it after patch_ple_offload.py when KERN_DECODE=1. Every change
is inert unless VLLM_PLE_GPU_WAIT=1 is set in the container, so the knob-off
files run the recipe's host handshake. Idempotent.

With VLLM_PLE_GPU_WAIT=1 (see files/kern/ple_gpu_wait.py):
  ple_offload_layer.py  the semaphore tensor has 4 int32 words; the wait op
                        runs ple_gpu_wait.wait_kernel (captured in graphs).
  connector.py          _launch writes the expected seq to the device with
                        an H2D copy, queues the request on the request
                        thread, and returns: no host wait before the graph
                        replay (when _GPU_WAIT_ON, the kd_ext knob
                        ple_gpu_wait). Dummy forwards write expect = 0. An
                        error word from a GPU timeout raises on the next
                        step. With _GPU_WAIT_ON False the host waits as the
                        recipe does (the kernel then passes at once).
  worker.py             v2: the rows go into shared staging buffers (the
                        connector registers them for GPU access) and the seq
                        is published with a release store into the shared
                        host flag; the GPU kernel polls the flag and copies
                        the rows (no DMA from the worker's CUDA context).
  protocol.py           the registration carries the staging buffers.
"""
import ast
import os
import sys

MARK = "VLLM_PLE_GPU_WAIT"

LAYER = [
    ("import vllm.envs as envs\n",
     "import os\n\nimport vllm.envs as envs\n"),
    ("        self._flag_tensor = torch.zeros(1, dtype=torch.int32, device=device)\n",
     "        # R3a (VLLM_PLE_GPU_WAIT=1): [done, expect, error, spare].\n"
     "        self._flag_tensor = torch.zeros(4 if _GPU_WAIT else 1, dtype=torch.int32, device=device)\n"),
    ('    """No-op: the connector already blocked the host until the CPU\n'
     '    result landed in gpu_output_buffer (GB10 has no stream mem ops)."""\n'
     "    return\n",
     '    """No-op: the connector already blocked the host until the CPU\n'
     '    result landed in gpu_output_buffer (GB10 has no stream mem ops).\n'
     "    With VLLM_PLE_GPU_WAIT=1 a one-program kernel spins on the GPU until\n"
     '    the CPU worker has written this forward\'s seq (ple_gpu_wait.py)."""\n'
     "    if _GPU_WAIT and sem_flag_tensor.numel() >= 4:\n"
     "        from vllm.model_executor.layers.ple_gpu_wait import wait_for_rows\n"
     "\n"
     "        wait_for_rows(sem_flag_tensor, gpu_output_buffer, hidden_states.shape[0])\n"
     "    return\n"),
    ("_offload_worker_flag = False\n",
     "_offload_worker_flag = False\n"
     "# R3a: the GPU-side PLE wait (files/kern/ple_gpu_wait.py).\n"
     '_GPU_WAIT = os.environ.get("VLLM_PLE_GPU_WAIT", "0") == "1"\n'),
]

CONNECTOR = [
    ("logger = init_logger(__name__)\n",
     "logger = init_logger(__name__)\n"
     "\n"
     "# R3a (files/kern/ple_gpu_wait.py): the GPU waits for the PLE rows, the\n"
     "# host does not. _GPU_WAIT_ON is the run-time knob (kd_ext ple_gpu_wait).\n"
     '_GPU_WAIT = os.environ.get("VLLM_PLE_GPU_WAIT", "0") == "1"\n'
     "_GPU_WAIT_ON = _GPU_WAIT\n"
     "_RING = 64\n"),
    ("            done_flag=self._done_flag,\n        )\n",
     "            done_flag=self._done_flag,\n"
     "            staging_bufs=self._kd_make_staging() if _GPU_WAIT else None,\n"
     "        )\n"),
    ("        self.dp_rank = get_dp_group().rank_in_group\n",
     "        self._kd_ipc_addr = ipc_addr\n"
     "        self.dp_rank = get_dp_group().rank_in_group\n"),
    ("                self._request_socket = self._zmq_ctx.socket(zmq.PUSH)\n"
     "                self._request_socket.connect(ipc_addr)\n",
     "                self._request_socket = self._zmq_ctx.socket(zmq.PUSH)\n"
     "                self._request_socket.connect(ipc_addr)\n"
     "            if _GPU_WAIT:\n"
     "                self._kd_setup_gpu_wait()\n"),
    ("        self._seq += 1\n"
     "        seq = self._seq\n"
     "        if self.tp_rank == 0:\n",
     "        self._seq += 1\n"
     "        seq = self._seq\n"
     "        if _GPU_WAIT:\n"
     "            self._kd_write_expect(seq)\n"
     "            if _GPU_WAIT_ON and self.tp_rank == 0 and self._uses_cuda_inputs:\n"
     "                self._kd_launch_async(num_reqs, num_tokens, seq)\n"
     "                return\n"
     "        if self.tp_rank == 0:\n"),
    ("    def _wait_done(self, seq: int) -> None:\n",
     "    # ---- R3a: GPU-side wait (VLLM_PLE_GPU_WAIT=1) ----------------------\n"
     "    def _kd_setup_gpu_wait(self) -> None:\n"
     "        self._kd_flags = [layer._sem.flag_tensor for layer in self._layers.values()]\n"
     "        if any(f.numel() < 4 for f in self._kd_flags):\n"
     '            raise RuntimeError("VLLM_PLE_GPU_WAIT: the semaphore tensors have no expect word")\n'
     "        self._kd_ring = torch.zeros(_RING, dtype=torch.int32).pin_memory()\n"
     "        self._kd_zero = torch.zeros(1, dtype=torch.int32).pin_memory()\n"
     "        self._kd_err = torch.zeros(len(self._kd_flags), dtype=torch.int32).pin_memory()\n"
     "        self._kd_err_event = None\n"
     "        # v2: register the shared staging rows and the shared host flag for\n"
     "        # GPU access; the GPU kernel polls the flag and copies the rows.\n"
     "        from vllm.model_executor.layers import ple_gpu_wait as _pgw\n"
     "\n"
     "        _iv = lambda e: int(getattr(e, \"value\", e))\n"
     "        flags = _iv(cuda_driver.CU_MEMHOSTREGISTER_PORTABLE) | _iv(cuda_driver.CU_MEMHOSTREGISTER_DEVICEMAP)\n"
     "        def _devptr(t):\n"
     "            _cuda_check(cuda_driver.cuMemHostRegister(t.data_ptr(), t.numel() * t.element_size(), flags),\n"
     "                        \"cuMemHostRegister(PLE GPU wait)\")\n"
     "            self._pinned_input_buffers.append(t)\n"
     "            dptr = _cuda_check(cuda_driver.cuMemHostGetDevicePointer(t.data_ptr(), 0),\n"
     "                               \"cuMemHostGetDevicePointer\")[1]\n"
     "            return int(dptr)\n"
     "        hflag = _devptr(self._done_flag)\n"
     "        for name, layer in self._layers.items():\n"
     "            _pgw.register(layer._gpu_output_buffer.data_ptr(), hflag, _devptr(self._kd_staging[name]))\n"
     "        logger.info(\"PleOffload: GPU-side PLE wait enabled (R3a, %d layer(s), on=%s)\",\n"
     "                    len(self._kd_flags), _GPU_WAIT_ON)\n"
     "\n"
     "    def _kd_make_staging(self) -> dict:\n"
     "        self._kd_staging = {\n"
     "            name: torch.zeros(layer._gpu_output_buffer.shape, dtype=layer._gpu_output_buffer.dtype).share_memory_()\n"
     "            for name, layer in self._layers.items()\n"
     "        }\n"
     "        return self._kd_staging\n"
     "\n"
     "    def _kd_write_expect(self, seq: int) -> None:\n"
     "        slot = self._kd_ring[seq % _RING: seq % _RING + 1]\n"
     "        slot[0] = seq\n"
     "        stream = torch.cuda.current_stream(self.device)\n"
     "        for i, flag in enumerate(self._kd_flags):\n"
     "            flag[1:2].copy_(slot, non_blocking=True)\n"
     "        # The error words of earlier forwards (one step late, no host sync).\n"
     "        ev = self._kd_err_event\n"
     "        if ev is not None and ev.query() and int(self._kd_err.max()) != 0:\n"
     "            raise RuntimeError(\n"
     '                "PLE offload worker did not complete a step within "\n'
     '                f"{self._wait_timeout_s}s (GPU-side wait timed out)"\n'
     "            )\n"
     "        if ev is None or ev.query():\n"
     "            for i, flag in enumerate(self._kd_flags):\n"
     "                self._kd_err[i : i + 1].copy_(flag[2:3], non_blocking=True)\n"
     "            if ev is None:\n"
     "                ev = self._kd_err_event = torch.cuda.Event()\n"
     "            ev.record(stream)\n"
     "\n"
     "    def _kd_launch_async(self, num_reqs: int, num_tokens: int, seq: int) -> None:\n"
     "        assert self._input_ready_event is not None\n"
     "        self._input_ready_event.record(torch.cuda.current_stream(self.device))\n"
     "        if self._request_thread is None:\n"
     "            self._start_request_thread(self._kd_ipc_addr)\n"
     "        self._request_queue.put(\n"
     "            PleOffloadRequest(\n"
     "                dp_rank=self.dp_rank,\n"
     "                num_tokens=num_tokens,\n"
     "                num_reqs=num_reqs,\n"
     "                seq=seq,\n"
     "            )\n"
     "        )\n"
     "\n"
     "    def _wait_done(self, seq: int) -> None:\n"),
    ("        for layer in self._layers.values():\n"
     "            layer._gpu_output_buffer[:num_tokens].zero_()\n",
     "        for layer in self._layers.values():\n"
     "            layer._gpu_output_buffer[:num_tokens].zero_()\n"
     "        if _GPU_WAIT and getattr(self, \"_kd_flags\", None):\n"
     "            for flag in self._kd_flags:\n"
     "                flag[1:2].copy_(self._kd_zero, non_blocking=True)\n"
     "            for t in self._kd_staging.values():\n"
     "                t[:num_tokens].zero_()\n"),
]

WORKER = [
    ("                for target in targets:\n"
     "                    with torch.cuda.stream(target.copy_stream):\n"
     "                        target.gpu_output_buffer[slices].copy_(\n"
     "                            result[slices], non_blocking=True\n"
     "                        )\n",
     "                # R3a: with the GPU-side wait the rows go to the shared staging\n"
     "                # buffer (CPU memcpy); the GPU kernel copies them. No DMA from\n"
     "                # this process's CUDA context.\n"
     "                staging = _KD_STAGING.get((dp_rank, layer_name)) if _GPU_WAIT else None\n"
     "                if staging is not None:\n"
     "                    staging[slices].copy_(result[slices])\n"
     "                for target in (() if staging is not None else targets):\n"
     "                    with torch.cuda.stream(target.copy_stream):\n"
     "                        target.gpu_output_buffer[slices].copy_(\n"
     "                            result[slices], non_blocking=True\n"
     "                        )\n"),
    ("            for flag in flags:\n"
     "                flag[0] = request.seq\n",
     "            for flag in flags:\n"
     "                if _GPU_WAIT:\n"
     "                    _kd_release_store(flag, request.seq)\n"
     "                else:\n"
     "                    flag[0] = request.seq\n"),
    ("            if registration.tp_rank == 0:\n"
     "                self._input_bufs[registration.dp_rank] = PleOffloadInputBuffers(\n",
     "            for _name, _buf in (getattr(registration, \"staging_bufs\", None) or {}).items():\n"
     "                _KD_STAGING[(registration.dp_rank, _name)] = _buf\n"
     "            if registration.tp_rank == 0:\n"
     "                self._input_bufs[registration.dp_rank] = PleOffloadInputBuffers(\n"),
    ("import vllm.envs as envs\n",
     "import vllm.envs as envs\n"
     "\n"
     "# R3a: the worker writes the rows into the shared staging buffers and\n"
     "# publishes the seq with a release store (libatomic), so the GPU kernel\n"
     "# that acquires the flag sees the rows.\n"
     '_GPU_WAIT = os.environ.get("VLLM_PLE_GPU_WAIT", "0") == "1"\n'
     "_KD_STAGING: dict = {}\n"
     "_KD_STORE = None\n"
     "\n"
     "\n"
     "def _kd_release_store(flag, value: int) -> None:\n"
     "    global _KD_STORE\n"
     "    if _KD_STORE is None:\n"
     "        import ctypes\n"
     "\n"
     '        fn = ctypes.CDLL("libatomic.so.1").__atomic_store_8\n'
     "        fn.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]\n"
     "        fn.restype = None\n"
     "        _KD_STORE = fn\n"
     "    _KD_STORE(flag.data_ptr(), int(value), 3)  # __ATOMIC_RELEASE\n"),
]

PROTOCOL = [
    ("    done_flag: torch.Tensor | None = None\n",
     "    done_flag: torch.Tensor | None = None\n"
     "    # R3a (VLLM_PLE_GPU_WAIT=1): shared CPU staging rows per PLE layer. The\n"
     "    # CPU worker writes the rows here; a GPU kernel copies them.\n"
     "    staging_bufs: dict[str, torch.Tensor] | None = None\n"),
]

FILES = [("ple_offload_layer.py", LAYER), ("connector.py", CONNECTOR), ("worker.py", WORKER),
         ("protocol.py", PROTOCOL)]


def patch_text(text: str, edits, name: str) -> str:
    if MARK in text:
        return text
    for old, new in edits:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"patch_ple_kern: {name}: anchor count {n} != 1: {old.strip()[:70]!r}")
        text = text.replace(old, new, 1)
    ast.parse(text)
    return text


def main() -> None:
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ple_offload")
    changed = []
    for name, edits in FILES:
        p = os.path.join(d, name)
        s = open(p).read()
        out = patch_text(s, edits, name)
        if out != s:
            open(p, "w").write(out)
            changed.append(name)
    print(f"patch_ple_kern: {'applied to ' + ', '.join(changed) if changed else 'already applied'}")


if __name__ == "__main__":
    main()
