#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hook the PLE row I/O module (files/ple_io/ple_io.py) into the offload path.

start.sh runs this after patch_ple_layer.py and patch_ple_offload.py. It
edits their generated outputs in place, so those two patch scripts stay
unchanged:

  files/ple_layer_patched.py      the packed-table gather calls ple_io.gather()
  files/ple_offload/worker.py     one trace record per request (trace only)
  files/ple_offload/connector.py  one trace record per launch (trace only)

With no VLLM_PLE_IO_* variable set, ple_io.gather() runs the shipped path
(fadvise per page, then index_select), and the trace hooks are two clock
reads and one attribute test per step.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.dirname(HERE)
MARK = "vllm.v1.ple_offload import ple_io as _ple_io"


def edit(path: str, edits: list[tuple[str, str]]) -> None:
    src = open(path).read()
    if MARK in src:
        print("ple_io: already hooked", os.path.basename(path))
        return
    for old, new in edits:
        n = src.count(old)
        if n != 1:
            print(f"ERROR: {path}: anchor count {n}:\n{old[:160]}", file=sys.stderr)
            sys.exit(1)
        src = src.replace(old, new)
    open(path, "w").write(src)
    print("ple_io: hooked", os.path.basename(path))


def main() -> None:
    edit(os.path.join(FILES, "ple_layer_patched.py"), [(
        "                _ple_prefetch_rows(\n"
        "                    getattr(emb, \"_packed_table_fd\", None), ids, row_width\n"
        "                )\n"
        "                torch.index_select(\n"
        "                    packed, 0, ids,\n"
        "                    out=output.reshape(-1, row_width).view(torch.uint8)\n"
        "                )\n",
        "                from vllm.v1.ple_offload import ple_io as _ple_io\n"
        "                _ple_io.gather(\n"
        "                    getattr(emb, \"_packed_table_fd\", None), packed, ids,\n"
        "                    output.reshape(-1, row_width).view(torch.uint8),\n"
        "                )\n",
    )])

    edit(os.path.join(FILES, "ple_offload", "worker.py"), [
        (
            "import msgspec\n",
            "import msgspec\n"
            "from vllm.v1.ple_offload import ple_io as _ple_io\n",
        ),
        (
            "        \"\"\"Run requests layer-first so each DP rank can resume promptly.\"\"\"\n",
            "        \"\"\"Run requests layer-first so each DP rank can resume promptly.\"\"\"\n"
            "        _io_t0 = _ple_io.now()\n",
        ),
        (
            "        # Host-side handshake (GB10 has no stream memory ops): wait for\n",
            "        _io_t1 = _ple_io.now()\n"
            "        # Host-side handshake (GB10 has no stream memory ops): wait for\n",
        ),
        (
            "            for flag in flags:\n"
            "                flag[0] = request.seq\n",
            "            for flag in flags:\n"
            "                flag[0] = request.seq\n"
            "        if _ple_io.TRACE:\n"
            "            for request in requests_by_dp.values():\n"
            "                _ple_io.trace_request(\n"
            "                    _io_t0, _io_t1, request.num_tokens, request.num_reqs\n"
            "                )\n",
        ),
    ])

    edit(os.path.join(FILES, "ple_offload", "connector.py"), [
        (
            "import msgspec\n",
            "import msgspec\n"
            "from vllm.v1.ple_offload import ple_io as _ple_io\n",
        ),
        (
            "        self._seq += 1\n"
            "        seq = self._seq\n",
            "        _io_t0 = _ple_io.now()\n"
            "        self._seq += 1\n"
            "        seq = self._seq\n",
        ),
        (
            "            self._process_request(request, self._request_socket)\n"
            "        self._wait_done(seq)\n",
            "            self._process_request(request, self._request_socket)\n"
            "        _io_t1 = _ple_io.now()\n"
            "        self._wait_done(seq)\n"
            "        if _ple_io.TRACE:\n"
            "            _ple_io.trace_launch(_io_t0, _io_t1, num_tokens, num_reqs)\n",
        ),
    ])


if __name__ == "__main__":
    main()
