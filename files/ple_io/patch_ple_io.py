#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hook the PLE row I/O module (files/ple_io/ple_io.py) into the offload path.

start.sh runs this after patch_ple_layer.py and patch_ple_offload.py. It
edits their generated outputs in place, so those two patch scripts stay
unchanged:

  files/ple_layer_patched.py      the packed-table gather calls ple_io.gather()
  files/ple_offload/worker.py     one trace record per request (trace only),
                                  the worker-side digest (check only)
  files/ple_offload/connector.py  the wait goes through ple_io.launch_wait():
                                  the launch record, the GPU gap and the
                                  GPU-side digest (trace or check only)

Then the hook loader imports each other files/ple_io/patch_*.py in sorted
order and calls its main(). Each of those files has its own MARK and exits 1
(which stops start.sh) when an anchor count is not 1.

With no VLLM_PLE_IO_* variable set, ple_io.gather() runs the shipped path
(fadvise per page, then index_select), and the hooks cost two clock reads
and one attribute test per step.
"""
import glob
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.dirname(HERE)
MARK = "vllm.v1.ple_offload import ple_io as _ple_io"


def edit(path: str, edits: list[tuple[str, str]], mark: str = MARK,
         tag: str = "ple_io") -> None:
    """Apply anchored edits to a generated file. Exit 1 if an anchor count is not 1."""
    src = open(path).read()
    if mark in src:
        print(f"{tag}: already hooked", os.path.basename(path))
        return
    for old, new in edits:
        n = src.count(old)
        if n != 1:
            print(f"ERROR: {tag}: {path}: anchor count {n}:\n{old[:200]}",
                  file=sys.stderr)
            sys.exit(1)
        src = src.replace(old, new)
    if mark not in src:
        print(f"ERROR: {tag}: {path}: the edits do not add the mark {mark!r}",
              file=sys.stderr)
        sys.exit(1)
    open(path, "w").write(src)
    print(f"{tag}: hooked", os.path.basename(path))


def own_edits() -> None:
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
            "        _io_t0 = _ple_io.now()\n"
            "        _io_chk = [] if _ple_io.CHECK else None\n",
        ),
        (
            "                slices = tuple(slice(0, size) for size in result.shape)\n",
            "                slices = tuple(slice(0, size) for size in result.shape)\n"
            "                if _io_chk is not None:\n"
            "                    _io_chk.append((request.seq, layer_name, result[slices]))\n",
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
            "                    _io_t0, _io_t1, request.num_tokens, request.num_reqs,\n"
            "                    request.seq, _ple_io.FAST,\n"
            "                )\n"
            "        if _io_chk:\n"
            "            for _seq, _name, _rows in _io_chk:\n"
            "                _ple_io.check_cpu(_seq, _name, _rows)\n",
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
            "        _ple_io.launch_wait(self, seq, num_tokens, num_reqs, _io_t0, _io_t1)\n",
        ),
    ])


def load_hooks() -> None:
    """Run the main() of each other files/ple_io/patch_*.py, in sorted order."""
    me = os.path.abspath(__file__)
    for path in sorted(glob.glob(os.path.join(HERE, "patch_*.py"))):
        if os.path.abspath(path) == me:
            continue
        name = "ple_io_hook_" + os.path.basename(path)[:-3]
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        mod.main()


def main() -> None:
    own_edits()
    load_hooks()


if __name__ == "__main__":
    main()
