#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""B3 hook: the numpy fast path of the PLE hash (files/ple_io/ple_io_fast.py).

The hook loader in patch_ple_io.py runs main() after its own edits. It adds
one early return at the top of Qwen3_8FlashNextNGramEmbedding.forward_impl
in files/ple_layer_patched.py. The return is taken only in the offload
process, with an output buffer, fast=1 and at most VLLM_PLE_IO_FAST_MAX
tokens. In all other cases forward_impl runs as before.

The edit is anchored. An anchor count that is not 1 stops start.sh.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.dirname(HERE)
MARK = "ple_io_fast as _ple_io_fast"

sys.path.insert(0, HERE)
from patch_ple_io import edit  # noqa: E402

EDITS = [(
    "        del hidden_states\n"
    "        input_ids = input_ids.reshape(-1).long()\n"
    "        query_start_loc = query_start_loc.long()\n",
    "        del hidden_states\n"
    "        if output_buffer is not None and is_offload_process():\n"
    "            from vllm.v1.ple_offload import ple_io_fast as _ple_io_fast\n"
    "            if _ple_io_fast.enabled(input_ids):\n"
    "                _fast_out = _ple_io_fast.small_forward(\n"
    "                    self, input_ids, query_start_loc, ngram_context, output_buffer\n"
    "                )\n"
    "                if _fast_out is not None:\n"
    "                    return _fast_out\n"
    "        input_ids = input_ids.reshape(-1).long()\n"
    "        query_start_loc = query_start_loc.long()\n",
)]


def main() -> None:
    edit(os.path.join(FILES, "ple_layer_patched.py"), EDITS, mark=MARK,
         tag="ple_io_fast")


if __name__ == "__main__":
    main()
