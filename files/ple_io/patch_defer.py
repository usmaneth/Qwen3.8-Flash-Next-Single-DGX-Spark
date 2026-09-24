#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""B2 hooks: the deferred PLE wait (files/ple_io/ple_io_defer.py).

The hook loader in patch_ple_io.py runs main() after its own edits. The
edits go into the generated files:

  files/ple_offload/connector.py
      _launch: before_launch() first; the wait goes through
               ple_io_defer.launch(), which defers it when defer=1
      release_outputs: the backstop, after_forward()
      __init__ end: install() wraps the graph replay methods
  files/ple_offload/ple_offload_layer.py
      _ple_offload_wait_impl (the eager placeholder): wait_pending()

Each edit is anchored. An anchor count that is not 1 stops start.sh.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.dirname(HERE)
MARK = "ple_io_defer as _ple_io_defer"
MARK_LAYER = "vllm.v1.ple_offload.ple_io_defer"

sys.path.insert(0, HERE)
from patch_ple_io import edit  # noqa: E402


def main() -> None:
    edit(os.path.join(FILES, "ple_offload", "connector.py"), [
        (
            "from vllm.v1.ple_offload import ple_io as _ple_io\n",
            "from vllm.v1.ple_offload import ple_io as _ple_io\n"
            "from vllm.v1.ple_offload import ple_io_defer as _ple_io_defer\n",
        ),
        (
            "        _io_t0 = _ple_io.now()\n"
            "        self._seq += 1\n",
            "        _ple_io_defer.before_launch(self)\n"
            "        _io_t0 = _ple_io.now()\n"
            "        self._seq += 1\n",
        ),
        (
            "        _ple_io.launch_wait(self, seq, num_tokens, num_reqs, _io_t0, _io_t1)\n",
            "        _ple_io_defer.launch(self, seq, num_tokens, num_reqs, _io_t0, _io_t1)\n",
        ),
        (
            "        # sent after the previous forward completed on the model stream.\n"
            "        return\n",
            "        # sent after the previous forward completed on the model stream.\n"
            "        _ple_io_defer.after_forward(self)\n"
            "        return\n",
        ),
        (
            "        except Exception:\n"
            "            self.close()\n"
            "            raise\n",
            "        except Exception:\n"
            "            self.close()\n"
            "            raise\n"
            "        _ple_io_defer.install(self)\n",
        ),
    ], mark=MARK, tag="ple_io_defer")

    edit(os.path.join(FILES, "ple_offload", "ple_offload_layer.py"), [
        (
            "import functools\n",
            "import functools\n"
            "import sys as _sys\n",
        ),
        (
            "    result landed in gpu_output_buffer (GB10 has no stream mem ops).\"\"\"\n"
            "    return\n",
            "    result landed in gpu_output_buffer (GB10 has no stream mem ops).\n"
            "    With ple_io defer=1 the connector returns before the wait; this\n"
            "    placeholder then waits for the pending request (eager steps).\"\"\"\n"
            "    _defer = _sys.modules.get(\"vllm.v1.ple_offload.ple_io_defer\")\n"
            "    if _defer is not None and _defer.pending is not None:\n"
            "        _defer.wait_pending()\n"
            "    return\n",
        ),
    ], mark=MARK_LAYER, tag="ple_io_defer")


if __name__ == "__main__":
    main()
