#!/usr/bin/env python3
"""Backport vllm-project/vllm#53388 (disable_eagle_block_drop) into this image.

With EAGLE-style drafters (MTP included), the prefix cache drops the trailing
matched block and re-prefills it on every request. On Codex traffic that is
about one 1664-token block per turn. The option keeps that block. The target
verifies every draft token, so the change can only move acceptance, never
output correctness.

Reads the image's vLLM sources from SRC and writes patched copies next to this
script. start.sh mounts them over the image files.

    python3 patch_block_drop.py /path/to/extracted/vllm
"""
import os, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "/models/usman/qwen38-tune/vllm-src/vllm"
OUT = os.path.dirname(os.path.abspath(__file__))


def edit(rel, out_name, pairs):
    s = open(os.path.join(SRC, rel)).read()
    for old, new in pairs:
        if old not in s:
            sys.exit(f"{rel}: anchor not found: {old[:70]!r}")
        s = s.replace(old, new, 1)
    open(os.path.join(OUT, out_name), "w").write(s)
    print(f"wrote {out_name}")


edit("config/speculative.py", "speculative.py", [
    ("    use_local_argmax_reduction: bool = False\n",
     "    disable_eagle_block_drop: bool = False\n"
     '    """Disable dropping the trailing prefix-cache block for EAGLE-like\n'
     "    speculative methods (backport of vllm#53388). The drafter still runs;\n"
     '    only prefix-cache reuse of that block changes."""\n'
     "    use_local_argmax_reduction: bool = False\n"),
    ('        return self.method in ("eagle", "eagle3", "mtp", "dflash", "dspark")\n',
     '        return self.method in ("eagle", "eagle3", "mtp", "dflash", "dspark")\n\n'
     "    def use_eagle_block_drop(self) -> bool:\n"
     '        """Whether volatile trailing cache blocks should be discarded."""\n'
     "        return self.use_eagle() and not self.disable_eagle_block_drop\n"),
])

edit("v1/core/kv_cache_utils.py", "kv_cache_utils.py", [
    ("    if spec_config is None or not spec_config.use_eagle():\n",
     "    if spec_config is None or not spec_config.use_eagle_block_drop():\n"),
])

edit("v1/core/sched/scheduler.py", "scheduler.py", [
    ("        self.use_eagle = False\n",
     "        self.use_eagle = False\n        self.use_eagle_block_drop = False\n"),
    ("            self.use_eagle = speculative_config.use_eagle()\n",
     "            self.use_eagle = speculative_config.use_eagle()\n"
     "            self.use_eagle_block_drop = speculative_config.use_eagle_block_drop()\n"
     "            if self.use_eagle and not self.use_eagle_block_drop:\n"
     "                logger.warning(\n"
     '                    "EAGLE trailing prefix-cache block dropping is disabled "\n'
     '                    "(vllm#53388 backport)."\n'
     "                )\n"),
    ("            use_eagle=self.use_eagle,\n",
     "            use_eagle=self.use_eagle_block_drop,\n"),
    ("        if self.use_eagle:\n            last_cache_position = max(last_cache_position - block_size, 0)\n",
     "        if self.use_eagle_block_drop:\n            last_cache_position = max(last_cache_position - block_size, 0)\n"),
])
