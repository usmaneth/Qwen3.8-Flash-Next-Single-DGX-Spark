#!/usr/bin/env python3
"""Generate files/ours/model_runner.py: the GPU-worker hook of the boot-fast
verification hash (gate G1).

The hook runs at the end of GPUModelRunner.load_model (v1/worker/gpu), after
the target load, the drafter load, both process steps and the embed/lm_head
share, and after the "Model loading took" line (so that line keeps the load
time only). It calls vllm.boot_hash.dump_runner only when VLLM_BOOT_HASH_DIR
is set. start.sh mounts this file only with BOOT_HASH=1.

    patch_model_runner_hash.py          (reads files/ours/model_runner.py.orig)
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "model_runner.py.orig")
OUT = os.path.join(HERE, "model_runner.py")
# vllm/vllm-openai:qwen38-flash-next, RepoDigest sha256:fc120ece0a38...
ORIG_MD5 = "d8249dd9d634b94a4012337d20cbe015"

ANCHOR = '''        logger.info(
            "Model loading took %s GiB memory and %.6f seconds",
            format_gib(m.consumed_memory),
            time_after_load - time_before_load,
        )
'''
NEW = ANCHOR + '''        import os as _bf_os

        if _bf_os.environ.get("VLLM_BOOT_HASH_DIR", ""):
            from vllm.boot_hash import dump_runner

            dump_runner(self.model, getattr(self.speculator, "model", None))
'''


def main():
    with open(ORIG, "rb") as fh:
        raw = fh.read()
    md5 = hashlib.md5(raw).hexdigest()
    if md5 != ORIG_MD5:
        sys.exit(f"model_runner.py.orig md5 {md5} != {ORIG_MD5}: image changed, re-check the patch")
    s = raw.decode()
    if s.count(ANCHOR) != 1:
        sys.exit(f"anchor found {s.count(ANCHOR)} times (want 1)")
    with open(OUT, "w") as fh:
        fh.write(s.replace(ANCHOR, NEW, 1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
