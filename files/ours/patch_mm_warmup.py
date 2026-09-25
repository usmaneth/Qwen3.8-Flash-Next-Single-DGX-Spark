#!/usr/bin/env python3
"""Generate files/ours/base.py: an opt-in switch for the API-server
multi-modal warmup (boot-fast R9).

Why: after the engine is ready, the API server runs the multi-modal
processor once on dummy inputs at max_model_len (about 14 s here) before it
serves. The warmup result is discarded and the mm cache is cleared after it,
so text outputs do not change. The first image request pays the cost
instead.

With VLLM_SKIP_MM_WARMUP=1, renderers/base.py does not call
_warmup_mm_processor for the multi-modal processor and logs one line. The
Jinja chat-template warmup still runs. start.sh knob:
BOOT_FAST_SKIP_MM_WARMUP (default 0). Usman decides if production uses it.

    patch_mm_warmup.py                  (reads files/ours/base.py.orig)
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "base.py.orig")
OUT = os.path.join(HERE, "base.py")
# vllm/vllm-openai:qwen38-flash-next, RepoDigest sha256:fc120ece0a38...
ORIG_MD5 = "a7e4c76bebd021e363ac216d1aa52935"

ANCHOR = '''            if self.mm_processor:
                try:
                    logger.debug("Warming up multi-modal processing...")
'''
NEW = '''            if self.mm_processor and __import__("os").environ.get(
                "VLLM_SKIP_MM_WARMUP", "0"
            ) == "1":
                logger.info(
                    "boot-fast: multi-modal warmup skipped (VLLM_SKIP_MM_WARMUP=1); "
                    "the first image request pays it"
                )
            elif self.mm_processor:
                try:
                    logger.debug("Warming up multi-modal processing...")
'''


def main():
    with open(ORIG, "rb") as fh:
        raw = fh.read()
    md5 = hashlib.md5(raw).hexdigest()
    if md5 != ORIG_MD5:
        sys.exit(f"base.py.orig md5 {md5} != {ORIG_MD5}: image changed, re-check the patch")
    s = raw.decode()
    if s.count(ANCHOR) != 1:
        sys.exit(f"anchor found {s.count(ANCHOR)} times (want 1)")
    with open(OUT, "w") as fh:
        fh.write(s.replace(ANCHOR, NEW, 1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
