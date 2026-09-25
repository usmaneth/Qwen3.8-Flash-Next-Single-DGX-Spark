#!/usr/bin/env python3
"""Add the kern-decode drafter hooks to files/mtp_patched.py (after
patch_mtp_draft_vocab.py and patch_mtp_fp8_head.py; start.sh runs it when
KERN_DECODE=1). Idempotent: a second run changes nothing.

R4: at the end of Qwen3_8FlashNextMTP.load_weights, a call to
enable_mtp_w8a16 (mtp_w8a16.py) when VLLM_MTP_DENSE_W8A16=1. With the env
var unset the hook does nothing, so the file runs the image drafter.
"""
import ast
import os
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mtp_patched.py")
MARK = "enable_mtp_w8a16"
OLD = "        _attach_draft_vocab(self)\n        return loaded\n"
NEW = ("        _attach_draft_vocab(self)\n"
       '        if os.environ.get("VLLM_MTP_DENSE_W8A16", "0") == "1":\n'
       "            # R4: FP8 row-scaled copies of the dense drafter linears.\n"
       "            from .mtp_w8a16 import enable_mtp_w8a16\n"
       "\n"
       "            enable_mtp_w8a16(self, logger)\n"
       "        return loaded\n")


def patch(s: str) -> str:
    if MARK in s:
        return s
    if s.count(OLD) != 1:
        raise SystemExit(f"patch_mtp_kern: anchor count {s.count(OLD)} != 1")
    if "\nimport os\n" not in s:
        raise SystemExit("patch_mtp_kern: mtp_patched.py does not import os")
    s = s.replace(OLD, NEW, 1)
    ast.parse(s)
    return s


def main() -> None:
    s = open(PATH).read()
    out = patch(s)
    if out == s:
        print("patch_mtp_kern: already applied")
        return
    open(PATH, "w").write(out)
    print("patch_mtp_kern: applied (R4 hook)")


if __name__ == "__main__":
    main()
