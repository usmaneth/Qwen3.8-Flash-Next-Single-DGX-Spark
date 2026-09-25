#!/usr/bin/env python3
"""Add the kern-decode drafter hooks to files/mtp_patched.py (after
patch_mtp_draft_vocab.py and patch_mtp_fp8_head.py; start.sh runs it when
KERN_DECODE=1). Idempotent: a second run changes nothing.

R8: a 4-bit draft head copy and its use in get_top_tokens when
VLLM_MTP_DRAFT_HEAD_W4=1 (module flag _KERN_HEAD_W4_ON for the run-time
knob).

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


# R8: a 4-bit copy of the reduced draft head (w4a16.py) when
# VLLM_MTP_DRAFT_HEAD_W4=1; _KERN_HEAD_W4_ON switches it at run time.
MARK8 = "_KERN_HEAD_W4_ON"
OLD8A = ("    if os.environ.get(\"VLLM_MTP_DRAFT_HEAD_FP8\", \"0\") == \"1\":\n"
         "        _attach_fp8_draft_head(model)\n")
NEW8A = OLD8A + (
    "    if os.environ.get(\"VLLM_MTP_DRAFT_HEAD_W4\", \"0\") == \"1\":\n"
    "        # R8: 4-bit draft head (int4, E4M3 group-32 scales, FP32 row scale).\n"
    "        from .w4a16 import quantize_w4\n"
    "\n"
    "        model._draft_head_w4 = quantize_w4(model._draft_lm_head_weight)\n"
    "        logger.info(\"MTP draft head: W4A16 copy engaged (%d rows, on=%s).\",\n"
    "                    model._draft_lm_head_weight.shape[0], _KERN_HEAD_W4_ON)\n")
OLD8B = ("        w8 = getattr(self, \"_draft_head_fp8\", None)\n"
         "        if w8 is not None:\n")
NEW8B = ("        w4 = getattr(self, \"_draft_head_w4\", None)\n"
         "        if w4 is not None and _KERN_HEAD_W4_ON and 0 < hidden_states.shape[0] <= 32:\n"
         "            from .w4a16 import w4a16_linear\n"
         "\n"
         "            x = hidden_states.to(torch.bfloat16).reshape(-1, hidden_states.shape[-1])\n"
         "            logits = w4a16_linear(x, *w4)\n"
         "            return self._draft_id_to_target_id[logits.argmax(dim=-1)].to(torch.long)\n"
         + OLD8B)
OLD8C = "logger = init_logger(__name__)\n"
NEW8C = OLD8C + ("# R8 run-time knob (kd_ext attr): use the 4-bit draft head copy.\n"
                 "_KERN_HEAD_W4_ON = os.environ.get(\"VLLM_MTP_DRAFT_HEAD_W4\", \"0\") == \"1\"\n")


def patch(s: str) -> str:
    if MARK8 not in s:
        for old, new in ((OLD8C, NEW8C), (OLD8A, NEW8A), (OLD8B, NEW8B)):
            if s.count(old) != 1:
                raise SystemExit(f"patch_mtp_kern: R8 anchor count {s.count(old)} != 1: {old[:60]!r}")
            s = s.replace(old, new, 1)
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
