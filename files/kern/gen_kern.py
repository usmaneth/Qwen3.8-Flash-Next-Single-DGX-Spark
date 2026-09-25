#!/usr/bin/env python3
"""Generate the kern-decode image overrides from the pristine image files.

start.sh extracts the image files into files/kern/orig/ and runs this
script when KERN_DECODE=1. It writes files/kern/out/. Each output file has
one owner here, and each change inside it stays off until its env var or
kd_ext knob turns it on, so the knob-off behavior is the image behavior.

  out/nvidia_model.py   models/qwen3_8_flash_next/nvidia/model.py
                        R5: "import os" and, at the end of
                        Qwen3_8FlashNextForCausalLM.__init__, a call to
                        enable_fp8_lm_head(self) when
                        VLLM_QWEN38_LM_HEAD_FP8=1 (the generator of
                        overhead-work/bf16-gemm/patch_lm_head_fp8_rescore.py).
  out/lm_head_fp8.py    models/qwen3_8_flash_next/nvidia/lm_head_fp8.py
                        R5: a byte copy of files/kern/lm_head_fp8.py.
  out/short_conv_attn.py v1/attention/backends/short_conv_attn.py
                        R2: the blocking index copies of the spec-decode
                        metadata build go through async_tensor_h2d when
                        VLLM_SHORTCONV_ASYNC_H2D=1 (the generator of
                        overhead-work/sched-wait/gen_shortconv_async_h2d.py,
                        the same idea as upstream vLLM #55054).
  out/kd_ext.py         the worker extension (runtime knobs and the step
                        timer), a byte copy of files/kern/kd_ext.py.

    python3 gen_kern.py --orig files/kern/orig --out files/kern/out
"""
import argparse
import ast
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- R5 model.py
R5_MARK = "from .lm_head_fp8 import enable_fp8_lm_head"
R5_PAIRS = [
    (
        '"""Inference-only Qwen3.8-Flash-Next model."""\n\n'
        "from collections.abc import Iterable\n",
        '"""Inference-only Qwen3.8-Flash-Next model."""\n\n'
        "import os\n"
        "from collections.abc import Iterable\n",
    ),
    (
        "        self.set_moe_parameters(self.model.layers)\n"
        "        enable_qwen38next_low_latency_gemm(self, self.model_config.dtype)\n",
        "        self.set_moe_parameters(self.model.layers)\n"
        "        enable_qwen38next_low_latency_gemm(self, self.model_config.dtype)\n"
        '        if os.environ.get("VLLM_QWEN38_LM_HEAD_FP8", "0") == "1":\n'
        "            # FP8 screen and BF16 rescore of the target output head\n"
        "            # (lm_head_fp8.py). It is off by default.\n"
        f"            {R5_MARK}\n"
        "\n"
        "            enable_fp8_lm_head(self)\n",
    ),
]

# ------------------------------------------------------- R2 short_conv_attn.py
R2_MARK = "VLLM_SHORTCONV_ASYNC_H2D"
R2_HELPER_ANCHOR = "from vllm.v1.kv_cache_interface import MambaSpec\n"
R2_HELPER = '''

# Set VLLM_SHORTCONV_ASYNC_H2D=1 to copy the small index tensors of the
# spec-decode metadata build without a host stream sync. The default keeps
# the blocking Tensor.to() copies. kd_ext.py can set _ASYNC_H2D at run time.
_ASYNC_H2D = os.environ.get("VLLM_SHORTCONV_ASYNC_H2D", "0") == "1"


def _index_to_device(t: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """Copy a small CPU index tensor to ``device``.

    The async path pins the tensor and copies it with non_blocking=True, so
    the host does not wait for the GPU stream. The copy stays in stream order
    before every later use of the result on that stream.
    """
    if not _ASYNC_H2D or torch.device(device).type == "cpu":
        return t.to(device)
    return async_tensor_h2d(t, device)
'''
R2_PAIRS = [
    ("from dataclasses import dataclass, replace\n",
     "import os\nfrom dataclasses import dataclass, replace\n"),
    ("        spec_req_idx = spec_req_idx_cpu.to(query_start_loc.device)\n",
     "        spec_req_idx = _index_to_device(spec_req_idx_cpu, query_start_loc.device)\n"),
    ("        non_spec_req_idx = non_spec_req_idx_cpu.to(query_start_loc.device)\n",
     "        non_spec_req_idx = _index_to_device(\n"
     "            non_spec_req_idx_cpu, query_start_loc.device\n"
     "        )\n"),
    ("            req_group[decode_req_idx_cpu.to(query_start_loc.device)] = 1\n",
     "            req_group[_index_to_device(decode_req_idx_cpu, query_start_loc.device)] = 1\n"),
    ("            spec_req_idx_cpu.to(num_accepted_tokens.device)\n",
     "            _index_to_device(spec_req_idx_cpu, num_accepted_tokens.device)\n"),
    ("                non_spec_req_idx = non_spec_req_idx_cpu.to(num_computed_tokens.device)\n",
     "                non_spec_req_idx = _index_to_device(\n"
     "                    non_spec_req_idx_cpu, num_computed_tokens.device\n"
     "                )\n"),
]


def _apply(text, pairs, name):
    for old, new in pairs:
        n = text.count(old)
        if n != 1:
            raise ValueError(f"{name}: anchor count {n} != 1: {old.strip()[:70]!r}")
        text = text.replace(old, new, 1)
    return text


def patch_model(text: str) -> str:
    if R5_MARK in text:
        raise ValueError("nvidia/model.py: already patched")
    out = _apply(text, R5_PAIRS, "nvidia/model.py")
    ast.parse(out)
    return out


def patch_short_conv(text: str) -> str:
    if R2_MARK in text:
        raise ValueError("short_conv_attn.py: already patched")
    if "async_tensor_h2d" not in text:
        raise ValueError("short_conv_attn.py: the source does not import async_tensor_h2d")
    out = _apply(text, R2_PAIRS, "short_conv_attn.py")
    if out.count(R2_HELPER_ANCHOR) != 1:
        raise ValueError("short_conv_attn.py: helper anchor not found once")
    out = out.replace(R2_HELPER_ANCHOR, R2_HELPER_ANCHOR + R2_HELPER, 1)
    left = [ln for ln in out.splitlines() if "_idx_cpu.to(" in ln]
    if left:
        raise ValueError(f"short_conv_attn.py: blocking copies left: {left}")
    ast.parse(out)
    return out


# (orig file name, output name, function)
JOBS = [
    ("nvidia_model.py", "nvidia_model.py", patch_model),
    ("short_conv_attn.py", "short_conv_attn.py", patch_short_conv),
]
COPIES = ["lm_head_fp8.py", "kd_ext.py", "mtp_w8a16.py", "ple_gpu_wait.py", "w4a16.py"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default=os.path.join(HERE, "orig"))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    record = {}
    for src_name, out_name, fn in JOBS:
        src = open(os.path.join(a.orig, src_name)).read()
        try:
            out = fn(src)
        except ValueError as e:
            print(f"gen_kern: {e}", file=sys.stderr)
            return 1
        with open(os.path.join(a.out, out_name), "w") as f:
            f.write(out)
        record[out_name] = {
            "orig_sha256": hashlib.sha256(src.encode()).hexdigest(),
            "out_sha256": hashlib.sha256(out.encode()).hexdigest(),
        }
    for name in COPIES:
        path = os.path.join(HERE, name)
        ast.parse(open(path).read())
        shutil.copyfile(path, os.path.join(a.out, name))
    with open(os.path.join(a.out, "SHA256.json"), "w") as f:
        json.dump(record, f, indent=1)
    print(f"gen_kern: wrote {', '.join(sorted(os.listdir(a.out)))} to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
