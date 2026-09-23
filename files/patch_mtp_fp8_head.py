#!/usr/bin/env python3
"""Add an FP8 copy of the reduced MTP draft head to files/mtp_patched.py.

Runs after patch_mtp_draft_vocab.py. Inert unless VLLM_MTP_DRAFT_HEAD_FP8=1 in
the container environment. Idempotent: a second run changes nothing.

The draft head is a batch-1 GEMV over the reduced vocabulary each draft step.
On GB10 the rowwise FP8 torch._scaled_mm measured 0.83 ms against 1.96 ms for
the BF16 linear at 47184 x 2560. Only the argmax is used and the target verifies
every draft token, so output stays exact.
"""
import os, sys

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mtp_patched.py")
MARK = "_attach_fp8_draft_head"

HELPER = '''

def _attach_fp8_draft_head(model: nn.Module) -> None:
    """FP8 (e4m3, per-row scale) copy of the reduced draft head.

    Rows are padded to a multiple of 16 for torch._scaled_mm; get_top_tokens
    drops the pad columns before the argmax.
    """
    w = model._draft_lm_head_weight
    rows = w.shape[0]
    pad = (-rows) % 16
    wf = w.float()
    if pad:
        wf = torch.cat([wf, wf.new_zeros(pad, wf.shape[1])], dim=0)
    scale = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
    model.register_buffer("_draft_head_fp8", (wf / scale).to(torch.float8_e4m3fn), persistent=False)
    model.register_buffer("_draft_head_fp8_scale", scale.t().contiguous(), persistent=False)
    model._draft_head_rows = rows
    logger.info("MTP draft head: FP8 rowwise copy engaged (%d rows, %d pad).", rows, pad)


def _remap_ignored_layers('''

OLD_ATTACH = '''    full_gib = weight.numel() * weight.element_size() / 2**30
    cut_gib = model._draft_lm_head_weight.numel() * weight.element_size() / 2**30'''
NEW_ATTACH = '''    if os.environ.get("VLLM_MTP_DRAFT_HEAD_FP8", "0") == "1":
        _attach_fp8_draft_head(model)
''' + OLD_ATTACH

OLD_TOP = '''        logits = torch.nn.functional.linear(hidden_states.to(weight.dtype), weight)
        return self._draft_id_to_target_id[logits.argmax(dim=-1)].to(torch.long)'''
NEW_TOP = '''        w8 = getattr(self, "_draft_head_fp8", None)
        if w8 is not None:
            x = hidden_states.float()
            xs = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
            logits = torch._scaled_mm(
                (x / xs).to(torch.float8_e4m3fn), w8.t(),
                scale_a=xs, scale_b=self._draft_head_fp8_scale,
                out_dtype=torch.bfloat16,
            )[:, : self._draft_head_rows]
            return self._draft_id_to_target_id[logits.argmax(dim=-1)].to(torch.long)
''' + OLD_TOP

s = open(PATH).read()
if MARK in s:
    print("patch_mtp_fp8_head: already applied")
    sys.exit(0)
for old in (OLD_ATTACH, OLD_TOP, "\n\ndef _remap_ignored_layers("):
    if old not in s:
        sys.exit(f"patch_mtp_fp8_head: anchor not found: {old[:60]!r}")
s = s.replace(OLD_ATTACH, NEW_ATTACH, 1).replace(OLD_TOP, NEW_TOP, 1)
s = s.replace("\n\ndef _remap_ignored_layers(", HELPER, 1)
open(PATH, "w").write(s)
print("patch_mtp_fp8_head: applied")
