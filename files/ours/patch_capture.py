#!/usr/bin/env python3
"""Capture the target hidden states that feed the MTP drafter.

Generates ar_speculator_capture.py: the image's autoregressive speculator plus a
hook in propose(). When VLLM_MTP_CAPTURE_DIR is set, every prefill chunk of a
request whose id contains "cap-<row>" writes one file:

    <dir>/<row>_<start>.pt = {"start": start, "hidden": bf16 [n, hc_count*hidden]}

`start` is the absolute token position of the chunk's first row, so the rows
line up with the request's token ids. This is the pre-final-mixer multi-stream
state the MTP head consumes at draft step 0. Only for the offline capture
server; the A/B server never mounts this file.

    python3 patch_capture.py /path/to/extracted/vllm
"""
import os, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "/models/usman/qwen38-tune/vllm-src/vllm"
OUT = os.path.dirname(os.path.abspath(__file__))
REL = "v1/worker/gpu/spec_decode/autoregressive/speculator.py"

HELPER = '''

_CAPTURE_DIR = os.environ.get("VLLM_MTP_CAPTURE_DIR", "")


def _capture_hidden(input_batch, hidden_states) -> None:
    """Write prefill hidden-state rows for requests tagged cap-<row>."""
    qsl = input_batch.query_start_loc_np
    for i, req_id in enumerate(input_batch.req_ids[: input_batch.num_reqs]):
        if "cap-" not in req_id or not input_batch.is_prefilling_np[i]:
            continue
        row = req_id.split("cap-", 1)[1].split("-", 1)[0]
        start = int(input_batch.num_computed_tokens_np[i])
        rows = hidden_states[int(qsl[i]) : int(qsl[i + 1])]
        torch.save(
            {"start": start, "hidden": rows.detach().to("cpu", torch.bfloat16)},
            os.path.join(_CAPTURE_DIR, f"{row}_{start}.pt"),
        )
'''

ANCHOR = "        self.hidden_states[:num_tokens_padded].copy_(hidden_states)\n"
HOOK = ANCHOR + (
    "        if _CAPTURE_DIR and not dummy_run and not is_profile:\n"
    "            _capture_hidden(input_batch, hidden_states)\n"
)

s = open(os.path.join(SRC, REL)).read()
if ANCHOR not in s:
    sys.exit("anchor not found in propose()")
s = s.replace(ANCHOR, HOOK, 1)
first_def = s.index("\nclass ")
s = s[:first_def] + HELPER + s[first_def:]
if "\nimport os\n" not in s:
    s = "import os\n" + s
open(os.path.join(OUT, "ar_speculator_capture.py"), "w").write(s)
print("wrote ar_speculator_capture.py")
