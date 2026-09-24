#!/usr/bin/env python3
"""Let GDN prefill use the FlashInfer SM12x kernel on GB10 (prefill-ttft B3).

The image gate (_resolve_gdn_prefill_backend) accepts FlashInfer only on SM90
and SM10.x, so GB10 (SM12.1) always runs Triton/FLA. FlashInfer 0.6.17 has the
SM120 kernel (gdn_kernels chunk_gated_delta_rule_sm120). This is the gate of
vllm#55715: SM12.x, head_k_dim 128, CUDA runtime 13 or newer.

The call site already L2-normalizes q and k (fused_post_conv_prep, apply_l2norm)
and passes use_qk_l2norm_in_kernel=False, so the missing flashinfer#5255
(in-kernel Q/K normalization) does not apply.

Runtime flag gdn_prefill (pt_flags): "flashinfer" (default when the gate
passes) or "triton" (FLA), read once per layer call, for the A/B only.

Reads the image source from SRC and writes qwen_gdn_linear_attn.py next to
this script. The .env mounts it over the image file.

    python3 patch_gdn_fi.py [/path/to/extracted/vllm]
"""
import os
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "/models/usman/qwen38-tune/vllm-src/vllm"
OUT = os.path.dirname(os.path.abspath(__file__))
REL = "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"

EDITS = [
    ("from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig\n",
     "from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig\n\n"
     "try:  # prefill-ttft B3: runtime A/B flags (files/ours/pt_flags.py)\n"
     "    from vllm import pt_flags as _pt_flags\n"
     "except ImportError:\n"
     "    _pt_flags = None\n"),
    ("""        supports_flashinfer = True
        supports_cutedsl = True
""",
     """        supports_flashinfer = True
        supports_cutedsl = True
    elif (
        # prefill-ttft B3 (vllm#55715): FlashInfer 0.6.17 has the SM120 kernel.
        current_platform.is_device_capability_family(120)
        and head_k_dim == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        supports_flashinfer = True
"""),
    ("""        if active_backend == "flashinfer":
            self._forward_method = self.forward_cuda
""",
     """        if active_backend == "flashinfer":
            self._forward_method = self._pt_forward_switch
"""),
    ("""    def forward_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
""",
     """    def _pt_forward_switch(self, *args, **kwargs):
        \"\"\"prefill-ttft B3: runtime flag gdn_prefill (flashinfer, triton).\"\"\"
        if (
            _pt_flags is not None
            and _pt_flags.get("gdn_prefill", "flashinfer") == "triton"
        ):
            return self.forward_native(*args, **kwargs)
        return self.forward_cuda(*args, **kwargs)

    def forward_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
"""),
]


def main() -> None:
    s = open(os.path.join(SRC, REL)).read()
    for old, new in EDITS:
        if s.count(old) != 1:
            sys.exit(f"{REL}: anchor found {s.count(old)} times: {old[:70]!r}")
        s = s.replace(old, new, 1)
    out = os.path.join(OUT, "qwen_gdn_linear_attn.py")
    open(out, "w").write(s)
    print("wrote qwen_gdn_linear_attn.py (FlashInfer SM12x GDN prefill gate)")


if __name__ == "__main__":
    main()
