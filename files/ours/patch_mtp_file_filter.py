#!/usr/bin/env python3
"""Load the MTP drafter from its own checkpoint files (boot-fast R2).

Runs after patch_mtp_draft_vocab.py and patch_mtp_fp8_head.py on
files/mtp_patched.py. Idempotent: a second run changes nothing.

Why: the drafter pass reads all 35 checkpoint files (99 GB of names, about
300,000 get_tensor calls) and keeps only the names for which
_remap_mtp_weight_name returns a value. Today these are in shards 33 and 34.
The pass takes 55-66 s.

With VLLM_MTP_FILE_GLOB set (start.sh computes it from the snapshot index,
files/ours/boot_fast_globs.py), the drafter sets allow_patterns_overrides to
that one glob, and DefaultModelLoader reads only the matching files.

Guard (vLLM does not check load completeness for a quantized model):
  * With the glob set, the drafter reads the headers of every index file and
    finds the names for which _remap_mtp_weight_name returns a value: the
    names that the stock path gives to the drafter. If one of them did not
    come through the filtered iterator, the load raises.
  * The drafter parameters that load_weights did not load are logged. With
    VLLM_MTP_UNLOADED_ALLOW=<json list file>, a name out of the list raises.
    VLLM_BOOT_HASH_DIR=<dir> writes the list to mtp-unloaded-<pid>.json, so
    a stock boot (VLLM_MTP_GUARD_RECORD=1) can record the allow-list.
"""
import os
import sys

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "mtp_patched.py")
MARK = "_bf_mtp_guard"

HELPER = '''

def _bf_mtp_guard(model: nn.Module, seen: set, loaded: set) -> None:
    """boot-fast R2 guard. See files/ours/patch_mtp_file_filter.py."""
    import json
    import struct

    glob = os.environ.get("VLLM_MTP_FILE_GLOB", "")
    record = os.environ.get("VLLM_MTP_GUARD_RECORD", "0") == "1"
    if not glob and not record:
        return
    root = model.vllm_config.model_config.model
    if glob:
        with open(os.path.join(root, "model.safetensors.index.json")) as fh:
            files = sorted(set(json.load(fh)["weight_map"].values()))
        expected = set()
        for f in files:
            with open(os.path.join(root, f), "rb") as fh:
                (n,) = struct.unpack("<Q", fh.read(8))
                hdr = json.loads(fh.read(n))
            expected.update(
                k for k in hdr if k != "__metadata__"
                and _remap_mtp_weight_name(k) is not None
            )
        missing = sorted(expected - seen)
        if missing:
            raise RuntimeError(
                f"boot-fast MTP guard: glob {glob!r} lost {len(missing)} drafter "
                f"names, for example {missing[:5]}"
            )
        logger.info(
            "boot-fast MTP guard: glob %s gave all %d drafter names of %d files",
            glob, len(expected), len(files),
        )
    unloaded = sorted({n for n, _ in model.named_parameters()} - set(loaded))
    logger.info(
        "boot-fast MTP guard: %d drafter parameters not loaded from the "
        "checkpoint: %s", len(unloaded), unloaded[:8],
    )
    out_dir = os.environ.get("VLLM_BOOT_HASH_DIR", "")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"mtp-unloaded-{os.getpid()}.json"), "w") as fh:
            json.dump(unloaded, fh, indent=0)
    allow_path = os.environ.get("VLLM_MTP_UNLOADED_ALLOW", "")
    if allow_path:
        with open(allow_path) as fh:
            allow = set(json.load(fh))
        extra = sorted(set(unloaded) - allow)
        if extra:
            raise RuntimeError(
                f"boot-fast MTP guard: {len(extra)} drafter parameters not "
                f"loaded and not in {allow_path}: {extra[:5]}"
            )


def _remap_ignored_layers('''

OLD_INIT = '''        super().__init__()
        self.config = config
        self.model = Qwen3_8FlashNextMultiTokenPredictor(
'''
NEW_INIT = '''        super().__init__()
        self.config = config
        # boot-fast R2: read only the files that hold drafter names.
        if os.environ.get("VLLM_MTP_FILE_GLOB", ""):
            self.allow_patterns_overrides = [os.environ["VLLM_MTP_FILE_GLOB"]]
        self.model = Qwen3_8FlashNextMultiTokenPredictor(
'''

OLD_LOAD = '''        def remap_weight_names():
            for name, weight in weights:
                remapped_name = _remap_mtp_weight_name(name)
                if remapped_name is not None:
                    yield remapped_name, weight
'''
NEW_LOAD = '''        _bf_seen = set()

        def remap_weight_names():
            for name, weight in weights:
                remapped_name = _remap_mtp_weight_name(name)
                if remapped_name is not None:
                    _bf_seen.add(name)
                    yield remapped_name, weight
'''

OLD_ATTACH = '''        loaded = loader.load_weights(remap_weight_names())
        _attach_draft_vocab(self)
'''
NEW_ATTACH = '''        loaded = loader.load_weights(remap_weight_names())
        _bf_mtp_guard(self, _bf_seen, loaded)
        _attach_draft_vocab(self)
'''


def main():
    s = open(PATH).read()
    if MARK in s:
        print("patch_mtp_file_filter: already applied")
        return
    edits = [("\n\ndef _remap_ignored_layers(", HELPER), (OLD_INIT, NEW_INIT),
             (OLD_LOAD, NEW_LOAD), (OLD_ATTACH, NEW_ATTACH)]
    for old, new in edits:
        if s.count(old) != 1:
            sys.exit(f"patch_mtp_file_filter: anchor found {s.count(old)} times: {old[:60]!r}")
        s = s.replace(old, new, 1)
    open(PATH, "w").write(s)
    print("patch_mtp_file_filter: applied")


if __name__ == "__main__":
    main()
