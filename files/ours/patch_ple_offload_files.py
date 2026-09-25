#!/usr/bin/env python3
"""Load the PLE offload worker from its own checkpoint files (boot-fast R6),
and add the offload side of the verification hash (G1).

Runs after files/patch_ple_offload.py on files/ple_offload/worker.py.
Idempotent: a second run changes nothing.

Why: the offload worker reads the names of all 35 checkpoint files (about
300,000 get_tensor calls, 45 s) to find about 10 non-packed PLE tensors. The
packed n-gram tensors come from the memory-mapped packed table. Today the
non-packed PLE names are in shards 1 and 19.

With VLLM_PLE_OFFLOAD_FILE_GLOB set (start.sh computes it from the snapshot
index, files/ours/boot_fast_globs.py, only when the packed table exists), the
worker sets allow_patterns_overrides on its meta model before
get_all_weights. The worker's own check stays: it raises when a materialized
PLE parameter is not loaded.

With VLLM_BOOT_HASH_DIR set, the worker writes a hash manifest of the offload
layers after the packed table is attached and the process step ran
(files/ours/boot_hash.py, mounted as vllm/boot_hash.py).
"""
import os
import sys

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "ple_offload", "worker.py")
MARK = "VLLM_PLE_OFFLOAD_FILE_GLOB"

OLD_GET = '''        elif isinstance(loader, DefaultModelLoader):
            all_weights = loader.get_all_weights(model_config, model)
'''
NEW_GET = '''        elif isinstance(loader, DefaultModelLoader):
            # boot-fast R6: read only the files that hold non-packed PLE names.
            _bf_glob = os.environ.get("VLLM_PLE_OFFLOAD_FILE_GLOB", "")
            if _bf_glob and packed_tables:
                model.allow_patterns_overrides = [_bf_glob]
                logger.info("boot-fast: PLE offload worker reads only %s", _bf_glob)
            all_weights = loader.get_all_weights(model_config, model)
'''

OLD_DONE = '''        self._layers.update(offload_layers)
        del model
        logger.info("PLE weight loading complete.")
'''
NEW_DONE = '''        self._layers.update(offload_layers)
        del model
        if os.environ.get("VLLM_BOOT_HASH_DIR", ""):
            from vllm.boot_hash import dump_offload

            dump_offload(offload_layers, packed_tables)
        logger.info("PLE weight loading complete.")
'''


def main():
    s = open(PATH).read()
    if MARK in s:
        print("patch_ple_offload_files: already applied")
        return
    for old, new in ((OLD_GET, NEW_GET), (OLD_DONE, NEW_DONE)):
        if s.count(old) != 1:
            sys.exit(f"patch_ple_offload_files: anchor found {s.count(old)} times: {old[:60]!r}")
        s = s.replace(old, new, 1)
    open(PATH, "w").write(s)
    print("patch_ple_offload_files: applied")


if __name__ == "__main__":
    main()
