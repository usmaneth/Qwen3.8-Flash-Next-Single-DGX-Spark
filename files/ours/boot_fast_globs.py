#!/usr/bin/env python3
"""Checkpoint file globs for the boot-fast loader knobs (host python, no vLLM).

The loader reads every checkpoint file in each process that loads weights.
Two of the three processes need only a few files:

  * The MTP drafter keeps only the names for which _remap_mtp_weight_name
    (files/mtp_patched.py) returns a value. Today these are in shards 33, 34.
  * The PLE offload worker keeps only the non-packed PLE names. Today these
    are in shards 1, 19.

This helper reads the snapshot index and prints one glob for each process.
default_loader._prepare_weights stops at the first allow pattern that
matches a file, so a filter must be one glob. The builder makes one glob
with one character class per position. Thus the glob matches the file set or
a superset, never less. If the file names do not all have the same length,
the builder gives no glob, and start.sh does not set the knob.

    boot_fast_globs.py <snapshot-dir> --mtp-source files/mtp_patched.py

Output (stdout), one line for each glob that the builder can make:
    MTP_GLOB=<glob>
    PLE_GLOB=<glob>
Information lines go to stderr. Exit 0 always; a missing line means "no glob".
"""
import argparse
import ast
import fnmatch
import json
import os
import re
import sys

INDEX = "model.safetensors.index.json"
PACKED_PLE = re.compile(
    r"\.ple\.ple_embedding\.ngram_embedding\.shard_\d+\.(weight|weight_scale)$")
GLOB_META = set("*?[]!")


def load_weight_map(snapshot):
    with open(os.path.join(snapshot, INDEX)) as fh:
        return json.load(fh)["weight_map"]


def load_remap(mtp_source):
    """Return _remap_mtp_weight_name, compiled from the patched MTP source.

    The function uses only str methods. The helper compiles only that one
    function, so the host needs no vLLM and uses the same rule as the image.
    """
    with open(mtp_source) as fh:
        src = fh.read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_remap_mtp_weight_name":
            mod = ast.Module(body=[node], type_ignores=[])
            ns = {}
            exec(compile(mod, mtp_source, "exec"), ns)
            return ns["_remap_mtp_weight_name"]
    raise SystemExit(f"_remap_mtp_weight_name not found in {mtp_source}")


def files_with(weight_map, pred):
    return sorted({f for name, f in weight_map.items() if pred(name)})


def make_glob(files):
    """One glob, one character class per position, that matches all files.

    Returns None if the set is empty, if the names differ in length, or if a
    name holds a glob character.
    """
    if not files:
        return None
    width = len(files[0])
    if any(len(f) != width for f in files):
        return None
    if any(ch in GLOB_META for f in files for ch in f):
        return None
    out = []
    for pos in range(width):
        chars = sorted({f[pos] for f in files})
        out.append(chars[0] if len(chars) == 1 else "[" + "".join(chars) + "]")
    return "".join(out)


def check_glob(glob, need, universe):
    """The glob must match every needed file. Return the extra files it matches."""
    matched = {f for f in universe if fnmatch.fnmatchcase(f, glob)}
    missing = set(need) - matched
    if missing:
        raise ValueError(f"glob {glob} misses {sorted(missing)}")
    return sorted(matched - set(need))


def mtp_files(weight_map, remap):
    return files_with(weight_map, lambda n: remap(n) is not None)


def ple_offload_files(weight_map):
    return files_with(weight_map, lambda n: ".ple." in n and not PACKED_PLE.search(n))


def ple_only_files(weight_map):
    """Files in which every name is a packed PLE n-gram tensor."""
    by_file = {}
    for name, f in weight_map.items():
        by_file.setdefault(f, []).append(name)
    return sorted(f for f, names in by_file.items()
                  if all(PACKED_PLE.search(n) for n in names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--mtp-source", required=True)
    args = ap.parse_args()
    wm = load_weight_map(args.snapshot)
    universe = sorted(set(wm.values()))
    on_disk = sorted(os.listdir(args.snapshot))
    remap = load_remap(args.mtp_source)
    for key, need in (("MTP_GLOB", mtp_files(wm, remap)),
                      ("PLE_GLOB", ple_offload_files(wm))):
        glob = make_glob(need)
        if glob is None:
            print(f"boot-fast: no {key} for files {need}", file=sys.stderr)
            continue
        try:
            extra = check_glob(glob, need, universe)
            # The loader globs the directory, then keeps only index files.
            check_glob(glob, need, on_disk)
        except ValueError as exc:
            print(f"boot-fast: {key} rejected: {exc}", file=sys.stderr)
            continue
        print(f"boot-fast: {key} {glob} needs {len(need)} files {need}; "
              f"extra {extra}", file=sys.stderr)
        print(f"{key}={glob}")
    skip = ple_only_files(wm)
    print(f"boot-fast: {len(skip)} PLE-only files: {skip[:1]}..{skip[-1:]}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
