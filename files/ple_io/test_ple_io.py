#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU test: ple_io.gather() gives the same bytes as the plain mmap gather.

    python3 files/ple_io/test_ple_io.py                  # synthetic table
    python3 files/ple_io/test_ple_io.py --table PATH     # the real packed_u8

For each mode (fadvise, none) and trace state (off, on), the test gathers
random ids and compares the output to torch.index_select over a separate
np.memmap of the same file. The real table is opened read-only.
"""
import argparse
import importlib.util
import json
import os
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_module():
    spec = importlib.util.spec_from_file_location("ple_io", os.path.join(HERE, "ple_io.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    tmp = None
    if args.table:
        meta = json.load(open(args.table + ".json"))
        rows, width = int(meta["total_rows"]), int(meta["row_width"])
        path = args.table
    else:
        rows, width = 700_001, 90
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".packed_u8")
        rng = np.random.default_rng(args.seed)
        rng.integers(0, 256, size=(rows, width), dtype=np.uint8).tofile(tmp.name)
        path = tmp.name

    ple_io = load_module()
    mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(rows, width))
    table = torch.from_numpy(mm)
    ref_mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(rows, width))
    ref_table = torch.from_numpy(ref_mm)
    fd = os.open(path, os.O_RDONLY)
    trace_dir = tempfile.mkdtemp(prefix="ple_io_trace_")
    gen = torch.Generator().manual_seed(args.seed)

    fails = 0
    checks = 0
    sizes = [1, 7, 16, 112, 1000, 32768]
    for mode in ("fadvise", "none"):
        for trace in ("", trace_dir):
            ple_io.MODE, ple_io.TRACE_DIR, ple_io.TRACE = mode, trace, bool(trace)
            for r in range(args.rounds):
                n = sizes[r % len(sizes)]
                ids = torch.randint(0, rows, (n,), generator=gen, dtype=torch.int64)
                if r % 3 == 0:  # repeated ids and the table ends
                    ids[: n // 2] = ids[n // 2 : n // 2 * 2]
                    ids[0] = 0
                    ids[-1] = rows - 1
                out = torch.full((n, width), 0xA5, dtype=torch.uint8)
                ple_io.gather(fd, table, ids, out)
                ref = torch.index_select(ref_table, 0, ids)
                checks += 1
                if not torch.equal(out, ref):
                    fails += 1
                    print(f"FAIL mode={mode} trace={bool(trace)} n={n}")
            # Empty gather: nothing to read, output untouched in shape.
            out = torch.empty((0, width), dtype=torch.uint8)
            ple_io.gather(fd, table, torch.empty(0, dtype=torch.int64), out)
    os.close(fd)
    if tmp is not None:
        os.unlink(tmp.name)
    print(f"ple_io gather exactness: {checks - fails}/{checks} identical "
          f"(table {'real' if args.table else 'synthetic'}, {rows} rows x {width} B)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
