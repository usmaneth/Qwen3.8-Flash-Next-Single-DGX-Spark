#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""B1 microbench: the prefetch backends on the real PLE table (no server).

Run it as root in the serving image (lease step 0), with no other process
that maps the table:

    docker run --rm --entrypoint python3 --cap-add SYS_NICE --cap-add SYS_PTRACE \
      --ulimit memlock=-1 -v <files/ple_io>:/t -v <table dir>:<table dir>:ro \
      -v <out>:/out IMAGE /t/bench_ple_io.py --table <packed_u8> --out /out

Id sets (synthetic and seeded, no traffic data):
  P   131,072 rows drawn with replacement from 50,000 unique random rows
      (about 51K unique pages, as one 8K prefill chunk)
  D   112 unique random rows (one 1-stream verify step)
  D4  448 random rows (one 4-stream verify step)
Configs: fadvise (the shipped loop), c and pm with threads 1, 2, 4, 8.
States: cold (MADV_PAGEOUT over this mapping, then POSIX_FADV_DONTNEED over
the file) and warm (the same ids read one time before).
Each measurement: pages + advise (the prefetch call) and select (the
index_select that follows, which waits for the reads). Each output is
compared to a plain np.memmap index_select (G1 for the backend).

Output: <out>/bench.jsonl (one row per measurement), <out>/b1_pick.json.
b1_pick: the config with the lowest cold P median total. A config within
10% of the best with fewer threads wins. drop=true when no batch config
beats fadvise by at least 10 ms on cold P.
"""
import argparse
import ctypes
import importlib.util
import json
import mmap
import os
import statistics as st
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MADV_PAGEOUT = getattr(mmap, "MADV_PAGEOUT", 21)


def load_ple_io():
    os.environ["VLLM_PLE_IO_TRACE_DIR"] = ""
    spec = importlib.util.spec_from_file_location("ple_io", os.path.join(HERE, "ple_io.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def other_mappers(path: str) -> list[int]:
    name = os.path.basename(path)
    me = os.getpid()
    pids = []
    for p in os.listdir("/proc"):
        if not p.isdigit() or int(p) == me:
            continue
        try:
            if name in open(f"/proc/{p}/maps").read():
                pids.append(int(p))
        except OSError:
            pass
    return pids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--threads", default="1,2,4,8")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    others = other_mappers(args.table)
    if others:
        print(f"ERROR: other processes map the table: {others}", file=sys.stderr)
        return 2

    ple_io = load_ple_io()
    meta = json.load(open(args.table + ".json"))
    rows, width = int(meta["total_rows"]), int(meta["row_width"])
    mm = np.memmap(args.table, dtype=np.uint8, mode="r", shape=(rows, width))
    mm._mmap.madvise(mmap.MADV_RANDOM)
    table = torch.from_numpy(mm)
    ref_mm = np.memmap(args.table, dtype=np.uint8, mode="r", shape=(rows, width))
    ref_table = torch.from_numpy(ref_mm)
    fd = os.open(args.table, os.O_RDONLY)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p)

    rng = np.random.default_rng(args.seed)
    base = rng.choice(rows, size=50_000, replace=False)
    sets = {
        "P": torch.from_numpy(rng.choice(base, size=131_072, replace=True).astype(np.int64)),
        "D": torch.from_numpy(rng.choice(rows, size=112, replace=False).astype(np.int64)),
        "D4": torch.from_numpy(rng.choice(rows, size=448, replace=True).astype(np.int64)),
    }
    for k, ids in sets.items():
        pages = ple_io._np_pages(ids.numpy(), width)
        print(f"set {k}: {ids.numel()} rows, {pages.size} unique pages", flush=True)

    threads = [int(x) for x in args.threads.split(",")]
    cfgs = [("fadvise", "-", 1)] + [(("batch"), b, t) for b in ("c", "pm") for t in threads]

    def evict():
        t0 = time.perf_counter()
        ref_mm._mmap.madvise(mmap.MADV_DONTNEED)
        mm._mmap.madvise(MADV_PAGEOUT)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return time.perf_counter() - t0

    def resident(ids) -> float:
        pages = ple_io._np_pages(ids.numpy(), width)
        sample = pages[:: max(1, pages.size // 256)]
        vec = ctypes.create_string_buffer(1)
        hit = 0
        for p in sample.tolist():
            if libc.mincore(table.data_ptr() + (p << 12), 4096, vec) == 0:
                hit += vec.raw[0] & 1
        return hit / max(1, len(sample))

    out_rows = []
    fails = 0
    fjs = open(os.path.join(args.out, "bench.jsonl"), "a")
    for rep in range(args.reps):
        for sname, ids in sets.items():
            order = cfgs if rep % 2 == 0 else cfgs[::-1]
            for mode, backend, thr in order:
                for state in ("cold", "warm"):
                    if state == "cold":
                        ev_s = evict()
                        res0 = resident(ids)
                    else:
                        ev_s = 0.0
                        torch.index_select(table, 0, ids)  # warm the pages
                        res0 = resident(ids)
                    ple_io.MODE, ple_io.BACKEND, ple_io.THREADS = mode, (backend if backend != "-" else "c"), thr
                    err0 = ple_io._batch_errors
                    out = torch.empty((ids.numel(), width), dtype=torch.uint8)
                    t0 = time.perf_counter_ns()
                    if mode == "fadvise":
                        pg = ple_io._pages(ids, width)
                        t1 = time.perf_counter_ns()
                        ple_io._advise(fd, pg)
                        npages = pg.numel()
                    else:
                        t1 = t0
                        npages = ple_io._prefetch_batch(fd, table, ids)
                    t2 = time.perf_counter_ns()
                    torch.index_select(table, 0, ids, out=out)
                    t3 = time.perf_counter_ns()
                    ok = torch.equal(out, torch.index_select(ref_table, 0, ids))
                    # Unmap the reference pages again: a page that two mappings
                    # share is skipped by MADV_PAGEOUT, so the next cold state
                    # would not be cold.
                    ref_mm._mmap.madvise(mmap.MADV_DONTNEED)
                    fell_back = ple_io._batch_errors != err0
                    if not ok or fell_back:
                        fails += 1
                    r = {"rep": rep, "set": sname, "mode": mode, "backend": backend,
                         "threads": thr, "state": state, "rows": ids.numel(),
                         "pages": npages, "resident_before": round(res0, 3),
                         "pages_ms": (t1 - t0) / 1e6, "advise_ms": (t2 - t1) / 1e6,
                         "select_ms": (t3 - t2) / 1e6, "total_ms": (t3 - t0) / 1e6,
                         "evict_s": round(ev_s, 3), "bytes_ok": ok, "fell_back": fell_back}
                    if state == "cold" and npages and npages > 0:
                        r["pages_per_s"] = round(npages / ((t3 - t0) / 1e9))
                    out_rows.append(r)
                    fjs.write(json.dumps(r) + "\n")
                    fjs.flush()
            print(f"rep {rep} set {sname} done", flush=True)
    fjs.close()

    # Summary: median total per set, config and state.
    def key(r):
        return (r["set"], r["mode"], r["backend"], r["threads"], r["state"])
    groups: dict = {}
    for r in out_rows:
        groups.setdefault(key(r), []).append(r)
    summ = []
    for k, rs in sorted(groups.items()):
        summ.append({"set": k[0], "cfg": f"{k[1]}-{k[2]}-{k[3]}", "state": k[4],
                     "n": len(rs), "total_ms_med": round(st.median(x["total_ms"] for x in rs), 3),
                     "total_ms_min": round(min(x["total_ms"] for x in rs), 3),
                     "total_ms_max": round(max(x["total_ms"] for x in rs), 3),
                     "pages_ms_med": round(st.median(x["pages_ms"] for x in rs), 3),
                     "advise_ms_med": round(st.median(x["advise_ms"] for x in rs), 3),
                     "select_ms_med": round(st.median(x["select_ms"] for x in rs), 3),
                     "resident_before_med": round(st.median(x["resident_before"] for x in rs), 3),
                     "pages_med": st.median(x["pages"] for x in rs)})
    for s in summ:
        print(f"{s['set']:3s} {s['cfg']:16s} {s['state']:4s} total {s['total_ms_med']:9.3f} ms "
              f"[{s['total_ms_min']:.3f}-{s['total_ms_max']:.3f}] pages {s['pages_ms_med']:.3f} "
              f"advise {s['advise_ms_med']:.3f} select {s['select_ms_med']:.3f} res0 {s['resident_before_med']}")

    cold_p = {s["cfg"]: s for s in summ if s["set"] == "P" and s["state"] == "cold"}
    base_ms = cold_p["fadvise---1"]["total_ms_med"] if "fadvise---1" in cold_p else None
    cands = [(v["total_ms_med"], k) for k, v in cold_p.items() if k.startswith("batch")]
    cands.sort()
    pick = None
    if cands:
        best_ms = cands[0][0]
        near = [(int(k.rsplit("-", 1)[1]), ms, k) for ms, k in cands if ms <= best_ms * 1.10]
        near.sort()
        pick = near[0][2]
    pick_doc = {"fadvise_cold_P_ms": base_ms, "summary": summ, "fails": fails}
    if pick:
        _, backend, thr = pick.split("-")
        pick_doc.update(backend=backend, threads=int(thr),
                        cold_P_ms=cold_p[pick]["total_ms_med"],
                        gain_ms=round(base_ms - cold_p[pick]["total_ms_med"], 3) if base_ms else None)
        pick_doc["drop"] = not (base_ms and base_ms - cold_p[pick]["total_ms_med"] >= 10.0)
    else:
        pick_doc["drop"] = True
    json.dump(pick_doc, open(os.path.join(args.out, "b1_pick.json"), "w"), indent=1)
    print("b1_pick:", json.dumps({k: pick_doc.get(k) for k in
                                   ("backend", "threads", "cold_P_ms", "fadvise_cold_P_ms",
                                    "gain_ms", "drop", "fails")}), flush=True)
    os.close(fd)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
