#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU test (gate G1): ple_io.gather() gives the same bytes as a plain mmap gather.

    python3 files/ple_io/test_ple_io.py                  # synthetic table
    python3 files/ple_io/test_ple_io.py --table PATH     # the real packed_u8
    python3 files/ple_io/test_ple_io.py --quick          # fewer rounds

For each mode and backend (fadvise, none, batch-c, batch-pm, with 1 and 4
threads) and trace state (off, on), the test gathers ids and compares the
output to torch.index_select over a separate np.memmap of the same file.
Cases: 0 ids, 1 id, duplicate ids, rows that cross a page boundary, the last
row of the table, and 112, 448 and 131,072 rows. A batch backend that falls
back to the fadvise loop is a failure, so the test proves that the backend
ran. It also checks the ctl parser and ple_io.digest(). The real table is
opened read-only.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_module(trace_dir: str = ""):
    os.environ["VLLM_PLE_IO_TRACE_DIR"] = trace_dir
    spec = importlib.util.spec_from_file_location("ple_io", os.path.join(HERE, "ple_io.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_native() -> str:
    """Compile ple_io_native.c into build/ with the local cc (tests only)."""
    out = os.path.join(HERE, "build", "libple_io_native.so")
    src = os.path.join(HERE, "ple_io_native.c")
    if os.environ.get("VLLM_PLE_IO_NATIVE"):
        return os.environ["VLLM_PLE_IO_NATIVE"]
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(src):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        subprocess.check_call(["cc", "-O2", "-Wall", "-shared", "-fPIC", "-pthread",
                               "-o", out, src])
    return out


def id_cases(rows: int, width: int, gen: torch.Generator, rounds: int):
    """(label, ids) cases. The ids are int64."""
    cases = [("empty", torch.empty(0, dtype=torch.int64)),
             ("one", torch.tensor([rows // 3], dtype=torch.int64)),
             ("last", torch.tensor([rows - 1, 0, rows - 1], dtype=torch.int64))]
    # rows that cross a page boundary
    cross = [r for r in range(0, min(rows, 20000)) if (r * width) >> 12 != (r * width + width - 1) >> 12]
    cases.append(("cross", torch.tensor(cross[:500], dtype=torch.int64)))
    for r in range(rounds):
        for n in (112, 448, 131072):
            ids = torch.randint(0, rows, (n,), generator=gen, dtype=torch.int64)
            if r % 2 == 0:  # duplicates and the table ends
                ids[: n // 2] = ids[n // 2: n // 2 * 2]
                ids[0] = 0
                ids[-1] = rows - 1
            cases.append((f"rand{n}", ids))
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.rounds = 1

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

    os.environ["VLLM_PLE_IO_NATIVE"] = build_native()
    trace_dir = tempfile.mkdtemp(prefix="ple_io_trace_")
    ple_io = load_module(trace_dir)
    mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(rows, width))
    table = torch.from_numpy(mm)
    ref_mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(rows, width))
    ref_table = torch.from_numpy(ref_mm)
    fd = os.open(path, os.O_RDONLY)
    gen = torch.Generator().manual_seed(args.seed)
    cases = id_cases(rows, width, gen, args.rounds)

    fails = checks = 0
    # batch_min 0: every gather uses the backend. batch_min 4096: the small
    # cases use the fadvise loop and the 131,072-row case uses the backend.
    arms = [("fadvise", "c", 1, 0), ("none", "c", 1, 0), ("batch", "c", 1, 0),
            ("batch", "c", 4, 0), ("batch", "pm", 1, 0), ("batch", "pm", 4, 0),
            ("batch", "c", 4, 4096), ("batch", "pm", 8, 4096)]
    for mode, backend, threads, bmin in arms:
        for tr in (False, True):
            ple_io.MODE, ple_io.BACKEND, ple_io.THREADS = mode, backend, threads
            ple_io.BATCH_MIN = bmin
            ple_io.TRACE, ple_io.TRACE_RES = tr, (128 if tr else 0)
            err0 = ple_io._batch_errors
            for label, ids in cases:
                n = ids.numel()
                out = torch.full((n, width), 0xA5, dtype=torch.uint8)
                ple_io.gather(fd, table, ids, out)
                ref = torch.index_select(ref_table, 0, ids)
                checks += 1
                if not torch.equal(out, ref):
                    fails += 1
                    print(f"FAIL bytes mode={mode} backend={backend} threads={threads} "
                          f"trace={tr} case={label} n={n}")
            if mode == "batch" and ple_io._batch_errors != err0:
                fails += 1
                print(f"FAIL fallback mode={mode} backend={backend}: the backend "
                      f"did not run ({ple_io._batch_errors - err0} errors)")
            if ple_io._batch_off:
                fails += 1
                print("FAIL the batch path is off")

    # Concurrent callers: 4 threads gather at the same time with the pm
    # backend (shared iovec buffer and pool). The bytes and the backend
    # must stay correct, with no fallback.
    import threading
    ple_io.MODE, ple_io.BACKEND, ple_io.THREADS, ple_io.BATCH_MIN = "batch", "pm", 4, 0
    ple_io.TRACE, ple_io.TRACE_RES = False, 0
    err0 = ple_io._batch_errors
    bad: list[str] = []
    seeds = [args.seed + 100 + k for k in range(4)]

    def worker(seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        for it in range(4 if args.quick else 12):
            n = (4096, 20000, 131072)[it % 3]
            ids = torch.randint(0, rows, (n,), generator=g, dtype=torch.int64)
            out = torch.empty((n, width), dtype=torch.uint8)
            ple_io.gather(fd, table, ids, out)
            if not torch.equal(out, torch.index_select(ref_table, 0, ids)):
                bad.append(f"seed {seed} it {it} n {n}")

    def run_threads() -> None:
        ths = [threading.Thread(target=worker, args=(sd,)) for sd in seeds]
        for t in ths:
            t.start()
        for t in ths:
            t.join()

    run_threads()
    checks += 1
    if bad or ple_io._batch_errors != err0 or ple_io._batch_off:
        fails += 1
        print(f"FAIL concurrent gathers: {bad[:3]} errors "
              f"{ple_io._batch_errors - err0} off {ple_io._batch_off}")

    # The advice itself: with threads=1 each _run() call runs in the caller
    # thread. The stand-in sleeps before it reads the shared iovec buffer,
    # then compares the pages to the pages of its own ids. Without the lock
    # another thread writes the buffer during the sleep.
    import time as _time
    ple_io.THREADS = 1
    real_run = ple_io._PM._run
    want_pages = threading.local()
    advice_bad: list[str] = []

    def run_check(self, iov, lo, hi):
        _time.sleep(0.002)
        got = iov[lo:hi, 0] - np.uint64(table.data_ptr())
        if not np.array_equal(got >> np.uint64(12), want_pages.v[lo:hi]):
            advice_bad.append(f"{threading.current_thread().name} [{lo},{hi})")
        return real_run(self, iov, lo, hi)

    real_gather = ple_io.gather

    def gather_check(fd_, t_, ids_, out_):
        want_pages.v = ple_io._np_pages(ids_.numpy().astype(np.int64), width).astype(np.uint64)
        real_gather(fd_, t_, ids_, out_)

    ple_io._PM._run = run_check
    ple_io.gather = gather_check
    try:
        run_threads()
    finally:
        ple_io._PM._run = real_run
        ple_io.gather = real_gather
    checks += 1
    if advice_bad or bad:
        fails += 1
        print(f"FAIL concurrent advice: {len(advice_bad)} runs saw another "
              f"thread's pages {advice_bad[:3]}")

    # batch_min: a gather below it does not call the backend, one at or
    # above it does.
    calls = []
    real_batch = ple_io._prefetch_batch
    ple_io._prefetch_batch = lambda fd_, t_, i_: calls.append(i_.numel()) or real_batch(fd_, t_, i_)
    ple_io.MODE, ple_io.BACKEND, ple_io.THREADS, ple_io.BATCH_MIN = "batch", "pm", 1, 4096
    for tr in (False, True):
        ple_io.TRACE, ple_io.TRACE_RES = tr, 0
        for n in (112, 4095, 4096):
            ids = torch.randint(0, rows, (n,), generator=gen, dtype=torch.int64)
            out = torch.empty((n, width), dtype=torch.uint8)
            ple_io.gather(fd, table, ids, out)
            checks += 1
            if not torch.equal(out, torch.index_select(ref_table, 0, ids)):
                fails += 1
                print(f"FAIL bytes batch_min case n={n} trace={tr}")
    ple_io._prefetch_batch = real_batch
    ple_io.TRACE = False
    checks += 1
    if calls != [4096, 4096]:
        fails += 1
        print("FAIL batch_min: backend calls", calls)

    # The ctl file: values, the id echo, an invalid value, a partial line.
    ctl = os.path.join(trace_dir, "ctl")
    with open(ctl, "w") as f:
        f.write("mode=batch\nbackend=pm\nthreads=8\nbatch_min=100\ndefer=1\nfast=1\ntrace=0\nres=0\nid=t1\n")
    ple_io._read_ctl()
    got = (ple_io.MODE, ple_io.BACKEND, ple_io.THREADS, ple_io.BATCH_MIN, ple_io.DEFER,
           ple_io.FAST, ple_io.TRACE, ple_io.TRACE_RES, ple_io.CTL_ID)
    checks += 1
    if got != ("batch", "pm", 8, 100, 1, 1, False, 0, "t1"):
        fails += 1
        print("FAIL ctl parse", got)
    with open(ctl, "w") as f:
        f.write("mode=bogus\nthreads=0\nid=t2\n")
    ple_io._read_ctl()
    checks += 1
    want = (ple_io.DEFAULTS["mode"], int(ple_io.DEFAULTS["threads"]), int(ple_io.DEFAULTS["defer"]))
    if (ple_io.MODE, ple_io.THREADS, ple_io.DEFER) != want:
        fails += 1
        print("FAIL ctl defaults", ple_io.MODE, ple_io.THREADS, ple_io.DEFER)
    with open(ctl, "w") as f:
        f.write("mode=batch")  # no newline: a partial write, not applied
    checks += 1
    if ple_io._read_ctl() or ple_io.MODE != ple_io.DEFAULTS["mode"]:
        fails += 1
        print("FAIL partial ctl applied")
    for t in ple_io._traces.values():
        t.flush()
    ctl_rows = [l for l in open([os.path.join(trace_dir, p) for p in os.listdir(trace_dir)
                                 if p.startswith("ctl-")][0]) if not l.startswith("#")]
    checks += 1
    if len(ctl_rows) != 2 or not ctl_rows[0].rstrip().endswith("t1"):
        fails += 1
        print("FAIL ctl records", ctl_rows)

    # digest(): the same bytes give the same digest, one changed byte does not.
    a = torch.arange(4096, dtype=torch.int32).reshape(64, 64)
    b = a.clone()
    checks += 1
    if ple_io.digest(a) != ple_io.digest(b) or ple_io.digest(a[:, :32]) != ple_io.digest(b[:, :32].contiguous()):
        fails += 1
        print("FAIL digest equal")
    b[3, 5] += 1
    checks += 1
    if ple_io.digest(a) == ple_io.digest(b):
        fails += 1
        print("FAIL digest differs")

    # launch_wait() with a CPU stand-in for the connector: the wait runs, the
    # launch record and the GPU-side digest record are written.
    class _Layer:
        _gpu_output_buffer = torch.randint(0, 256, (64, 2560), dtype=torch.uint8)

    class _Conn:
        device = torch.device("cpu")
        _layers = {"layers.1.ple": _Layer()}
        waited = []

        def _wait_done(self, seq):
            self.waited.append(seq)

    conn = _Conn()
    ple_io.TRACE, ple_io.CHECK = True, 1
    for seq in (1, 2):
        ple_io.launch_wait(conn, seq, 7, 1, ple_io.now(), ple_io.now(), defer=seq - 1)
    ple_io._drain_launches()
    ple_io.check_cpu(2, "layers.1.ple", _Layer._gpu_output_buffer[:7, :1440])
    ple_io.TRACE, ple_io.CHECK = False, 0
    for t in ple_io._traces.values():
        t.flush()

    def rows_of(kind):
        path = [os.path.join(trace_dir, p) for p in os.listdir(trace_dir) if p.startswith(kind + "-")][0]
        lines = [l.rstrip("\n").split("\t") for l in open(path) if not l.startswith("#")]
        hdr = open(path).readline()[2:].split()
        return [dict(zip(hdr, l)) for l in lines]

    la, ck = rows_of("launch"), rows_of("check")
    checks += 1
    if conn.waited != [1, 2] or len(la) != 2 or la[1]["defer"] != "1" or la[0]["gap_us"] != "-1":
        fails += 1
        print("FAIL launch records", conn.waited, la)
    gpu = [r for r in ck if r["side"] == "gpu" and r["seq"] == "2"]
    cpu = [r for r in ck if r["side"] == "cpu" and r["seq"] == "2"]
    checks += 1
    if not gpu or not cpu or gpu[0]["digest"] != cpu[0]["digest"] or gpu[0]["width"] != "1440":
        fails += 1
        print("FAIL check records", gpu, cpu)

    os.close(fd)
    if tmp is not None:
        os.unlink(tmp.name)
    print(f"ple_io gather exactness (G1): {checks - fails}/{checks} identical "
          f"(table {'real' if args.table else 'synthetic'}, {rows} rows x {width} B, "
          f"native {os.environ['VLLM_PLE_IO_NATIVE']}, pm probe "
          f"{'ok' if ple_io._pm.pidfd >= 0 else ple_io._pm.error})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
