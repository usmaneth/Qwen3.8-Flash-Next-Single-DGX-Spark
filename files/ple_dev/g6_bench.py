#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G6 micro bench (TECHNIQUES.md #7, PLAN.md in frontier/next/ple): the device PLE path.

    python3 g6_bench.py --golden DIR --coldmiss FILE --out DIR [--cpu-dry]

Runs in the pinned vLLM image (torch + triton), with no server on the node.
Status: the GPU part did not run yet. --cpu-dry runs the harness logic with
the numpy twin in place of the kernels, with no GPU and no eviction.

Steps, in order. A step that fails stops the run (the stop rules of PLAN.md):
  P  probe: CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS (88) and
     ..._USES_HOST_PAGE_TABLES (100) must be 1.
  K1 the id kernel on every golden step (the twin rows, which parity_real.py
     proved equal to the CPU worker): all T_pad x 16 ids bit-exact.
  K2 the gather kernel on the golden valid rows, from the table mmap through
     ATS, against a CPU read of the same rows: bit-exact.
  K3 one CUDA graph (ids + gather, persistent buffers) per T_pad in
     {7, 14, 21, 28}: 1,000 replays with new golden inputs copied in, each
     replay equal to the golden rows, and repeat-bitwise.
  G6 timing of the captured graph for one decode step (R = 1, T = 7, 112
     rows), real row sets from the golden decode steps. Conditions, in
     interleaved blocks (A B A B), 200 replays per block:
       warm       all pages of the step resident and mapped (A arm, twice:
                  noise band)
       cold       the cold pages of the step evicted, no prefetch
       minor      the page table entries of the cold pages removed, the
                  pages stay in the page cache: no I/O, one ATS fault per
                  page. (p50(minor) - p50(warm)) / cold pages is the cost of
                  one GPU fault on a cached page.
       staged     the cold pages evicted, a CPU thread issues
                  posix_fadvise(WILLNEED) for the pages of token j at its
                  lead time before the replay: token j (0 = bonus) gets
                  (6 - j) x LEAD_PASS_MS + LEAD_TAIL_MS (the d6 pages get
                  the tail only). The pages are cached but not mapped.
       staged_pop the same, then MADV_POPULATE_READ on the same pages in the
                  same thread, so the page table entries exist before the
                  replay (the design candidate; CPU data in
                  coldfault-20260925: 0.02 us per page at the consumer).
       staged_pop_0.3  staged_pop with every lead cut to 0.3 ms (margin)
     The prefetch thread records how late each batch starts against its
     target (lateness). The main thread sleeps (it does not spin) until the
     replay, and the switch interval is 50 us, so the GIL does not delay the
     prefetch thread.
     The cold page count per step comes from coldmiss.json (S1 own mix),
     so the evicted fraction is the measured one, not "all".
     Eviction: madvise(MADV_DONTNEED) on the pages in this process's
     mapping, then posix_fadvise(DONTNEED) on the same pages. Only the pages
     of the step are evicted, never the whole file. mincore (root in the
     container) checks that the pages left the cache.
Output: DIR/g6.json with node, condition, per-replay stall (CUDA events,
microseconds), p50/p90/p99/max and the K checks.
"""
import argparse
import ctypes
import glob
import json
import mmap
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ple_gpu_twin as tw  # noqa: E402

TABLE = ("/models/usman/vllm-ple-cache/Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/"
         "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.packed_u8")
ROW = 90
PAGE = 4096
EOS = 248044
# Lead times. One MTP draft pass: 8.06 ms / 6 = 1.34 ms (LADDER.md arm D4,
# spark2). Tail: the embedding plus layer 0 before the PLE layer, about
# 1.1 ms (kern-decode PLAN.md R3a). Draft i is known (6 - i) passes plus the
# tail before the gather; the bonus token 6 passes plus the tail.
# Update 2026-09-25: the shipping stack F0 (LADDER.md kern-r21) has a draft
# part of 6.45 ms, so one pass is 1.08 ms. The d6 lead does not change.
LEAD_PASS_MS = 1.08
LEAD_TAIL_MS = 1.1


def load_golden(d: str):
    steps = []
    for f in sorted(glob.glob(os.path.join(d, "golden-part*.npz"))):
        z = np.load(f)
        n = len(z["classes"])
        for i in range(n):
            steps.append({k: z[f"s{i}_{k}"] for k in ("ids", "qsl", "nctx", "num", "rowids")}
                         | {"cls": str(z["classes"][i])})
    return steps


def pages_of(rids: np.ndarray) -> np.ndarray:
    off = rids.reshape(-1).astype(np.int64) * ROW
    return np.unique(np.concatenate((off >> 12, (off + ROW - 1) >> 12)))


def pct(a):
    a = np.asarray(a, dtype=np.float64)
    return {"n": int(a.size), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max()), "mean": float(a.mean())}


class Table:
    """The table mmap, the eviction and the prefetch calls."""

    def __init__(self, path: str) -> None:
        self.size = os.path.getsize(path)
        self.fd = os.open(path, os.O_RDONLY)
        self.mm = mmap.mmap(self.fd, self.size, mmap.MAP_SHARED, mmap.PROT_READ)
        self.host = np.frombuffer(self.mm, dtype=np.uint8)
        self.addr = self.host.__array_interface__["data"][0]
        self.libc = ctypes.CDLL("libc.so.6", use_errno=True)
        self.libc.madvise(ctypes.c_void_p(self.addr), ctypes.c_size_t(self.size), 1)  # RANDOM
        self.libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]

    def touch(self, pages: np.ndarray) -> None:
        for p in pages.tolist():
            _ = int(self.host[min(p * PAGE, self.size - 1)])

    def evict(self, pages: np.ndarray) -> None:
        for p in pages.tolist():
            self.libc.madvise(ctypes.c_void_p(self.addr + p * PAGE), ctypes.c_size_t(PAGE), 4)
            os.posix_fadvise(self.fd, p * PAGE, PAGE, os.POSIX_FADV_DONTNEED)

    def prefetch(self, pages: np.ndarray, populate: bool = False) -> None:
        for p in pages.tolist():
            os.posix_fadvise(self.fd, p * PAGE, PAGE, os.POSIX_FADV_WILLNEED)
        if populate:  # MADV_POPULATE_READ (22): wait for the I/O, fill the PTEs
            for p in pages.tolist():
                self.libc.madvise(ctypes.c_void_p(self.addr + p * PAGE), ctypes.c_size_t(PAGE), 22)

    def zap(self, pages: np.ndarray) -> None:
        # MADV_DONTNEED on the mapping only: the PTE goes, the page stays cached.
        for p in pages.tolist():
            self.libc.madvise(ctypes.c_void_p(self.addr + p * PAGE), ctypes.c_size_t(PAGE), 4)

    def resident(self, pages: np.ndarray) -> int:
        vec = (ctypes.c_ubyte * 1)()
        n = 0
        for p in pages.tolist():
            if self.libc.mincore(ctypes.c_void_p(self.addr + p * PAGE), PAGE, vec) == 0:
                n += vec[0] & 1
        return n


def probe() -> dict:
    cu = ctypes.CDLL("libcuda.so.1")
    dev = ctypes.c_int()
    cu.cuInit(0)
    cu.cuDeviceGet(ctypes.byref(dev), 0)
    out = {}
    for name, attr in (("pageable", 88), ("host_page_tables", 100)):
        v = ctypes.c_int(0)
        rc = cu.cuDeviceGetAttribute(ctypes.byref(v), attr, dev)
        out[name] = v.value if rc == 0 else -rc
    return out


def cuda_view(addr: int, nbytes: int):
    """A uint8 "cuda" tensor over host memory (DLPack kDLCUDA; GB10 ATS)."""
    import torch

    class DLDevice(ctypes.Structure):
        _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]

    class DLDataType(ctypes.Structure):
        _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]

    class DLTensor(ctypes.Structure):
        _fields_ = [("data", ctypes.c_void_p), ("device", DLDevice), ("ndim", ctypes.c_int32),
                    ("dtype", DLDataType), ("shape", ctypes.POINTER(ctypes.c_int64)),
                    ("strides", ctypes.POINTER(ctypes.c_int64)), ("byte_offset", ctypes.c_uint64)]

    class DLManaged(ctypes.Structure):
        _fields_ = [("dl_tensor", DLTensor), ("manager_ctx", ctypes.c_void_p),
                    ("deleter", ctypes.c_void_p)]

    shape = (ctypes.c_int64 * 1)(nbytes)
    mt = DLManaged()
    mt.dl_tensor = DLTensor(addr, DLDevice(2, 0), 1, DLDataType(1, 8, 1), shape, None, 0)
    new = ctypes.pythonapi.PyCapsule_New
    new.restype = ctypes.py_object
    new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    cuda_view.keep = getattr(cuda_view, "keep", []) + [shape, mt]
    return torch.utils.dlpack.from_dlpack(new(ctypes.addressof(mt), b"dltensor", None))


def g6_loop(a, tab, dec, cold_frac, set_inputs, replay_us, res) -> None:
    """The G6 conditions. set_inputs(g) loads one step; replay_us(g) runs it
    and returns its time in microseconds."""
    rng = np.random.default_rng(1)
    leads = [(6 - j) * LEAD_PASS_MS + LEAD_TAIL_MS for j in range(7)]
    order = ["warm_a", "cold", "minor", "staged", "staged_pop", "warm_b", "staged_pop_0.3"]
    stats = {c: [] for c in order}
    late = {c: [] for c in order}
    ncold = {c: [] for c in order}
    evict_check = []
    minor_check = []
    sys.setswitchinterval(5e-5)
    for blk in range(a.blocks):
        for cond in (order if blk % 2 == 0 else order[::-1]):
            for _ in range(a.replays):
                g = dec[int(rng.integers(0, len(dec)))]
                pg = [pages_of(g["rowids"][j]) for j in range(7)]
                allp = np.unique(np.concatenate(pg))
                set_inputs(g)
                if cond.startswith("warm"):
                    tab.touch(allp)
                    th = None
                elif cond == "minor":
                    tab.touch(allp)
                    cold = rng.choice(allp, size=int(round(cold_frac * allp.size)), replace=False)
                    tab.zap(cold)
                    ncold[cond].append(int(cold.size))
                    if len(minor_check) < 50:
                        minor_check.append(tab.resident(cold) / max(cold.size, 1))
                    th = None
                else:
                    tab.touch(allp)
                    cold = rng.choice(allp, size=int(round(cold_frac * allp.size)), replace=False)
                    tab.evict(cold)
                    ncold[cond].append(int(cold.size))
                    if len(evict_check) < 50:
                        evict_check.append(tab.resident(cold) / max(cold.size, 1))
                    th = None
                    if cond != "cold":
                        cut = 0.3 if cond.endswith("_0.3") else None
                        pop = cond.startswith("staged_pop")
                        sched = [((cut if cut else leads[j]), np.intersect1d(pg[j], cold))
                                 for j in range(7)]
                        sched.sort(key=lambda x: -x[0])
                        t_rep = time.perf_counter() + sched[0][0] / 1e3

                        # Batches with the same lead form one group: the thread
                        # issues WILLNEED for the whole group first, then
                        # populates it (the design of the C++ thread).
                        groups = {}
                        for lead, pp in sched:
                            groups.setdefault(lead, []).append(pp)
                        gsched = [(ld, np.concatenate(v)) for ld, v in groups.items()]

                        def run(sched=gsched, t_rep=t_rep, pop=pop, lt=late[cond]):
                            for lead, pp in sched:
                                target = t_rep - lead / 1e3
                                while time.perf_counter() < target:
                                    pass
                                lt.append((time.perf_counter() - target) * 1e6)
                                tab.prefetch(pp, populate=pop)
                        th = threading.Thread(target=run)
                        th.start()
                        while time.perf_counter() < t_rep - 2e-4:
                            time.sleep(1e-4)
                        while time.perf_counter() < t_rep:
                            pass
                stats[cond].append(replay_us(g))
                if th is not None:
                    th.join()
    res["G6"] = {c: pct(v) for c, v in stats.items()}
    res["G6_lateness_us"] = {c: pct(v) for c, v in late.items() if v}
    res["G6_cold_pages_mean"] = {c: float(np.mean(v)) for c, v in ncold.items() if v}
    if stats["minor"] and ncold["minor"]:
        res["ats_minor_fault_us_per_page"] = (
            (np.percentile(stats["minor"], 50) - np.percentile(stats["warm_a"] + stats["warm_b"], 50))
            / float(np.mean(ncold["minor"])))
    res["minor_resident_frac_mean"] = float(np.mean(minor_check)) if minor_check else None
    res["evict_resident_frac_mean"] = float(np.mean(evict_check)) if evict_check else None
    res["leads_ms"] = leads

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--coldmiss", required=True)
    ap.add_argument("--parity", required=True, help="parity.json (the layer consts)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--table", default=TABLE)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--replays", type=int, default=200)
    ap.add_argument("--cpu-dry", action="store_true")
    ap.add_argument("--cpu-loop", action="store_true",
                    help="with --cpu-dry: run the G6 loop with a CPU gather. Use only with a "
                         "private --table (it evicts pages), never with the served table")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {"node": os.uname().nodename, "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "cpu_dry": a.cpu_dry}
    cst = json.load(open(a.parity))["consts"]
    k = tw.Consts(cst["mult"], cst["sizes"], cst["offs"], cst["eos"], 8)
    cm = json.load(open(a.coldmiss))
    cold_frac = cm["cold_pages_per_step"]["S1_own"]["mean"] / cm["pages_per_step"]["mean"]
    res["cold_frac_S1"] = cold_frac
    gold = load_golden(a.golden)
    res["golden_steps"] = len(gold)
    tab = Table(a.table)
    flat = tab.host

    def done(code: int) -> int:
        res["exit"] = code
        json.dump(res, open(os.path.join(a.out, "g6.json"), "w"), indent=1)
        print(json.dumps({kk: v for kk, v in res.items() if kk not in ("G6",)})[:2000])
        return code

    # P
    if not a.cpu_dry:
        res["probe"] = probe()
        if res["probe"] != {"pageable": 1, "host_page_tables": 1}:
            return done(10)
        import torch
        sys.path.insert(0, HERE)
        import ple_ids_triton as kt
        dev = torch.device("cuda")
        table_dev = cuda_view(tab.addr, tab.size)
        mult = torch.tensor(k.mult, device=dev)
        sizes = torch.tensor(k.sizes, device=dev)
        offs = torch.tensor(k.offs, device=dev)

        def ids_dev(ids, qsl, nctx):
            out = torch.empty((ids.shape[0], 16), dtype=torch.int64, device=dev)
            kt.ple_ids(ids, qsl, nctx, mult, sizes, offs, out, eos=EOS, hpn=8)
            return out
    # K1
    bad = 0
    for g in gold:
        if a.cpu_dry:
            got = tw.ids_twin(k, g["ids"], g["qsl"], g["nctx"])
        else:
            got = ids_dev(torch.from_numpy(g["ids"]).to(dev), torch.from_numpy(g["qsl"]).to(dev),
                          torch.from_numpy(g["nctx"]).to(dev)).cpu().numpy()
        bad += int(not np.array_equal(got, g["rowids"]))
    res["K1_ids_bad_steps"] = bad
    if bad:
        return done(11)
    # K2: valid rows of the golden steps (at most 200k rows).
    # --cpu-dry reads 64 rows only: it must not warm the served table.
    cap = 64 if a.cpu_dry else 200_000
    rows = np.concatenate([g["rowids"][: int(g["num"])].reshape(-1) for g in gold])[:cap]
    ref = tw.gather_twin(flat, rows, ROW)
    if a.cpu_dry:
        got = ref
    else:
        out = torch.empty((rows.size, ROW), dtype=torch.uint8, device=dev)
        kt.ple_gather(table_dev, torch.from_numpy(rows).to(dev), out)
        got = out.cpu().numpy()
    res["K2_rows"] = int(rows.size)
    res["K2_equal"] = bool(np.array_equal(got, ref))
    if not res["K2_equal"]:
        return done(12)
    # K3 + G6 need the GPU.
    dec = [g for g in gold if int(g["num"]) == 7 and g["qsl"].size == 2 and g["cls"] in ("real", "pad")]
    res["G6_steps_available"] = len(dec)
    if a.cpu_dry:
        # Harness logic only: the lead schedule and the cold page choice.
        rng = np.random.default_rng(0)
        g = dec[0]
        pg = [pages_of(g["rowids"][j]) for j in range(7)]
        leads = [(6 - j) * LEAD_PASS_MS + LEAD_TAIL_MS for j in range(7)]
        allp = np.unique(np.concatenate(pg))
        cold = rng.choice(allp, size=int(round(cold_frac * allp.size)), replace=False)
        res["dry"] = {"leads_ms": leads, "pages": int(allp.size), "cold_pages": int(cold.size)}
        if a.cpu_loop:
            if os.path.realpath(a.table) == os.path.realpath(TABLE):
                return done(14)

            def replay_us(g):
                t0 = time.perf_counter()
                tw.gather_twin(flat, g["rowids"][:7], ROW)
                return (time.perf_counter() - t0) * 1e6
            g6_loop(a, tab, dec, cold_frac, lambda g: None, replay_us, res)
        return done(0)
    ids_b = torch.zeros(28, dtype=torch.int32, device=dev)
    qsl_b = torch.zeros(5, dtype=torch.int32, device=dev)
    nctx_b = torch.full((4, 2), EOS, dtype=torch.int32, device=dev)
    rid_b = torch.zeros((28, 16), dtype=torch.int64, device=dev)
    rows_b = torch.zeros((28 * 16, ROW), dtype=torch.uint8, device=dev)
    graphs = {}
    for t_pad, r_pad in ((7, 1), (14, 2), (21, 3), (28, 4)):
        gph = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                kt.ple_ids(ids_b[:t_pad], qsl_b[:r_pad + 1], nctx_b[:r_pad], mult, sizes, offs,
                           rid_b[:t_pad], eos=EOS, hpn=8)
                kt.ple_gather(table_dev, rid_b[:t_pad], rows_b[: t_pad * 16])
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(gph):
            kt.ple_ids(ids_b[:t_pad], qsl_b[:r_pad + 1], nctx_b[:r_pad], mult, sizes, offs,
                       rid_b[:t_pad], eos=EOS, hpn=8)
            kt.ple_gather(table_dev, rid_b[:t_pad], rows_b[: t_pad * 16])
        graphs[(t_pad, r_pad)] = gph
    k3_bad = k3_n = 0
    for g in gold:
        key = (g["ids"].size, g["qsl"].size - 1)
        if key not in graphs or k3_n >= 1000:
            continue
        ids_b[: key[0]].copy_(torch.from_numpy(g["ids"]))
        qsl_b[: key[1] + 1].copy_(torch.from_numpy(g["qsl"]))
        nctx_b[: key[1]].copy_(torch.from_numpy(g["nctx"]))
        graphs[key].replay()
        torch.cuda.synchronize()
        r = rid_b[: key[0]].cpu().numpy()
        b = rows_b[: key[0] * 16].cpu().numpy()
        k3_bad += int(not (np.array_equal(r, g["rowids"])
                           and np.array_equal(b, tw.gather_twin(flat, g["rowids"], ROW))))
        k3_n += 1
    res["K3_replays"], res["K3_bad"] = k3_n, k3_bad
    if k3_bad or not k3_n:
        return done(13)
    # G6
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def set_inputs(g):
        ids_b[:7].copy_(torch.from_numpy(g["ids"]))
        qsl_b[:2].copy_(torch.from_numpy(g["qsl"]))
        nctx_b[:1].copy_(torch.from_numpy(g["nctx"]))
        torch.cuda.synchronize()

    def replay_us(g):
        ev0.record()
        graphs[(7, 1)].replay()
        ev1.record()
        ev1.synchronize()
        return ev0.elapsed_time(ev1) * 1e3

    g6_loop(a, tab, dec, cold_frac, set_inputs, replay_us, res)
    return done(0)


if __name__ == "__main__":
    sys.exit(main())
