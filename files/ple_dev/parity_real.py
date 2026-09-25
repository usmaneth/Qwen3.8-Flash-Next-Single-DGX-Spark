#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate G-ple (CPU part): the twin of the device PLE kernels equals the CPU worker.

    python3 parity_real.py --streams DIR/streams.npz --out DIR [--workers 4]
                           [--bytes-rows 65536] [--min-avail-gib 20]

Paths under test, on the same step inputs:
  T  ple_gpu_twin.ids_twin: the CPU twin of _ple_ids_kernel. It reads the
     GPU-side buffers (input_ids [T_pad], the padded query_start_loc
     [R_pad + 1], ngram_context [R_pad, 2]).
  W  forward_impl of the recipe's files/ple_layer_patched.py, offload branch
     (is_offload_process() = True): the CPU worker. It reads what the
     connector sends: T_pad tokens, the unpadded query_start_loc [R + 1] and
     ngram_context [R, 2].
  B  ple_io_fast.row_ids (the B3 numpy path) on the worker inputs.
  G  forward_impl, GPU branch (is_offload_process() = False), on the GPU
     buffers, for 1 step in --g-every.
The script takes the class and the hash helpers from the source with ast (no
vllm import) and builds the layer with its own __init__ from config.json.

Pass rule: for every valid token (t < num_tokens), the 16 row ids of T, W and
B (and G where it runs) are identical. Padding tokens are not compared: the
kernel gives them the all-EOS trigram, the worker gives them the ids of a
clamped real position. The script counts those rows (information only).

Step classes (labels in the output):
  real      the replay of streams.py with no change (served profile shapes).
  pad       real, plus CUDA graph padding: T_pad from the default vLLM
            capture list, R_pad up to 4, stale token values in the padding.
  eos       real streams with EOS (<|endoftext|>) put in at 2% of positions.
            The served streams have no EOS (the model stops on <|im_end|>).
  sentinel  decode steps where the last 1-5 drafts are the -1 sentinel.
  wrap      valid tokens replaced by random int32 values up to 2^31 - 1
            (not real ids): int64 wrap and the torch.remainder sign rule.
Negative controls: each mutation in ple_gpu_twin.VARIANTS runs on the same
inputs (1 step in --neg-every, plus each step of the eos, sentinel and wrap
classes). A mutation that no class detects is a hole in the gate.

Bytes (--bytes-rows > 0): the gather twin and the worker's
torch.index_select read the same rows. The table pages come from the packed
table with O_DIRECT into a private sparse copy (MAP_NORESERVE), so the study
does not add pages to the page cache of the served table. A short list of
edge rows (0, the int32 byte-offset limits, the last row) also reads through
the real mmap, as the worker does.

Memory rule: each worker stops between steps while MemAvailable is under
--min-avail-gib, and goes on when it is above again.
"""
import __future__
import argparse
import ast
import glob
import hashlib
import importlib.util
import json
import math
import mmap
import multiprocessing as mp
import os
import sys
import time
import types
import warnings

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ple_gpu_twin as tw  # noqa: E402
import streams as sm  # noqa: E402

REF = "/models/usman/frontier/next/ple/ref_ple_layer_patched.py"
TABLE = ("/models/usman/vllm-ple-cache/Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/"
         "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.packed_u8")
CLASS = "Qwen3_8FlashNextNGramEmbedding"
HELPERS = ("_splitmix64", "_is_prime_64", "_nth_prime_after", "_ple_prefetch_rows")
CAPTURE = [1, 2, 4] + list(range(8, 257, 8))  # vLLM default capture sizes up to 256
ROW = 90
PAGE = 4096
# Mutations of the id kernel. i32_offset is a gather mutation (bytes_check).
IDS_VARIANTS = [v for v in tw.VARIANTS if v not in ("exact", "i32_offset")]


def default_config() -> str:
    hits = sorted(glob.glob("/models/usman/hf/hub/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/"
                            "snapshots/*/config.json"))
    return hits[-1] if hits else ""


def mem_avail_gib() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    return 0.0


class _Emb:
    """Stand-in for VocabParallelEmbedding. The call returns the row ids."""

    def __init__(self, *args, **kwargs) -> None:
        self._packed_table = None
        self._packed_table_fd = None

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        return ids.unsqueeze(-1)


def build_layer(ref: str, cfg_path: str, layer_id: int, max_tokens: int, max_reqs: int):
    src = open(ref).read()
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and all(
                isinstance(t, ast.Name) and t.id.startswith(("_SPLITMIX", "_MASK64", "_PLE_"))
                for t in node.targets):
            keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in HELPERS:
            keep.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == CLASS:
            node.bases = [ast.Name(id="PleOffloadLayer", ctx=ast.Load())]
            keep.append(node)
    code = compile(ast.fix_missing_locations(ast.Module(body=keep, type_ignores=[])),
                   "ple_layer_patched.py", "exec",
                   flags=__future__.annotations.compiler_flag, dont_inherit=True)
    flag = {"offload": True}
    ns = {"torch": torch, "nn": torch.nn, "math": math, "os": os,
          "PleOffloadLayer": torch.nn.Module, "VocabParallelEmbedding": _Emb,
          "_get_ple_embedding_quant_method": lambda *a, **k: None,
          "is_offload_process": lambda: flag["offload"], "__name__": "ple_ref"}
    exec(code, ns)
    cfg = json.load(open(cfg_path))
    tc = types.SimpleNamespace(**cfg.get("text_config", cfg))
    layer = ns[CLASS](tc, int(tc.ple_embed_dim), layer_id, max_tokens, max_reqs,
                      "ple.ple_embedding", quant_config=None, params_dtype=torch.bfloat16)
    return layer, flag, tc, hashlib.sha256(src.encode()).hexdigest()[:16]


def load_fast(path: str):
    for name in ("vllm", "vllm.v1", "vllm.v1.ple_offload"):
        sys.modules.setdefault(name, types.ModuleType(name))
    stub = types.ModuleType("vllm.v1.ple_offload.ple_io")
    stub.FAST = 1
    sys.modules["vllm.v1.ple_offload.ple_io"] = stub
    sys.modules["vllm.v1.ple_offload"].ple_io = stub
    spec = importlib.util.spec_from_file_location("ple_io_fast_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Buffers:
    """The persistent runner buffers (stale values survive between steps)."""

    def __init__(self, max_tokens: int, max_reqs: int, eos: int) -> None:
        self.ids = np.zeros(max_tokens, np.int32)
        self.qsl = np.zeros(max_reqs + 1, np.int32)
        self.nctx = np.full((max_reqs, 2), eos, np.int32)
        self.eos = eos

    def fill(self, segs, ctxs, t_pad: int, r_pad: int):
        r = len(segs)
        lens = [s[2].size for s in segs]
        num = int(sum(lens))
        pos = 0
        for s in segs:
            self.ids[pos:pos + s[2].size] = s[2]
            pos += s[2].size
        self.qsl[0] = 0
        self.qsl[1:r + 1] = np.cumsum(lens)
        self.qsl[r + 1:] = num  # model_runner.py:1229
        self.nctx[:r_pad] = self.eos  # _prepare_ngram_context fills, then copies
        for i, c in enumerate(ctxs):
            self.nctx[i] = c
        return (self.ids[:t_pad].copy(), self.qsl[:r_pad + 1].copy(),
                self.nctx[:r_pad].copy(), num)


def pad_shape(num: int, r: int, rng, max_reqs: int) -> tuple[int, int]:
    t_pad = next((c for c in CAPTURE if c >= num), num)
    r_pad = min(max_reqs, r + int(rng.integers(0, max_reqs - r + 1)))
    return t_pad, r_pad


def run_part(args, part: int, nparts: int, q) -> None:
    torch.set_num_threads(1)
    layer, flag, tc, ref_sha = build_layer(args.ref, args.config, args.layer_id,
                                           args.max_tokens, args.max_reqs)
    eos = int(tc.eos_token_id)
    k = tw.Consts.from_layer(layer)
    fast = load_fast(args.fast)
    kb = fast._Consts(layer)
    st = sm.Streams(args.streams)
    rng = np.random.default_rng(args.seed * 1000 + part)
    rids = [r for r in range(len(st)) if r % nparts == part][: args.max_records or None]
    eos_rids = set(r for r in rids if rng.random() < args.eos_frac)

    def hook(rid, seq):
        if rid not in eos_rids:
            return seq
        s = seq.copy()
        s[rng.random(s.size) < 0.02] = eos
        return s

    rep = sm.Replay(st, rids, rng, k=6, max_seqs=args.max_reqs_live, budget=args.budget,
                    seq_hook=hook)
    buf = Buffers(args.max_tokens, args.max_reqs, eos)
    lo = k.offs.reshape(1, -1)
    hi = (k.offs + k.sizes).reshape(1, -1)
    cnt = {}
    neg = {v: {} for v in IDS_VARIANTS}
    golden = []
    golden_n = 0
    digest = hashlib.blake2b(digest_size=16)
    fails = []
    step = 0
    t_paths = {"T": 0.0, "W": 0.0, "B": 0.0, "G": 0.0}
    while not rep.done():
        if step % 500 == 0:
            while mem_avail_gib() < args.min_avail_gib:
                q.put(("pause", part, mem_avail_gib()))
                time.sleep(30)
        out = rep.step()
        if out is None:
            continue
        segs, ctxs = out
        r = len(segs)
        num = int(sum(s[2].size for s in segs))
        cls = "eos" if any(s[0] in eos_rids for s in segs) else "real"
        decode_only = all(s[1] == "decode" for s in segs)
        u = rng.random()
        if cls == "real" and decode_only and u < 0.05:
            cls = "sentinel"
            j = int(rng.integers(0, r))
            rid, kind, toks, n, tl_ = segs[j]
            toks = toks.copy()
            m = int(rng.integers(1, min(5, toks.size - 1) + 1))
            toks[-m:] = -1
            segs[j] = (rid, kind, toks, n, min(tl_, toks.size - m))
        elif cls == "real" and u < 0.07:
            cls = "wrap"
            segs = [(a, b, rng.integers(0, 2**31 - 1, size=c.size).astype(np.int32), d, e)
                    for a, b, c, d, e in segs]
        t_pad, r_pad = num, r
        if cls == "real" and u > 0.7 and num <= CAPTURE[-1]:
            cls = "pad"
            t_pad, r_pad = pad_shape(num, r, rng, args.max_reqs_live)
        ids, qsl_g, nctx_g, num = buf.fill(segs, ctxs, t_pad, r_pad)
        qsl_w = qsl_g[:r + 1].copy()
        nctx_w = nctx_g[:r].copy()

        t0 = time.perf_counter()
        idT = tw.ids_twin(k, ids, qsl_g, nctx_g)
        t1 = time.perf_counter()
        flag["offload"] = True
        idW = layer.forward_impl(None, torch.from_numpy(ids), torch.from_numpy(qsl_w),
                                 torch.from_numpy(nctx_w)).numpy()
        t2 = time.perf_counter()
        idB = fast.row_ids(kb, ids.astype(np.int64), qsl_w.astype(np.int64),
                           nctx_w.astype(np.int64))
        t3 = time.perf_counter()
        t_paths["T"] += t1 - t0
        t_paths["W"] += t2 - t1
        t_paths["B"] += t3 - t2
        c = cnt.setdefault(cls, {"steps": 0, "tokens": 0, "rows": 0, "decode_steps": 0,
                                 "rejected_tokens": 0, "pad_tokens": 0,
                                 "pad_rows_differ": 0, "g_steps": 0, "g_tokens": 0,
                                 "mismatch_TW": 0, "mismatch_TB": 0, "mismatch_TG": 0,
                                 "range_fail": 0, "max_row_id": 0, "min_mix_neg": 0})
        c["steps"] += 1
        c["tokens"] += num
        c["rows"] += num * k.heads
        c["decode_steps"] += int(decode_only)
        c["rejected_tokens"] += sum(s[2].size - s[4] for s in segs if s[1] == "decode")
        c["pad_tokens"] += t_pad - num
        v = slice(0, num)
        if not np.array_equal(idT[v], idW[v]):
            c["mismatch_TW"] += 1
        if idB is None or not np.array_equal(idT[v], idB[v]):
            c["mismatch_TB"] += 1
        if t_pad > num:
            c["pad_rows_differ"] += int((idT[num:] != idW[num:]).sum())
        if ((idT[v] < lo) | (idT[v] >= hi)).any():
            c["range_fail"] += 1
        c["max_row_id"] = max(c["max_row_id"], int(idT[v].max()) if num else 0)
        if step % args.g_every == 0:
            flag["offload"] = False
            t4 = time.perf_counter()
            idG = layer.forward_impl(None, torch.from_numpy(ids), torch.from_numpy(qsl_g),
                                     torch.from_numpy(nctx_g)).numpy()
            t_paths["G"] += time.perf_counter() - t4
            c["g_steps"] += 1
            c["g_tokens"] += num
            if not np.array_equal(idT[v], idG[v]):
                c["mismatch_TG"] += 1
        bad = c["mismatch_TW"] + c["mismatch_TB"] + c["mismatch_TG"] + c["range_fail"]
        if bad and len(fails) < 5:
            fails.append({"class": cls, "step": step, "qsl_g": qsl_g.tolist(),
                          "nctx_g": nctx_g.tolist(), "ids": ids.tolist()})
        if cls in ("real", "pad"):
            digest.update(idT[v].tobytes())
        if step % args.neg_every == 0 or cls in ("eos", "sentinel", "wrap"):
            for var in neg:
                idV = tw.ids_twin(k, ids, qsl_g, nctx_g, variant=var)
                d = neg[var].setdefault(cls, [0, 0])
                d[0] += 1
                d[1] += int(not np.array_equal(idV[v], idW[v]))
        if golden_n < args.golden_steps and (step % 7 == 0 or cls != "real"):
            golden.append((ids, qsl_g, nctx_g, np.int32(num), idT, cls))
            golden_n += 1
        if (args.bytes_rows and step % args.bytes_every == 0 and cls in ("real", "pad")
                and decode_only):
            q.put(("bytes", part, ids[:num].copy(), qsl_w.copy(), nctx_w.copy(),
                   idT[v].copy()))
        step += 1
        if step % 20000 == 0:
            q.put(("progress", part, step, sum(x["tokens"] for x in cnt.values())))
    gpath = os.path.join(args.out, f"golden-part{part}.npz")
    if golden:
        np.savez_compressed(
            gpath,
            **{f"s{i}_{name}": arr for i, g in enumerate(golden)
               for name, arr in zip(("ids", "qsl", "nctx", "num", "rowids"), g[:5])},
            classes=np.asarray([g[5] for g in golden]))
    q.put(("done", part, {"counts": cnt, "neg": neg, "fails": fails, "steps": step,
                          "records": len(rids), "eos_records": len(eos_rids),
                          "digest_real_pad": digest.hexdigest(), "ref_sha": ref_sha,
                          "time_s": t_paths}))


class Direct:
    """Reads table pages with O_DIRECT into a private sparse copy."""

    def __init__(self, path: str, rows: int) -> None:
        self.size = os.path.getsize(path)
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        self.mm = mmap.mmap(-1, self.size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
                            | getattr(mmap, "MAP_NORESERVE", 0x4000))
        self.mv = memoryview(self.mm)
        self.np = np.frombuffer(self.mm, dtype=np.uint8)
        self.table = torch.from_numpy(self.np[: rows * ROW].reshape(rows, ROW))
        self.pages = 0

    def load(self, row_ids: np.ndarray) -> np.ndarray:
        off = row_ids.astype(np.int64).reshape(-1) * ROW
        pages = np.unique(np.concatenate((off >> 12, (off + ROW - 1) >> 12)))
        for p in pages.tolist():
            o = p << 12
            n = min(PAGE, self.size - o)
            got = os.preadv(self.fd, [self.mv[o:o + PAGE]], o)
            if got != n:
                raise OSError(f"short O_DIRECT read at page {p}: {got} != {n}")
        self.pages += pages.size
        return pages

    def drop(self, pages: np.ndarray) -> None:
        for p in pages.tolist():
            self.mm.madvise(mmap.MADV_DONTNEED, p << 12, PAGE)


def bytes_check(args, items, layer, flag) -> dict:
    """Row bytes: the gather twin against the worker's index_select."""
    meta = json.load(open(args.table + ".json"))
    rows = int(meta["total_rows"])
    d = Direct(args.table, rows)
    emb = layer.ngram_embedding
    emb._packed_table = d.table
    emb._packed_table_fd = None
    flat = d.np
    res = {"steps": 0, "tokens": 0, "rows": 0, "pages_read": 0, "mismatch": 0,
           "total_rows": rows}
    heads = layer.ngram_heads
    for ids, qsl_w, nctx_w, idT in items:
        if res["rows"] >= args.bytes_rows:
            break
        pages = d.load(idT)
        num = ids.size
        outbuf = torch.zeros((num, heads * ROW), dtype=torch.uint8)
        flag["offload"] = True
        ref = layer.forward_impl(None, torch.from_numpy(ids), torch.from_numpy(qsl_w),
                                 torch.from_numpy(nctx_w), output_buffer=outbuf)
        twin = tw.gather_twin(flat, idT, ROW).reshape(num, heads * ROW)
        if not np.array_equal(ref.numpy()[:num], twin):
            res["mismatch"] += 1
        res["steps"] += 1
        res["tokens"] += num
        res["rows"] += idT.size
        d.drop(pages)
    res["pages_read"] = d.pages
    # Edge rows through the real mmap (a few pages), as the worker maps it.
    edges = np.asarray([0, 1, (2**31) // ROW - 1, (2**31) // ROW, (2**31) // ROW + 1,
                        (2**32) // ROW, (2**32) // ROW + 1, rows - 2, rows - 1], np.int64)
    warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
    mm = np.memmap(args.table, dtype=np.uint8, mode="r", shape=(rows, ROW))
    try:
        mm._mmap.madvise(mmap.MADV_RANDOM)
    except Exception:  # noqa: BLE001
        pass
    real = torch.index_select(torch.from_numpy(mm), 0, torch.from_numpy(edges)).numpy()
    pages = d.load(edges)
    direct = torch.index_select(d.table, 0, torch.from_numpy(edges)).numpy()
    twin = tw.gather_twin(flat, edges, ROW)
    twin_real = tw.gather_twin(np.memmap(args.table, dtype=np.uint8, mode="r"), edges, ROW)
    d.drop(pages)
    res["edge_rows"] = edges.tolist()
    res["edge_offsets_max"] = int(edges.max() * ROW + ROW)
    res["edge_ok"] = bool(np.array_equal(real, direct) and np.array_equal(real, twin)
                          and np.array_equal(real, twin_real))
    res["file_size"] = d.size
    # The mutations of the gather must fail on the same rows.
    res["neg_i32_offset_detected"] = bool(not np.array_equal(
        tw.gather_twin(flat, edges, ROW, variant="i32_offset"), real))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--fast", default=os.path.join(FILES, "ple_io", "ple_io_fast.py"))
    ap.add_argument("--config", default=default_config())
    ap.add_argument("--table", default=TABLE)
    ap.add_argument("--layer-id", type=int, default=0)  # ple_dense_layer_id (model.py:195-203)
    ap.add_argument("--max-tokens", type=int, default=8192 + 256)
    ap.add_argument("--max-reqs", type=int, default=8)
    ap.add_argument("--max-reqs-live", type=int, default=4)
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--seed", type=int, default=25)
    ap.add_argument("--eos-frac", type=float, default=0.1)
    ap.add_argument("--g-every", type=int, default=50)
    ap.add_argument("--neg-every", type=int, default=20)
    ap.add_argument("--golden-steps", type=int, default=1500)
    ap.add_argument("--bytes-rows", type=int, default=65536)
    ap.add_argument("--bytes-every", type=int, default=97)
    ap.add_argument("--min-avail-gib", type=float, default=20.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    # The layer constants must describe the served table (dense id 0 gives
    # 320,001,446 rows, padded to 320,001,536; dense id 1 does not fit).
    lay0, _, tc0, _ = build_layer(args.ref, args.config, args.layer_id, 64, 4)
    k0 = tw.Consts.from_layer(lay0)
    div0 = int(tc0.make_ngram_vocab_size_divisible_by)
    rows0 = ((int(k0.offs[-1] + k0.sizes[-1]) + div0 - 1) // div0) * div0
    meta0 = json.load(open(args.table + ".json"))
    if rows0 != int(meta0["total_rows"]):
        print(f"ERROR: layer id {args.layer_id} gives {rows0} rows, the table has "
              f"{meta0['total_rows']}")
        return 2
    t0 = time.time()
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    procs = [ctx.Process(target=run_part, args=(args, i, args.workers, q))
             for i in range(args.workers)]
    for p in procs:
        p.start()
    done, items = {}, []
    while len(done) < args.workers:
        m = q.get()
        if m[0] == "done":
            done[m[1]] = m[2]
        elif m[0] == "bytes":
            if sum(x[3].size for x in items) < args.bytes_rows:
                items.append(m[2:])
        elif m[0] == "pause":
            print(f"part {m[1]}: paused, MemAvailable {m[2]:.1f} GiB", flush=True)
        else:
            print(f"part {m[1]}: step {m[2]}, tokens {m[3]}", flush=True)
    for p in procs:
        p.join()
    layer, flag, tc, ref_sha = build_layer(args.ref, args.config, args.layer_id,
                                           args.max_tokens, args.max_reqs)
    k = tw.Consts.from_layer(layer)
    div = int(tc.make_ngram_vocab_size_divisible_by)
    padded = ((int(k.offs[-1] + k.sizes[-1]) + div - 1) // div) * div
    byt = bytes_check(args, items, layer, flag) if args.bytes_rows else {}
    agg, neg = {}, {}
    for part in done.values():
        for cls, c in part["counts"].items():
            a = agg.setdefault(cls, {})
            for key, val in c.items():
                a[key] = max(a.get(key, 0), val) if key == "max_row_id" else a.get(key, 0) + val
        for var, per in part["neg"].items():
            for cls, (n, det) in per.items():
                d = neg.setdefault(var, {}).setdefault(cls, [0, 0])
                d[0] += n
                d[1] += det
    summary = {
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "node": os.uname().nodename, "wall_s": round(time.time() - t0, 1),
        "ref": args.ref, "ref_sha256_16": ref_sha, "streams": args.streams,
        "config": args.config, "layer_id": args.layer_id,
        "consts": {"mult": k.mult.tolist(), "sizes": k.sizes.tolist(),
                   "offs": k.offs.tolist(), "eos": k.eos, "padded_rows": padded,
                   "table_rows": byt.get("total_rows")},
        "classes": agg, "negative_controls": neg, "bytes": byt,
        "parts": {p: {kk: vv for kk, vv in d.items() if kk not in ("counts", "neg")}
                  for p, d in done.items()},
    }
    ok = all(c["mismatch_TW"] == 0 and c["mismatch_TB"] == 0 and c["mismatch_TG"] == 0
             and c["range_fail"] == 0 for c in agg.values())
    holes = [v for v in IDS_VARIANTS if sum(d[1] for d in neg.get(v, {}).values()) == 0]
    if byt and not byt["neg_i32_offset_detected"]:
        holes.append("i32_offset")
    ok_bytes = (not byt) or (byt["mismatch"] == 0 and byt["edge_ok"])
    summary["pass_ids"] = ok
    summary["pass_bytes"] = ok_bytes
    summary["negative_control_holes"] = holes
    summary["padded_rows_equal_table"] = (byt.get("total_rows") in (None, padded))
    path = os.path.join(args.out, "parity.json")
    json.dump(summary, open(path, "w"), indent=1)
    real_tokens = sum(agg.get(c, {}).get("tokens", 0) for c in ("real", "pad"))
    print(json.dumps({"pass_ids": ok, "pass_bytes": ok_bytes, "holes": holes,
                      "real_pad_tokens": real_tokens, "summary": path}))
    return 0 if ok and ok_bytes and not holes else 1


if __name__ == "__main__":
    sys.exit(main())
