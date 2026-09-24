#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate G2: the numpy fast path gives the same row ids and bytes as forward_impl.

    python3 files/ple_io/test_fast.py [--tokens 1000000] [--config PATH]

The reference is the torch forward_impl of the generated
files/ple_layer_patched.py with fast=0. The candidate is the same
forward_impl with fast=1, which takes the early return into
ple_io_fast.small_forward(). The test takes the class
Qwen3_8FlashNextNGramEmbedding and the hash helpers from the generated file
with ast (no vllm import), and builds the layer with its own __init__ from
the checkpoint config.json. The PLE hook of patch_ple_io.py must be in the
file (./start.sh --no-launch makes it). The test adds the fast hook of
patch_fast.py to an in-memory copy when the file does not have it.

Both paths call a stand-in ple_io.gather() that records the ids and writes
a pattern made from the ids. Random batches: 1-8 requests, up to 64 tokens,
EOS at random positions (also in ngram_context), token ids up to the vocab
size, empty requests, CUDA-graph padding (num_tokens > query_start_loc[-1])
with stale token values, and query_start_loc[-1] > num_tokens. The ids must
be identical, and the whole output buffers (with the padding rows) must be
byte identical. The test stops at the first mismatch.
"""
import __future__
import argparse
import ast
import glob
import importlib.util
import json
import math
import os
import sys
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.dirname(HERE)
CLASS = "Qwen3_8FlashNextNGramEmbedding"
HELPERS = ("_splitmix64", "_is_prime_64", "_nth_prime_after")


def default_config() -> str:
    pats = ["/models/usman/hf/hub/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/snapshots/*/config.json",
            "/root/.cache/huggingface/hub/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/snapshots/*/config.json"]
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[-1]
    return ""


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _VocabStub:
    """Stand-in for VocabParallelEmbedding: only the packed-table attributes."""

    def __init__(self, *args, **kwargs):
        self._packed_table = torch.zeros((1, 90), dtype=torch.uint8)
        self._packed_table_fd = None


def build_class(src: str):
    """Exec the helpers and the class from the generated source in a namespace."""
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
    mod = ast.fix_missing_locations(ast.Module(body=keep, type_ignores=[]))
    code = compile(mod, "ple_layer_patched.py", "exec",
                   flags=__future__.annotations.compiler_flag, dont_inherit=True)
    ns = {"torch": torch, "nn": torch.nn, "math": math, "os": os,
          "PleOffloadLayer": torch.nn.Module, "VocabParallelEmbedding": _VocabStub,
          "_get_ple_embedding_quant_method": lambda *a, **k: None,
          "is_offload_process": lambda: True, "__name__": "ple_layer_patched_g2"}
    exec(code, ns)
    return ns[CLASS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--config", default="")
    ap.add_argument("--layer-id", type=int, default=1)
    args = ap.parse_args()

    for name in ("vllm", "vllm.v1", "vllm.v1.ple_offload"):
        sys.modules.setdefault(name, types.ModuleType(name))
    os.environ["VLLM_PLE_IO_TRACE_DIR"] = ""
    ple_io = load_module("vllm.v1.ple_offload.ple_io", os.path.join(HERE, "ple_io.py"))
    fast = load_module("vllm.v1.ple_offload.ple_io_fast", os.path.join(HERE, "ple_io_fast.py"))
    sys.modules["vllm.v1.ple_offload"].ple_io = ple_io
    sys.modules["vllm.v1.ple_offload"].ple_io_fast = fast

    path = os.path.join(FILES, "ple_layer_patched.py")
    src = open(path).read()
    if "vllm.v1.ple_offload import ple_io as _ple_io" not in src:
        print(f"ERROR: {path} has no ple_io hook: run ./start.sh --no-launch first")
        return 2
    sys.path.insert(0, HERE)
    import patch_fast
    if patch_fast.MARK not in src:
        for old, new in patch_fast.EDITS:
            assert src.count(old) == 1, "fast-path anchor"
            src = src.replace(old, new)
        hooked = "added in memory"
    else:
        hooked = "in the file"
    cls = build_class(src)

    cfg_path = args.config or default_config()
    cfg = json.load(open(cfg_path))
    tc = types.SimpleNamespace(**cfg.get("text_config", cfg))
    max_tokens, max_reqs = 8192, 8
    layer = cls(tc, int(tc.ple_embed_dim), args.layer_id, max_tokens, max_reqs,
                "g2.ple_embedding", quant_config=None, params_dtype=torch.bfloat16)
    heads = layer.ngram_heads
    row_width = layer.ngram_embedding._packed_table.shape[-1]
    width = heads * row_width
    eos = int(tc.eos_token_id)
    vocab = int(tc.vocab_size)

    rec: list[torch.Tensor] = []

    def gather(fd, table, ids, out):
        rec.append(ids.clone())
        pat = (ids.to(torch.int64)[:, None] * 2654435761 + torch.arange(out.shape[1])) & 0xFF
        out.copy_(pat.to(torch.uint8))

    ple_io.gather = gather
    rng = np.random.default_rng(args.seed)
    done_tokens = batches = fast_hits = 0
    t_ref = t_fast = 0.0
    garbage = torch.from_numpy(rng.integers(0, 256, size=(64, width), dtype=np.uint8))
    while done_tokens < args.tokens:
        r = int(rng.integers(1, max_reqs + 1))
        kind = rng.random()
        if kind < 0.5:  # decode verify: 1 + K tokens per request
            k = int(rng.choice([1, 2, 4, 7]))
            lens = np.full(r, k)
        else:
            lens = rng.integers(0 if kind > 0.9 else 1, 17, size=r)
        while lens.sum() > 64:
            lens[int(rng.integers(0, r))] //= 2
        q = np.concatenate(([0], np.cumsum(lens))).astype(np.int32)
        valid = int(q[-1])
        t = valid
        u = rng.random()
        if u < 0.3:
            t = min(64, valid + int(rng.integers(1, 8)))  # graph padding
        elif u < 0.35 and valid > 1:
            t = valid - int(rng.integers(1, valid))  # query_start_loc[-1] > num_tokens
        t = max(t, 1)
        toks = rng.integers(0, vocab, size=t)
        toks[rng.random(t) < 0.1] = eos
        if u < 0.3 and rng.random() < 0.5:
            toks[valid:] = rng.integers(0, 1 << 20, size=t - valid)  # stale padding values
        ctx = rng.integers(0, vocab, size=(max_reqs, tc.ngram_size - 1))
        ctx[rng.random(ctx.shape) < 0.15] = eos
        input_ids = torch.from_numpy(toks.astype(np.int32))
        qsl = torch.from_numpy(q)
        ngc = torch.from_numpy(ctx.astype(np.int32))[:r]

        outs, ids_seen = [], []
        for f in (0, 1):
            ple_io.FAST = f
            buf = garbage.clone()
            rec.clear()
            t0 = time.perf_counter()
            res = layer.forward_impl(input_ids, input_ids, qsl, ngc, output_buffer=buf)
            dt = time.perf_counter() - t0
            if f:
                t_fast += dt
            else:
                t_ref += dt
            if len(rec) != 1:
                print(f"FAIL batch {batches}: fast={f} gather calls {len(rec)}")
                return 1
            outs.append((res.clone(), buf))
            ids_seen.append(rec[0])
        fast_hits += 1
        if not torch.equal(ids_seen[0], ids_seen[1]):
            bad = (ids_seen[0] != ids_seen[1]).nonzero()[:5].flatten().tolist()
            print(f"FAIL ids batch {batches}: q={q.tolist()} t={t} first bad {bad}")
            print("tokens", toks.tolist(), "ctx", ctx[:r].tolist())
            return 1
        if not torch.equal(outs[0][1], outs[1][1]) or not torch.equal(outs[0][0], outs[1][0]):
            print(f"FAIL bytes batch {batches}: q={q.tolist()} t={t}")
            return 1
        done_tokens += t
        batches += 1

    # Above FAST_MAX tokens the torch path runs (one check).
    ple_io.FAST = 1
    big = torch.from_numpy(rng.integers(0, vocab, size=100).astype(np.int32))
    rec.clear()
    before = fast.small_forward
    calls = []
    fast.small_forward = lambda *a, **k: calls.append(1) or before(*a, **k)
    layer.forward_impl(big, big, torch.tensor([0, 100], dtype=torch.int32),
                       torch.from_numpy(ctx[:1].astype(np.int32)),
                       output_buffer=torch.empty((128, width), dtype=torch.uint8))
    fast.small_forward = before
    if calls:
        print("FAIL the fast path ran for 100 tokens")
        return 1
    print(f"G2 fast-path ids: PASS {batches} batches, {done_tokens} tokens, "
          f"ids and bytes identical (hook {hooked}, config {cfg_path}); "
          f"mean forward_impl torch {1e3 * t_ref / batches:.3f} ms, fast {1e3 * t_fast / batches:.3f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
