#!/usr/bin/env python3
"""L7 micro gate (no server): one process per section, in the pinned image, on one node.

  docker run --rm --gpus all --entrypoint python3 -v <kern-l7 files/kern>:/k:ro \
      -v <kern-decode files/kern>:/k6:ro -v /models/usman/kern-decode/tools:/t:ro \
      -v <fx-hc12 tools/hc12>:/h:ro -v <hub>:/hub:ro -e HC12_HUB=/hub -v <out>:/o \
      vllm/vllm-openai:qwen38-flash-next /o/l7_micro.py --only s1 --out /o/s1.json

Sections (rungs of plans/l7-skinny.json). Every bound and gate below was
written before any GPU data (2026-09-25, planner, CPU only):

  s1   S1/S1q skinny MXFP8 GEMV (skinny_mx.py) against FlashInfer mm_mxfp8
       tactic 1 (the served tactic) on the 6 target MXFP8 shapes, M = 1 and 7,
       cold weights. Exactness: (a) repeat bitwise; (b) every element within
       2^-8 |ref| + K 2^-24 sum|x w| of the FP64 reference of the dequantized
       operands; (c) the in-kernel quantizer equals the FlashInfer quantizer
       bitwise (codes and values) on the test activations, and the fused GEMV
       equals the unfused GEMV bitwise at the same tile. The share of outputs
       bitwise equal to FlashInfer is reported (class B, not a gate).
       Speed gate (PLAN.md 5.7): the verify-step sum at M = 7 (served path:
       quantize + GEMM) falls by >= 0.30 ms (60% of the -0.5 ms estimate).
       S1q passes alone if it falls >= 0.18 ms more (60% of -0.3 ms).
  s3   S3 shared-expert gate row dot (sgate.py) against cuBLAS F.linear
       (1 x 2560), M = 1 and 7. Exactness: within 1 BF16 ulp + the FP32 sum
       term, repeat bitwise. Speed gate: 48 calls save >= 0.30 ms of kernel
       time (60% of the -0.5 ms kernel estimate).
  r6t  R6t tile sweep of the R6 skinny BF16 kernel (router, GDN ba, HC down,
       HC up) against the R6 tiles of the kern-decode head (/k6) and cuBLAS.
       Speed gate: the verify-step sum at M = 7 falls >= 0.12 ms against R6.
  s2   S2 tile sweep of the R4 W8A16 kernel on the MTP dense shapes: 5 passes
       at M = 1 + pass 0 at M = 7. Gate: the per-step sum falls >= 0.06 ms.
  hc12 Item 3: the lossless 12-bit HC GEMV (fx-hc12 tools/hc12) on the REAL HC
       weights against the item-1 baseline = the R6 kernel at its best tile of
       the r6t sweep on the same weights (not cuBLAS). G1: the M = 7
       verify-step sum of hc12 <= 0.80 x the baseline, and the difference is
       larger than the sum of the two A-vs-A bands. G2: hc12 == R6 kernel
       bitwise at the same tile. G3: repeat bitwise.
  h1   H1 weight memory kinds: the same kernels on weights in the default
       allocator (cudaMalloc) and in the kern_hostalloc pool (mode from
       KERN_HOSTALLOC_MODE), interleaved A B B A rounds, cold weights. Also a
       32 GiB pool allocation check (time, MemAvailable). Gate: the pool arm is
       >= 2% faster on the median of the GEMV kernels, and outside the A-vs-A
       band; else H1 stops before any server lease.
"""
import argparse
import json
import os
import socket
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/k")
sys.path.insert(0, "/t")
import mtp_w8a16 as m  # noqa: E402
from w8a16_bench import L2_BYTES, graph_time  # noqa: E402

DEV = torch.device("cuda")


def ulp_bf16(t):
    b = t.double().abs().clamp_min(1e-30)
    return torch.pow(2.0, torch.floor(torch.log2(b)) - 7)


def copies_for(nbytes, cap=64):
    return min(cap, max(2, int(L2_BYTES * 2 // max(nbytes, 1)) + 1))


def rot(fn_of_i, copies):
    it = iter(list(range(copies)) * 1000)
    return graph_time(lambda: fn_of_i(next(it) % copies))


def ext_tiles(n, k, bns=(16, 32, 64, 128), bks=(64, 128, 256), splits=(1, 2, 4, 8, 16), max_ctas=2048):
    out = []
    for bn in bns:
        for bk in bks:
            if bk > max(64, k):
                continue
            ctas = -(-n // bn)
            for split in splits:
                if split > 1 and k // split < bk:
                    continue
                if ctas * split > max_ctas:
                    continue
                for warps in (4, 8):
                    out.append((bn, bk, split, warps, 3))
    return out


def ab_rounds(fns, copies, rounds=6):
    """Interleaved rounds A B .. then .. B A; median and A-vs-A band per arm."""
    names = list(fns)
    res = {a: [] for a in names}
    for r in range(rounds):
        for a in (names if r % 2 == 0 else names[::-1]):
            res[a].append(rot(fns[a], copies))
    out = {}
    for a, v in res.items():
        med = float(np.median(v))
        out[a] = {"us": round(med, 3), "band": round((max(v) - min(v)) / med, 4), "rounds": [round(t, 3) for t in v]}
    return out


# ------------------------------------------------------------------ s1
def sec_s1(a):
    import skinny_mx as mx
    from flashinfer import mxfp8_quantize
    from flashinfer.gemm import gemm_base as gb

    major, _ = gb.get_compute_capability(DEV)
    runner = gb.get_cutlass_mxfp8_gemm_module(major).cutlass_mxfp8_gemm_runner()
    ws_buf = gb._get_cache_buf("mm_mxfp8_workspace", gb.DEFAULT_WORKSPACE_SIZE, DEV)
    out = {"rcp448_gpu": mx.rcp448(DEV), "rcp448_exact": mx.RCP448_EXACT, "shapes": {}}
    tot = {"fi": 0.0, "skinny": 0.0, "fused": 0.0}
    for (n, k), calls in mx.SHAPES.items():
        wb = n * k + n * k // 32
        copies = copies_for(wb, 48)
        wq = []
        for _ in range(copies):
            w = (torch.randn(n, k, device=DEV) / k ** 0.5).to(torch.bfloat16)
            w_mx, w_sf = mxfp8_quantize(w, is_sf_swizzled_layout=True)
            wq.append((w_mx.contiguous(), w_sf.reshape(-1).contiguous()))
        sres = {"calls": calls, "copies": copies, "m": {}}
        for mrows in (1, 7):
            g = torch.Generator(device=DEV).manual_seed(n + k + mrows)
            x = (torch.randn(mrows, k, device=DEV, generator=g)
                 * torch.exp(torch.randn(mrows, k // 32, device=DEV, generator=g) * 2).repeat_interleave(32, 1)
                 ).to(torch.bfloat16)
            a_mx, a_sf = mxfp8_quantize(x, is_sf_swizzled_layout=True)
            a_sf = a_sf.reshape(-1)
            outb = torch.empty(mrows, n, device=DEV, dtype=torch.bfloat16)

            def fi_gemm(i, a_mx=a_mx, a_sf=a_sf, outb=outb):
                runner.forward([a_mx, wq[i][0].t(), a_sf, wq[i][1], None, outb, ws_buf], tactic=1)
                return outb

            def fi_served(i, x=x, outb=outb):
                am, asf = mxfp8_quantize(x, is_sf_swizzled_layout=True)
                runner.forward([am, wq[i][0].t(), asf.reshape(-1), wq[i][1], None, outb, ws_buf], tactic=1)
                return outb

            sweep = {"unfused": [], "fused": []}
            for t in ext_tiles(n, k, bks=(128, 256), splits=(1, 2, 4, 8)):
                try:
                    sweep["unfused"].append((rot(lambda i, t=t: mx.skinny_mx_linear(
                        x, wq[i][0], wq[i][1], x_q=a_mx, x_sf=a_sf, tile=t, fused=False), copies), t))
                    sweep["fused"].append((rot(lambda i, t=t: mx.skinny_mx_linear(
                        x, wq[i][0], wq[i][1], tile=t, fused=True), copies), t))
                except Exception as e:  # noqa: BLE001
                    print("tile error", n, k, t, repr(e)[:160], flush=True)
            bu, bf = min(sweep["unfused"])[1], min(sweep["fused"])[1]

            def sk_served(i, t=bu, x=x):
                am, asf = mxfp8_quantize(x, is_sf_swizzled_layout=True)
                return mx.skinny_mx_linear(x, wq[i][0], wq[i][1], x_q=am, x_sf=asf.reshape(-1), tile=t, fused=False)

            arms = ab_rounds({"fi_served": fi_served, "fi_gemm": fi_gemm, "skinny_served": sk_served,
                              "skinny_gemv": lambda i, t=bu: mx.skinny_mx_linear(
                                  x, wq[i][0], wq[i][1], x_q=a_mx, x_sf=a_sf, tile=t, fused=False),
                              "fused": lambda i, t=bf: mx.skinny_mx_linear(x, wq[i][0], wq[i][1], tile=t,
                                                                           fused=True)}, copies)
            # exactness on copy 0
            y = mx.skinny_mx_linear(x, wq[0][0], wq[0][1], x_q=a_mx, x_sf=a_sf, tile=bu, fused=False)
            y2 = mx.skinny_mx_linear(x, wq[0][0], wq[0][1], x_q=a_mx, x_sf=a_sf, tile=bu, fused=False)
            yfu = mx.skinny_mx_linear(x, wq[0][0], wq[0][1], tile=bu, fused=True)
            yfi = fi_gemm(0).clone()
            # FP64 reference of the dequantized operands
            xs2 = a_sf.cpu().numpy()
            ws2 = wq[0][1].cpu().numpy()
            kb = k // 32
            rr, cc = np.meshgrid(np.arange(mrows), np.arange(kb), indexing="ij")
            xs_rm = xs2[mx.swizzled_sf_index(rr, cc, kb)]
            rr, cc = np.meshgrid(np.arange(n), np.arange(kb), indexing="ij")
            ws_rm = ws2[mx.swizzled_sf_index(rr, cc, kb)]
            xd = a_mx.float().double().cpu() * torch.from_numpy(
                mx.e8m0_to_f32_np(xs_rm).astype(np.float64)).repeat_interleave(32, 1)
            wd = wq[0][0].float().double().cpu() * torch.from_numpy(
                mx.e8m0_to_f32_np(ws_rm).astype(np.float64)).repeat_interleave(32, 1)
            ref = xd @ wd.t()
            absdot = xd.abs() @ wd.abs().t()
            d = (y.double().cpu() - ref).abs()
            within = bool((d <= ref.abs() * 2.0 ** -8 + absdot * k * 2.0 ** -24 + 1e-30).all())
            q, s = mx.mx_quant_rows(x)
            fi_q = a_mx.float()
            fi_s = torch.from_numpy(xs_rm.astype(np.uint8)).to(DEV)
            quant_eq = bool(torch.equal(s, fi_s)) and bool(((q == fi_q) | (q.isnan() & fi_q.isnan())).all())
            chk = {"repeat": bool(torch.equal(y.view(torch.int16), y2.view(torch.int16))), "within": within,
                   "bitwise_vs_fi": round(float((y.view(torch.int16) == yfi.view(torch.int16)).float().mean()), 5),
                   "fi_within": bool(((yfi.double().cpu() - ref).abs()
                                      <= ref.abs() * 2.0 ** -8 + absdot * k * 2.0 ** -24 + 1e-30).all()),
                   "quant_eq_fi": quant_eq,
                   "fused_eq_unfused": bool(torch.equal(yfu.view(torch.int16), y.view(torch.int16)))}
            sres["m"][mrows] = {"best_unfused": list(bu), "best_fused": list(bf), "arms": arms, "check": chk,
                                "sweep_top": {kk: [(round(t, 2), list(tt)) for t, tt in sorted(v)[:5]]
                                              for kk, v in sweep.items()}}
            print("s1", n, k, mrows, json.dumps({kk: v["us"] for kk, v in arms.items()}), chk, flush=True)
            if mrows == 7:
                tot["fi"] += calls * arms["fi_served"]["us"]
                tot["skinny"] += calls * arms["skinny_served"]["us"]
                tot["fused"] += calls * arms["fused"]["us"]
        out["shapes"][f"{n}x{k}"] = sres
    ex = all(v["m"][mm]["check"]["repeat"] and v["m"][mm]["check"]["within"]
             for v in out["shapes"].values() for mm in (1, 7))
    qe = all(v["m"][mm]["check"]["quant_eq_fi"] and v["m"][mm]["check"]["fused_eq_unfused"]
             for v in out["shapes"].values() for mm in (1, 7))
    out["verify_step_us_m7"] = {kk: round(v, 1) for kk, v in tot.items()}
    out["gate"] = {"S1_saving_ms": round((tot["fi"] - tot["skinny"]) / 1e3, 3),
                   "S1q_extra_ms": round((tot["skinny"] - tot["fused"]) / 1e3, 3),
                   "S1_exact": ex, "S1q_exact": qe,
                   "S1_pass": ex and (tot["fi"] - tot["skinny"]) >= 300.0,
                   "S1q_pass": ex and qe and (tot["skinny"] - tot["fused"]) >= 180.0}
    print("s1 gate", out["gate"], flush=True)
    return out


# ------------------------------------------------------------------ s3
def sec_s3(a):
    import sgate

    out = {"m": {}}
    n, k = 1, 2560
    copies = 64
    ws = [(torch.randn(n, k, device=DEV) / 50).to(torch.bfloat16) for _ in range(copies)]
    step = {}
    for mrows in (1, 7):
        x = torch.randn(mrows, k, device=DEV).to(torch.bfloat16)
        arms = ab_rounds({"cublas": lambda i: F.linear(x, ws[i]),
                          "rowdot": lambda i: sgate.rowdot(x, ws[i]),
                          "cublas_sig": lambda i: torch.sigmoid(F.linear(x, ws[i])),
                          "rowdot_sig": lambda i: torch.sigmoid(sgate.rowdot(x, ws[i]))}, copies)
        y, y2, yc = sgate.rowdot(x, ws[0]), sgate.rowdot(x, ws[0]), F.linear(x, ws[0])
        ref = x.double() @ ws[0].double().t()
        absdot = x.double().abs() @ ws[0].double().abs().t()
        chk = {"repeat": bool(torch.equal(y.view(torch.int16), y2.view(torch.int16))),
               "within": bool(((y.double() - ref).abs() <= ulp_bf16(ref) + absdot * k * 2.0 ** -24).all()),
               "bitwise_vs_cublas": round(float((y.view(torch.int16) == yc.view(torch.int16)).float().mean()), 4)}
        out["m"][mrows] = {"arms": arms, "check": chk}
        step[mrows] = 48 * (arms["cublas_sig"]["us"] - arms["rowdot_sig"]["us"])
        print("s3", mrows, {kk: v["us"] for kk, v in arms.items()}, chk, flush=True)
    ex = all(out["m"][mm]["check"]["repeat"] and out["m"][mm]["check"]["within"] for mm in (1, 7))
    out["gate"] = {"saving_ms_m7": round(step[7] / 1e3, 3), "exact": ex, "pass": ex and step[7] >= 300.0}
    print("s3 gate", out["gate"], flush=True)
    return out


# ------------------------------------------------------------------ r6t
R6_SHAPES = [("router", 512, 2560, 48), ("gdn_ba", 96, 2560, 36), ("hc_down_inject", 336, 10240, 96),
             ("hc_mixer_down", 320, 10240, 1), ("hc_up", 10240, 320, 97)]


def r6_tile(n, k):
    try:
        sys.path.insert(0, "/k6")
        import skinny_bf16 as s6  # the R6 file of the kern-decode head
        t = s6.TILES.get((n, k))
        if t:
            return tuple(t), "kern-decode skinny_bf16.TILES"
    except ImportError:
        pass
    return m.default_tile(n, k), "mtp_w8a16.default_tile (R6 TILES empty)"


def sweep_bf16(n, k, mrows, ws, copies, x, tiles):
    one = torch.ones(n, device=DEV)
    res = []
    for t in tiles:
        try:
            res.append((rot(lambda i, t=t: m.w8a16_linear(x, ws[i], one, tile=t), copies), t))
        except Exception as e:  # noqa: BLE001
            print("tile error", n, k, t, repr(e)[:120], flush=True)
    return sorted(res)


def sec_r6t(a):
    out = {"shapes": {}, "tiles": {}}
    tot = {"cublas": 0.0, "r6": 0.0, "r6t": 0.0}
    for name, n, k, calls in R6_SHAPES:
        wb = n * k * 2
        copies = copies_for(wb)
        ws = [(torch.randn(n, k, device=DEV) / k ** 0.5).to(torch.bfloat16) for _ in range(copies)]
        one = torch.ones(n, device=DEV)
        t6, src = r6_tile(n, k)
        sres = {"calls": calls, "r6_tile": list(t6), "r6_tile_src": src, "m": {}}
        for mrows in (1, 7):
            x = torch.randn(mrows, k, device=DEV).to(torch.bfloat16)
            sw = sweep_bf16(n, k, mrows, ws, copies, x, ext_tiles(n, k))
            best = sw[0][1]
            arms = ab_rounds({"cublas": lambda i: F.linear(x, ws[i]),
                              "r6": lambda i: m.w8a16_linear(x, ws[i], one, tile=t6),
                              "r6t": lambda i: m.w8a16_linear(x, ws[i], one, tile=best)}, copies)
            y, y2 = m.w8a16_linear(x, ws[0], one, tile=best), m.w8a16_linear(x, ws[0], one, tile=best)
            ref = x.double() @ ws[0].double().t()
            rowmax = ref.abs().amax(dim=1, keepdim=True)
            d = (y.double() - ref).abs()
            chk = {"repeat": bool(torch.equal(y.view(torch.int16), y2.view(torch.int16))),
                   "within": bool(((d <= ulp_bf16(ref) * 1.0001) | (d <= rowmax * 2.0 ** -7)).all())}
            sres["m"][mrows] = {"best": list(best), "arms": arms, "check": chk,
                                "sweep_top": [(round(t, 2), list(tt)) for t, tt in sw[:6]]}
            print("r6t", name, mrows, {kk: v["us"] for kk, v in arms.items()}, best, chk, flush=True)
            if mrows == 7:
                for kk in tot:
                    tot[kk] += calls * arms[kk]["us"]
                out["tiles"][f"{n},{k}"] = list(best)
        out["shapes"][name] = sres
    ex = all(v["m"][mm]["check"]["repeat"] and v["m"][mm]["check"]["within"]
             for v in out["shapes"].values() for mm in (1, 7))
    out["verify_step_us_m7"] = {kk: round(v, 1) for kk, v in tot.items()}
    out["gate"] = {"saving_vs_r6_ms": round((tot["r6"] - tot["r6t"]) / 1e3, 3), "exact": ex,
                   "pass": ex and (tot["r6"] - tot["r6t"]) >= 120.0}
    print("r6t gate", out["gate"], flush=True)
    return out


# ------------------------------------------------------------------ s2
S2_SHAPES = [("qkv", 13312, 2560, 7, 1, 1), ("o_proj", 2560, 6144, 7, 1, 1), ("indexer", 640, 2560, 7, 1, 1),
             ("fc_embedding", 2560, 2560, 7, 1, 1), ("fc_hidden", 2560, 2560, 28, 4, 1),
             ("hc_down_inject", 336, 10240, 7, 1, 2), ("hc_mixer_down", 320, 10240, 7, 1, 1),
             ("hc_up", 10240, 320, 7, 1, 3), ("shared_gate_up", 1280, 2560, 7, 1, 1),
             ("shared_down", 2560, 640, 7, 1, 1), ("router", 512, 2560, 7, 1, 1)]


def sec_s2(a):
    out = {"shapes": {}, "tiles": {}}
    tot = {"cur": 0.0, "s2": 0.0}
    for name, n, k, m0, m1, cnt in S2_SHAPES:
        wb = n * k + n * 4
        copies = copies_for(wb)
        q8 = [m.quantize_rows((torch.randn(n, k, device=DEV) / k ** 0.5).to(torch.bfloat16)) for _ in range(copies)]
        cur = m.TILES.get((n, k)) or m.default_tile(n, k)
        sres = {"cur_tile": list(cur), "m": {}}
        best_by_m = {}
        for mrows, passes in ((m1, 5), (m0, 1)):
            x = torch.randn(mrows, k, device=DEV).to(torch.bfloat16)
            sw = []
            for t in ext_tiles(n, k, bks=(128, 256)):
                try:
                    sw.append((rot(lambda i, t=t: m.w8a16_linear(x, q8[i][0], q8[i][1], tile=t), copies), t))
                except Exception as e:  # noqa: BLE001
                    print("tile error", name, t, repr(e)[:120], flush=True)
            sw.sort()
            best = sw[0][1]
            best_by_m[mrows] = best
            arms = ab_rounds({"cur": lambda i: m.w8a16_linear(x, q8[i][0], q8[i][1], tile=cur),
                              "s2": lambda i: m.w8a16_linear(x, q8[i][0], q8[i][1], tile=best)}, copies)
            sres["m"][mrows] = {"best": list(best), "arms": arms, "sweep_top": [(round(t, 2), list(tt))
                                                                                for t, tt in sw[:5]]}
            for kk in tot:
                tot[kk] += passes * cnt * arms[kk]["us"]
            print("s2", name, mrows, {kk: v["us"] for kk, v in arms.items()}, best, flush=True)
        # one tile per (N, K): the M = 1 winner (5 of the 6 passes)
        out["tiles"][f"{n},{k}"] = list(best_by_m[m1])
        out["shapes"][name] = sres
    out["step_us"] = {kk: round(v, 1) for kk, v in tot.items()}
    out["gate"] = {"saving_ms": round((tot["cur"] - tot["s2"]) / 1e3, 3), "pass": (tot["cur"] - tot["s2"]) >= 60.0}
    print("s2 gate", out["gate"], flush=True)
    return out


# ------------------------------------------------------------------ hc12
def sec_hc12(a):
    sys.path.insert(0, "/h")
    import hc12
    import hc12_gemv as g
    from ckpt import hc_modules, load_u16

    calls = {"down": 97, "up": 97}
    data_rows = {"down": 324}
    hc_tiles = {"down": [(32, 256), (16, 256), (64, 256), (32, 128)],
                "up": [(32, 128), (16, 128), (32, 64), (64, 128), (16, 64)]}
    res = {"shapes": {}}
    mods = [mm for mm in hc_modules() if mm["name"] != "final_mixer"]
    for part in ("down", "up"):
        wb16 = 336 * 10240 * 2 if part == "down" else 10240 * 320 * 2
        copies = min(40, copies_for(wb16))
        us = []
        for mod in mods[:copies]:
            u = load_u16(mod[part])
            if part == "down":
                u = np.concatenate([u, np.zeros((336 - u.shape[0], u.shape[1]), np.uint16)], 0)
            us.append(np.ascontiguousarray(u))
        n, k = us[0].shape
        wbf = [torch.from_numpy(u.view(np.int16)).view(torch.bfloat16).to(DEV) for u in us]
        one = torch.ones(n, device=DEV)
        enc, hbytes = {}, {}
        nd = data_rows.get(part, n)
        for tn, tk in hc_tiles[part]:
            ps = [hc12.encode(np.ascontiguousarray(u[:nd]), tn, tk) for u in us]
            enc[(tn, tk)] = [g.to_device(p, DEV, n_out=n) for p in ps]
            hbytes[(tn, tk)] = sum(hc12.nbytes(p)["total"] for p in ps) / len(ps)
        sres = {"n": n, "k": k, "copies": copies, "m": {}}
        for mrows in (1, 7):
            x = torch.randn(mrows, k, device=DEV).to(torch.bfloat16)
            swb = sweep_bf16(n, k, mrows, wbf, copies, x, ext_tiles(n, k))
            bb = swb[0][1]
            swc = []
            for tn, tk in hc_tiles[part]:
                for split in (1, 2, 4, 8, 16):
                    if split > 1 and k // split < tk:
                        continue
                    for warps in (4, 8):
                        for fast in (True, False):
                            t = (tn, tk, split, warps, 3, fast)
                            try:
                                swc.append((rot(lambda i, t=t: g.hc12_linear(
                                    x, enc[t[:2]][i], split=t[2], warps=t[3], stages=t[4], fast=t[5]), copies), t))
                            except Exception as e:  # noqa: BLE001
                                print("tile error C", t, repr(e)[:120], flush=True)
            swc.sort()
            bc = swc[0][1]
            arms = ab_rounds({"B_r6_best": lambda i: m.w8a16_linear(x, wbf[i], one, tile=bb),
                              "C_hc12": lambda i: g.hc12_linear(x, enc[bc[:2]][i], split=bc[2], warps=bc[3],
                                                                 stages=bc[4], fast=bc[5]),
                              "A_cublas": lambda i: F.linear(x, wbf[i])}, copies)
            exact = []
            for _, t in swc[:6]:
                for i in range(min(3, copies)):
                    y = g.hc12_linear(x, enc[t[:2]][i], split=t[2], warps=t[3], stages=t[4], fast=t[5])
                    y2 = g.hc12_linear(x, enc[t[:2]][i], split=t[2], warps=t[3], stages=t[4], fast=t[5])
                    r6 = m.w8a16_linear(x, wbf[i], one, tile=(t[0], t[1], t[2], t[3], t[4]))
                    exact.append(bool(torch.equal(y.view(torch.int16), r6.view(torch.int16)))
                                 and bool(torch.equal(y.view(torch.int16), y2.view(torch.int16))))
            sres["m"][mrows] = {"best_B": list(bb), "best_C": list(bc), "hc12_bytes": hbytes[bc[:2]],
                                "arms": arms, "exact_all": all(exact)}
            print("hc12", part, mrows, {kk: v["us"] for kk, v in arms.items()}, bb, bc, all(exact), flush=True)
        res["shapes"][part] = sres
    s = {arm: sum(calls[p] * res["shapes"][p]["m"][7]["arms"][arm]["us"] for p in ("down", "up"))
         for arm in ("A_cublas", "B_r6_best", "C_hc12")}
    band = {arm: max(res["shapes"][p]["m"][7]["arms"][arm]["band"] for p in ("down", "up")) for arm in s}
    ratio = s["C_hc12"] / s["B_r6_best"]
    g1 = ratio <= 0.80 and (s["B_r6_best"] - s["C_hc12"]) > band["C_hc12"] * s["C_hc12"] + band["B_r6_best"] * s["B_r6_best"]
    g23 = all(res["shapes"][p]["m"][mm]["exact_all"] for p in ("down", "up") for mm in (1, 7))
    res["verify_step_us_m7"] = {kk: round(v, 1) for kk, v in s.items()}
    res["gate"] = {"ratio_vs_item1": round(ratio, 4), "G1": g1, "G2_G3": g23,
                   "saving_vs_item1_ms": round((s["B_r6_best"] - s["C_hc12"]) / 1e3, 3), "pass": g1 and g23}
    print("hc12 gate", res["gate"], flush=True)
    return res


# ------------------------------------------------------------------ h1
def meminfo():
    d = {}
    with open("/proc/meminfo") as f:
        for ln in f:
            kk, v = ln.split(":", 1)
            d[kk] = int(v.split()[0]) * 1024
    return {"MemAvailable_gib": round(d.get("MemAvailable", 0) / 2 ** 30, 2),
            "MemFree_gib": round(d.get("MemFree", 0) / 2 ** 30, 2)}


def sec_h1(a):
    import triton
    import triton.language as tl

    import hostalloc as ha

    @triton.jit
    def _stream(X, S, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        acc = tl.zeros((BLOCK,), tl.int32)
        for off in range(pid * BLOCK, N, tl.num_programs(0) * BLOCK):
            idx = off + tl.arange(0, BLOCK)
            acc ^= tl.load(X + idx, idx < N, other=0)
        tl.store(S + pid, tl.sum(acc, 0))  # keeps the loads live

    mode = os.environ.get("KERN_HOSTALLOC_MODE", "host")
    res = {"mode": mode, "mem_before": meminfo(), "kernels": {}}
    pool = ha.pool()

    def make(shape, dtype, in_pool):
        if in_pool:
            with torch.cuda.use_mem_pool(pool):
                return torch.empty(shape, dtype=dtype, device=DEV)
        return torch.empty(shape, dtype=dtype, device=DEV)

    # 1 GiB stream read
    nint = (1 << 30) // 4
    bufs = {p: make((nint,), torch.int32, p) for p in (False, True)}
    for b in bufs.values():
        b.random_(0, 1 << 30)
    sink = torch.empty(4096, dtype=torch.int32, device=DEV)
    grid = (192,)
    res["kernels"]["stream_1g"] = {"bytes": 1 << 30, **ab_rounds(
        {"cudaMalloc": lambda i: _stream[grid](bufs[False], sink, nint, BLOCK=1024, num_warps=8),
         "pool": lambda i: _stream[grid](bufs[True], sink, nint, BLOCK=1024, num_warps=8)}, 1, rounds=8)}
    del bufs
    torch.cuda.empty_cache()
    # GEMV-shaped kernels on cold weights
    from flashinfer import mxfp8_quantize
    from flashinfer.gemm import gemm_base as gb

    major, _ = gb.get_compute_capability(DEV)
    runner = gb.get_cutlass_mxfp8_gemm_module(major).cutlass_mxfp8_gemm_runner()
    ws_buf = gb._get_cache_buf("mm_mxfp8_workspace", gb.DEFAULT_WORKSPACE_SIZE, DEV)
    cases = [("r6_hc_down_bf16", 336, 10240, "bf16"), ("r4_mtp_qkv_fp8", 13312, 2560, "fp8row"),
             ("fi_mxfp8_qkvz", 16384, 2560, "mxfp8"), ("cublas_bf16_o", 2560, 6144, "cublas")]
    for name, n, k, kind in cases:
        per = n * k * (2 if kind in ("bf16", "cublas") else 1)
        copies = copies_for(per, 48)
        sets = {}
        for p in (False, True):
            ws = []
            for _ in range(copies):
                w = (torch.randn(n, k, device=DEV) / k ** 0.5).to(torch.bfloat16)
                if kind in ("bf16", "cublas"):
                    t = make((n, k), torch.bfloat16, p)
                    t.copy_(w)
                    ws.append((t,))
                elif kind == "fp8row":
                    w8, s = m.quantize_rows(w)
                    t = make(w8.shape, w8.dtype, p)
                    t.copy_(w8)
                    ws.append((t, s))
                else:
                    wm, wsf = mxfp8_quantize(w, is_sf_swizzled_layout=True)
                    t = make(wm.shape, wm.dtype, p)
                    t.copy_(wm)
                    ws.append((t, wsf.reshape(-1).contiguous()))
            sets[p] = ws
        x = torch.randn(7, k, device=DEV).to(torch.bfloat16)
        one = torch.ones(n, device=DEV)
        outb = torch.empty(7, n, device=DEV, dtype=torch.bfloat16)
        am, asf = mxfp8_quantize(x, is_sf_swizzled_layout=True)

        def fn(p, kind=kind):
            ws = sets[p]
            if kind == "bf16":
                return lambda i: m.w8a16_linear(x, ws[i][0], one)
            if kind == "fp8row":
                return lambda i: m.w8a16_linear(x, ws[i][0], ws[i][1])
            if kind == "cublas":
                return lambda i: F.linear(x, ws[i][0])
            return lambda i: runner.forward([am, ws[i][0].t(), asf.reshape(-1), ws[i][1], None, outb, ws_buf],
                                            tactic=1)

        r = ab_rounds({"cudaMalloc": fn(False), "pool": fn(True)}, copies, rounds=8)
        r["bytes"] = per
        r["speedup"] = round(r["cudaMalloc"]["us"] / r["pool"]["us"] - 1, 4)
        res["kernels"][name] = r
        print("h1", mode, name, r["cudaMalloc"]["us"], r["pool"]["us"], r["speedup"], flush=True)
        del sets
        torch.cuda.empty_cache()
    # big allocation check
    big = {"skipped": True}
    if meminfo()["MemAvailable_gib"] > 64:
        t0 = time.time()
        chunks = []
        try:
            for _ in range(32):
                chunks.append(make((1 << 30,), torch.uint8, True))
            big = {"skipped": False, "gib": len(chunks), "s": round(time.time() - t0, 2),
                   "mem_during": meminfo(), "alloc": ha.stats()}
        except Exception as e:  # noqa: BLE001
            big = {"skipped": False, "error": repr(e)[:300], "gib": len(chunks)}
        del chunks
        torch.cuda.empty_cache()
    res["big_alloc"] = big
    res["mem_after"] = meminfo()
    g = [res["kernels"][c[0]]["speedup"] for c in cases]
    band = max(max(res["kernels"][c[0]]["cudaMalloc"]["band"], res["kernels"][c[0]]["pool"]["band"]) for c in cases)
    res["gate"] = {"gemv_speedup_median": round(float(np.median(g)), 4), "band_max": band,
                   "pass": float(np.median(g)) >= 0.02 and float(np.median(g)) > band}
    print("h1 gate", mode, res["gate"], flush=True)
    return res


SECTIONS = {"s1": sec_s1, "s3": sec_s3, "r6t": sec_r6t, "s2": sec_s2, "hc12": sec_hc12, "h1": sec_h1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", required=True, choices=sorted(SECTIONS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    head = {"node": socket.gethostname(), "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "section": a.only}
    print(head, flush=True)
    t0 = time.time()
    res = SECTIONS[a.only](a)
    res.update(head)
    res["seconds"] = round(time.time() - t0, 1)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
