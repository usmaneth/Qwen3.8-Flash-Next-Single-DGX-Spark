#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G24: size the cold-row rate of an in-graph PLE gather from real streams.

    python3 coldmiss.py --streams streams.npz --g24 DIR --out DIR

A device gather (TECHNIQUES.md #7, rung R2) reads each row from the page
cache through ATS. A row whose 4 KiB page is not resident stalls the step
until the page arrives from the NVMe. This script counts, per decode step,
the pages that are not resident, for real token streams and these states of
the page cache:
  S2 relief    empty cache at decode start (after a drop_caches relief).
               Only the earlier steps of the same decode are warm.
  S1 own       the rows of the request's own prompt are resident (its
               prefill read them), plus the earlier decode steps.
  S3 snap+own  S1 plus the pages that the G24 mincore snapshot shows
               resident (a real served Codex session on spark1).
  S4 session   all records in file order share one cache that starts empty
               and never evicts (upper bound on warmth).
  S5 lru-N     S4 with an LRU of N GiB. The LRU uses the touch count as the
               distance, which is at least the distinct-page distance, so
               it counts more misses than a true LRU (conservative).
A decode step sends [bonus, d1 .. d6] (streams.Replay, measured acceptance),
so the rejected drafts are in the row set too. The step rows are the 16 row
ids of each token (ple_gpu_twin.ids_twin, parity-proven against the worker).
A row that crosses a page boundary needs both pages.

It also summarizes DIR/mincore.tsv (the residency time series) and reports
the hit rate of the decode rows in the snapshot against the resident
fraction (a hit rate above the fraction means a hot set that sessions share).
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import parity_real as pr  # noqa: E402
import ple_gpu_twin as tw  # noqa: E402
import streams as sm  # noqa: E402

ROW = 90


def row_pages(rids: np.ndarray) -> np.ndarray:
    off = rids.reshape(-1).astype(np.int64) * ROW
    return np.unique(np.concatenate((off >> 12, (off + ROW - 1) >> 12)))


def pct(a, qs=(50, 90, 99, 99.9)):
    a = np.asarray(a)
    if a.size == 0:
        return {}
    out = {f"p{q}": float(np.percentile(a, q)) for q in qs}
    out.update(mean=float(a.mean()), max=int(a.max()), zero_frac=float((a == 0).mean()),
               n=int(a.size))
    return out


def load_tsv(path: str) -> dict:
    rows = [line.rstrip("\n").split("\t") for line in open(path)]
    hdr, rows = rows[0], rows[1:]
    res = np.asarray([int(r[1]) for r in rows])
    tot = int(rows[0][2]) if rows else 0
    out = {"samples": len(rows), "first": rows[0][0] if rows else None,
           "last": rows[-1][0] if rows else None, "total_pages": tot,
           "resident_last": int(res[-1]) if rows else 0,
           "resident_max": int(res.max()) if rows else 0,
           "labels": sorted(set(r[7] for r in rows)),
           "mem_available_gib_min": min(int(r[4]) for r in rows) / 2**20 if rows else None}
    # Growth: pages per minute over the longest run of non-decreasing samples.
    t = [time.mktime(time.strptime(r[0], "%Y%m%dT%H%M%SZ")) for r in rows]
    best = (0, 0, 0)
    i = 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and res[j + 1] >= res[j]:
            j += 1
        if res[j] - res[i] > best[0]:
            best = (int(res[j] - res[i]), i, j)
        i = j + 1
    if best[0]:
        _, i, j = best
        out["growth"] = {"from": rows[i][0], "to": rows[j][0], "pages": best[0],
                         "pages_per_min": round(best[0] / ((t[j] - t[i]) / 60), 1),
                         "mib_per_s": round(best[0] * 4096 / 2**20 / (t[j] - t[i]), 2)}
    drops = [(rows[k][0], int(res[k - 1]), int(res[k])) for k in range(1, len(rows))
             if res[k] < 0.8 * res[k - 1] and res[k - 1] > 1000]
    out["drops"] = drops[:20]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", required=True)
    ap.add_argument("--g24", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--lru-gib", type=float, nargs="*", default=[1.5, 6.0])
    ap.add_argument("--seed", type=int, default=25)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    layer, _, tc, _ = pr.build_layer(pr.REF, pr.default_config(), 0, 8192 + 256, 8)
    k = tw.Consts.from_layer(layer)
    st = sm.Streams(a.streams)
    meta = json.load(open(pr.TABLE + ".json"))
    npages = (int(meta["total_rows"]) * ROW + 4095) // 4096
    tsv = load_tsv(os.path.join(a.g24, "mincore.tsv"))
    snaps = sorted(glob.glob(os.path.join(a.g24, "snap-*.npy")))
    # The snapshot with the most resident pages (a relief or a boot empties it).
    best, snap = None, None
    for f in snaps:
        b = np.unpackbits(np.load(f))[:npages].astype(bool)
        if snap is None or b.sum() > snap.sum():
            best, snap = f, b
    snap_frac = float(snap.mean()) if snap is not None else None

    rng = np.random.default_rng(a.seed)
    rec = range(len(st)) if not a.max_records else range(min(a.max_records, len(st)))
    own = np.zeros(npages, np.int32)  # record stamp of the S1 touches
    dec = np.zeros(npages, np.int32)  # record stamp of the S2 touches
    sess = np.zeros(npages, bool)
    lru = {g: np.full(npages, -(1 << 62), np.int64) for g in a.lru_gib}
    cap = {g: int(g * 2**30 / 4096) for g in a.lru_gib}
    clock = 0
    cold = {"S2_relief": [], "S1_own": [], "S3_snap_own": [], "S4_session": []}
    for g in a.lru_gib:
        cold[f"S5_lru_{g:g}GiB"] = []
    step_pages, step_rows, snap_hits, snap_rows = [], [], 0, 0
    per_rec = []
    # Cold pages by token position in the step (0 = bonus, i = draft i), S1 and S4.
    pos_s1 = np.zeros(7, np.int64)
    pos_s4 = np.zeros(7, np.int64)
    pos_n = np.zeros(7, np.int64)
    eos = k.eos
    for r in rec:
        stamp = r + 1
        seq = st.seq(r)
        lp = int(st.plen[r])
        # Prompt rows: one pass over the whole prompt (context before 0 is EOS).
        q = np.asarray([0, lp], np.int32)
        nctx = np.asarray([[eos, eos]], np.int32)
        pr_ids = tw.ids_twin(k, seq[:lp].astype(np.int32), q, nctx)
        pp = row_pages(pr_ids)
        own[pp] = stamp
        sess[pp] = True
        for g in a.lru_gib:
            lru[g][pp] = clock + np.arange(pp.size)
        clock += pp.size
        rep = sm.Replay(st, [r], rng, k=6, max_seqs=1, budget=8192)
        rep.run = [sm._Req(r, seq, lp, lp)]
        rep.next = 1
        rep.target = 1
        n_dec = 0
        c0 = len(cold["S1_own"])
        while rep.run:
            out = rep.step()
            if out is None:
                break
            segs, ctxs = out
            toks = segs[0][2]
            ids = tw.ids_twin(k, toks, np.asarray([0, toks.size], np.int32),
                              np.asarray([ctxs[0]], np.int32))
            pg = row_pages(ids)
            for j in range(min(7, ids.shape[0])):
                pj = row_pages(ids[j])
                pos_s1[j] += int((own[pj] != stamp).sum())
                pos_s4[j] += int((~sess[pj]).sum())
                pos_n[j] += 1
            step_pages.append(pg.size)
            step_rows.append(ids.size)
            cold["S2_relief"].append(int((dec[pg] != stamp).sum()))
            miss_own = (own[pg] != stamp)
            cold["S1_own"].append(int(miss_own.sum()))
            if snap is not None:
                cold["S3_snap_own"].append(int((miss_own & ~snap[pg]).sum()))
                snap_hits += int(snap[pg].sum())
                snap_rows += pg.size
            cold["S4_session"].append(int((~sess[pg]).sum()))
            for g in a.lru_gib:
                cold[f"S5_lru_{g:g}GiB"].append(int((clock - lru[g][pg] >= cap[g]).sum()))
                lru[g][pg] = clock + np.arange(pg.size)
            clock += pg.size
            dec[pg] = stamp
            own[pg] = stamp
            sess[pg] = True
            n_dec += 1
        per_rec.append((r, lp, seq.size - lp, n_dec,
                        float(np.mean(cold["S1_own"][c0:])) if n_dec else 0.0))
    res = {"date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "node": os.uname().nodename, "records": len(per_rec),
           "decode_steps": len(step_pages), "table_pages": npages,
           "rows_per_step": pct(step_rows), "pages_per_step": pct(step_pages),
           "cold_pages_per_step": {s: pct(v) for s, v in cold.items() if v},
           "snapshot": {"file": best, "snapshots": len(snaps), "resident_frac": snap_frac,
                        "decode_page_hit_rate": snap_hits / snap_rows if snap_rows else None},
           "mincore_tsv": tsv,
           "session_touched_frac": float(sess.mean()),
           "cold_pages_by_position": {
               "note": "0 = bonus token, i = draft i; own rows of earlier steps count warm",
               "S1_own": (pos_s1 / np.maximum(pos_n, 1)).round(2).tolist(),
               "S4_session": (pos_s4 / np.maximum(pos_n, 1)).round(2).tolist(),
               "steps": pos_n.tolist()}}
    json.dump(res, open(os.path.join(a.out, "coldmiss.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "coldmiss-steps.npz"),
                        pages=np.asarray(step_pages, np.int16),
                        **{s: np.asarray(v, np.int16) for s, v in cold.items() if v})
    print(json.dumps({s: v.get("p99") for s, v in res["cold_pages_per_step"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
