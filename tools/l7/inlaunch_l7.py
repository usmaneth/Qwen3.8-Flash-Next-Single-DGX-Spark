#!/usr/bin/env python3
"""L7 wrapper of kern-decode ab/inlaunch.py: the same arms, plus the H1 memory stops.

Each kd_set of an arm with a "call" knob (the H1 weight move) is checked:
  - the freed share of a move to the pool is 0.95 or more (a smaller share
    means that a stale reference keeps the old copy alive);
  - MemAvailable after the kd_set is not more than 2 GiB below the value
    before it.
A failed check raises, so inlaunch.main marks the arm as failed and skips
its later visits. The rows go to <out>/h1_moves.jsonl.

    inlaunch_l7.py --plan plan.json --out DIR [--port 8888] [--deadline T]
"""
import json
import os
import sys

sys.path.insert(0, "/models/usman/kern-decode/ab")
import inlaunch  # noqa: E402

_orig_rpc = inlaunch.rpc
_OUT = {"dir": None}


def mem_avail_gib():
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1048576
    return 0.0


def rpc(port, method, *args, timeout=600):
    if method != "kd_set":
        return _orig_rpc(port, method, *args, timeout=timeout)
    knobs = json.loads(args[0])
    calls = knobs.get("call") or {}
    m0 = mem_avail_gib()
    res = _orig_rpc(port, method, *args, timeout=timeout)
    if not calls:
        return res
    m1 = mem_avail_gib()
    row = {"calls": calls, "mem_before_gib": round(m0, 2), "mem_after_gib": round(m1, 2), "set": res}
    bad = []
    # kd_set returns {"done": {"call:<target>": <result>, ...}, "info": ...}.
    done = res.get("done", res) if isinstance(res, dict) else {}
    for key, val in done.items():
        if key.startswith("call:") and "hostalloc" in key and isinstance(val, dict):
            fs = val.get("freed_share")
            if val.get("aborted"):
                bad.append(f"move aborted: {val['aborted']}")
            if val.get("to") == "host" and val.get("bytes") and (fs is None or fs < 0.95):
                bad.append(f"freed share {fs} < 0.95")
    if m0 - m1 > 2.0:
        bad.append(f"MemAvailable fell {m0 - m1:.2f} GiB > 2")
    row["stop"] = bad
    if _OUT["dir"]:
        with open(os.path.join(_OUT["dir"], "h1_moves.jsonl"), "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
    print(f"h1 move {calls}: MemAvailable {m0:.1f} -> {m1:.1f} GiB, stop {bad}", flush=True)
    if bad:
        raise RuntimeError("H1 stop: " + "; ".join(bad))
    return res


inlaunch.rpc = rpc

if __name__ == "__main__":
    if "--out" in sys.argv:
        _OUT["dir"] = sys.argv[sys.argv.index("--out") + 1]
    inlaunch.main()
