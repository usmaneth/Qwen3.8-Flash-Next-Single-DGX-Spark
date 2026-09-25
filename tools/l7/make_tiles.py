#!/usr/bin/env python3
"""Write <run>/l7_tiles.json (the l7.py table) from the micro sections that passed.

  python3 make_tiles.py <run dir>

skinny_bf16 (R6t) from r6t.json, mtp_w8a16 (S2) from s2.json, skinny_mx (S1)
from s1.json (the best unfused tile at M = 7). A section that did not run or
did not pass its gate adds nothing; the file records why.
"""
import json
import os
import sys


def main(run):
    table, note = {}, {}
    for sec, mod in (("r6t", "skinny_bf16"), ("s2", "mtp_w8a16"), ("s1", "skinny_mx")):
        p = os.path.join(run, f"{sec}.json")
        if not os.path.exists(p):
            note[sec] = "not run"
            continue
        d = json.load(open(p))
        gate = d.get("gate", {})
        ok = gate.get("pass", gate.get("S1_pass"))
        if not ok:
            note[sec] = f"gate failed: {gate}"
            continue
        if sec == "s1":
            tiles = {k.replace("x", ","): v["m"]["7"]["best_unfused"] if "7" in v["m"] else v["m"][7]["best_unfused"]
                     for k, v in d["shapes"].items()}
        else:
            tiles = d["tiles"]
        table[mod] = tiles
        note[sec] = f"pass: {len(tiles)} tiles"
    table["_note"] = note
    out = os.path.join(run, "l7_tiles.json")
    json.dump(table, open(out, "w"), indent=1)
    print(json.dumps(note))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
