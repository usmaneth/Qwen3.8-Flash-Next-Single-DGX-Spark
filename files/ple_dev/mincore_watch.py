#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G24 CPU study: page-cache residency of the packed PLE table over time.

    python3 mincore_watch.py --table PATH --out DIR [--every 20] [--hours 3]
                             [--snap-every 300]

The script maps the table read-only and calls mincore(2) on the full mapping.
mincore does not read the file and does not fault pages in. The script does
not touch the table pages and does not change the page cache.

Each sample appends one line to DIR/mincore.tsv:
  utc  resident_pages  total_pages  resident_frac  mem_available_kib
  mem_free_kib  cached_kib  lease_label
Each --snap-every seconds, the script also writes the full residency vector
as a packed bit array (np.packbits) to DIR/snap-<utc>.npy. A later step maps
row ids to pages and reads the cold-miss rate from these snapshots.

Permission trap: the kernel reports true page-cache residency only when the
caller owns the file or can open it for writing (mm/mincore.c
can_do_mincore). For all other callers, mincore sets every byte to 1, so
the table looks 100% resident. The table file is root:0644, so run this
script as root. The script stops when a probe says the answer is not real.
"""
import argparse
import ctypes
import mmap
import os
import subprocess
import time

import numpy as np

LEASE = "/models/usman/pulse-tp/u0/bin/gpu-lease"


def meminfo() -> dict:
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            out[k] = int(v.split()[0])
    return out


def lease_label() -> str:
    try:
        cmd = [LEASE, "status"]
        if os.geteuid() == 0 and os.environ.get("SUDO_USER"):
            cmd = ["runuser", "-u", os.environ["SUDO_USER"], "--"] + cmd
        s = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return "?"
    for line in s.splitlines():
        if line.startswith(os.uname().nodename):
            for tok in line.split():
                if tok.startswith("label="):
                    return tok[6:]
            return line.split()[1] if len(line.split()) > 1 else "?"
    return "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=float, default=20.0)
    ap.add_argument("--snap-every", type=float, default=300.0)
    ap.add_argument("--hours", type=float, default=3.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fd = os.open(a.table, os.O_RDONLY)
    size = os.fstat(fd).st_size
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
    if addr in (None, ctypes.c_void_p(-1).value):
        raise OSError(ctypes.get_errno(), "mmap failed")
    page = os.sysconf("SC_PAGE_SIZE")
    n = (size + page - 1) // page
    st = os.fstat(fd)
    if os.geteuid() != 0 and st.st_uid != os.geteuid() and not os.access(a.table, os.W_OK):
        raise SystemExit("mincore would report fake residency (not owner, no write "
                         "access): run as root")
    vec = (ctypes.c_ubyte * n)()
    path = os.path.join(a.out, "mincore.tsv")
    new = not os.path.exists(path)
    f = open(path, "a", buffering=1)
    if new:
        f.write("utc\tresident_pages\ttotal_pages\tresident_frac\tmem_available_kib"
                "\tmem_free_kib\tcached_kib\tlease\n")
    t_end = time.time() + a.hours * 3600
    t_snap = 0.0
    while time.time() < t_end:
        if libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) != 0:
            raise OSError(ctypes.get_errno(), "mincore failed")
        bits = np.frombuffer(vec, dtype=np.uint8) & 1
        res = int(bits.sum())
        m = meminfo()
        utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        f.write(f"{utc}\t{res}\t{n}\t{res / n:.5f}\t{m['MemAvailable']}\t{m['MemFree']}"
                f"\t{m['Cached']}\t{lease_label()}\n")
        if time.time() - t_snap >= a.snap_every:
            np.save(os.path.join(a.out, f"snap-{utc}.npy"), np.packbits(bits))
            t_snap = time.time()
        time.sleep(a.every)


if __name__ == "__main__":
    main()
