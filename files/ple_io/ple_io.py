# SPDX-License-Identifier: Apache-2.0
"""PLE row I/O for the CPU offload worker: the row gather and its trace.

The container mounts this file as ``vllm.v1.ple_offload.ple_io``. The
generated ple_layer, worker and connector files call it through small hooks
that files/ple_io/patch_ple_io.py adds (see that file).

``gather()`` reads the packed PLE rows for one forward from the
memory-mapped table into the output buffer. The default mode is the
shipped path: one posix_fadvise(WILLNEED) per unique 4 KiB page, then one
torch.index_select over the mmap. The result is always the plain gather of
the same rows, so every mode gives the same bytes.

Environment (read once at import):
  VLLM_PLE_IO_MODE        fadvise (default, the shipped path) | none
                          (no prefetch: each missing page is a fault).
  VLLM_PLE_IO_TRACE_DIR   empty (default) = no trace. Else a directory for
                          one TSV file per process and record kind.
  VLLM_PLE_IO_TRACE_RES   pages to check with mincore before the prefetch,
                          per gather (default 128, 0 = no check). The check
                          reads the page-cache state only; it starts no I/O.

With a trace directory, the file <dir>/ctl (lines "mode=fadvise|none",
"trace=0|1", "res=N") is read each second, so a benchmark can change the
arm without a server restart. The last applied state goes to the trace as
a "ctl" record.
"""

import atexit
import ctypes
import os
import random
import resource
import threading
import time

import torch

_PAGE_SHIFT = 12
_PAGE = 1 << _PAGE_SHIFT

MODE = os.environ.get("VLLM_PLE_IO_MODE", "fadvise").strip() or "fadvise"
if MODE not in ("fadvise", "none"):
    MODE = "fadvise"
TRACE_DIR = os.environ.get("VLLM_PLE_IO_TRACE_DIR", "").strip()
TRACE_RES = int(os.environ.get("VLLM_PLE_IO_TRACE_RES", "128") or 0)
# The hooks test TRACE. It is True when TRACE_DIR is set, and the control
# file can change it (see _read_ctl).
TRACE = bool(TRACE_DIR)

_now = time.perf_counter_ns


class _TraceFile:
    """Buffered TSV records. A daemon thread writes the buffer each second."""

    def __init__(self, kind: str, header: str) -> None:
        os.makedirs(TRACE_DIR, exist_ok=True)
        path = os.path.join(TRACE_DIR, f"{kind}-{os.getpid()}.tsv")
        self._f = open(path, "a", buffering=1 << 16)
        self._f.write("# " + header + "\n")
        self._f.flush()
        self._buf: list[str] = []
        self._lock = threading.Lock()
        atexit.register(self.flush)

    def add(self, *fields) -> None:
        line = "\t".join(str(x) for x in fields)
        with self._lock:
            self._buf.append(line)

    def flush(self) -> None:
        with self._lock:
            buf, self._buf = self._buf, []
            if buf:
                self._f.write("\n".join(buf) + "\n")
                self._f.flush()



_traces: dict[str, _TraceFile] = {}
_trace_lock = threading.Lock()
_ctl_mtime = 0.0


def _read_ctl() -> None:
    """Apply <TRACE_DIR>/ctl if it changed. Unknown lines are ignored."""
    global MODE, TRACE, TRACE_RES, _ctl_mtime
    path = os.path.join(TRACE_DIR, "ctl")
    try:
        mtime = os.stat(path).st_mtime
    except OSError:
        return
    if mtime == _ctl_mtime:
        return
    _ctl_mtime = mtime
    try:
        for line in open(path).read().split():
            key, _, val = line.partition("=")
            if key == "mode" and val in ("fadvise", "none"):
                MODE = val
            elif key == "trace" and val in ("0", "1"):
                TRACE = val == "1"
            elif key == "res" and val.isdigit():
                TRACE_RES = int(val)
    except OSError:
        return
    _trace("ctl", _CTL_HDR).add(_now(), MODE, int(TRACE), TRACE_RES)


def _flush_loop() -> None:
    """Each second: apply the control file, then write every trace buffer."""
    while True:
        time.sleep(1.0)
        _read_ctl()
        for t in list(_traces.values()):
            t.flush()


def _trace(kind: str, header: str) -> _TraceFile:
    t = _traces.get(kind)
    if t is None:
        with _trace_lock:
            t = _traces.get(kind)
            if t is None:
                t = _traces[kind] = _TraceFile(kind, header)
    return t


_GATHER_HDR = ("t_ns mode rows uniq_rows pages res_checked res_hit "
               "pages_ns trace_ns advise_ns select_ns inblock_bytes_advise "
               "inblock_bytes_select majflt minflt")
_REQ_HDR = "t_ns num_tokens num_reqs forward_ns copy_ns total_ns"
_LAUNCH_HDR = "t_ns num_tokens num_reqs send_ns wait_ns total_ns"
_CTL_HDR = "t_ns mode trace res"

_libc = None


def _mincore_hits(base: int, pages: torch.Tensor, limit: int) -> tuple[int, int]:
    """Count the resident pages in a random sample of ``pages``."""
    global _libc
    if limit <= 0 or pages.numel() == 0:
        return 0, 0
    if _libc is None:
        _libc = ctypes.CDLL(None, use_errno=True)
        _libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.c_char_p)
    plist = pages.tolist()
    if len(plist) > limit:
        plist = random.sample(plist, limit)
    vec = ctypes.create_string_buffer(1)
    hits = 0
    for p in plist:
        if _libc.mincore(base + (p << _PAGE_SHIFT), _PAGE, vec) == 0:
            hits += vec.raw[0] & 1
    return len(plist), hits


def _pages(ids: torch.Tensor, row_width: int) -> torch.Tensor:
    offsets = ids.to(torch.int64) * row_width
    pages = torch.cat((offsets >> _PAGE_SHIFT,
                       (offsets + row_width - 1) >> _PAGE_SHIFT))
    # unique() sorts, so the reads are issued in ascending file order.
    return torch.unique(pages)


def _advise(fd, pages: torch.Tensor) -> None:
    for page in pages.tolist():
        os.posix_fadvise(fd, page << _PAGE_SHIFT, _PAGE,
                         os.POSIX_FADV_WILLNEED)


def gather(fd, table: torch.Tensor, ids: torch.Tensor,
           out: torch.Tensor) -> None:
    """Copy ``table[ids]`` into ``out`` (a uint8 [rows, row_width] view)."""
    if not TRACE:
        if MODE == "fadvise" and fd is not None and ids.numel():
            try:
                _advise(fd, _pages(ids, table.shape[-1]))
            except Exception:  # advisory prefetch only
                pass
        torch.index_select(table, 0, ids, out=out)
        return

    row_width = table.shape[-1]
    t0 = _now()
    pages = _pages(ids, row_width) if ids.numel() else ids[:0]
    t0b = _now()
    # Trace-only work (not in the shipped path): unique rows, residency.
    uniq_rows = int(torch.unique(ids).numel())
    res_checked, res_hit = _mincore_hits(table.data_ptr(), pages, TRACE_RES)
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    t1 = _now()
    if MODE == "fadvise" and fd is not None and pages.numel():
        try:
            _advise(fd, pages)
        except Exception:
            pass
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    t2 = _now()
    torch.index_select(table, 0, ids, out=out)
    t3 = _now()
    r2 = resource.getrusage(resource.RUSAGE_SELF)
    _trace("gather", _GATHER_HDR).add(
        t0, MODE, ids.numel(), uniq_rows, pages.numel(), res_checked,
        res_hit, t0b - t0, t1 - t0b, t2 - t1, t3 - t2,
        (r1.ru_inblock - r0.ru_inblock) * 512,
        (r2.ru_inblock - r1.ru_inblock) * 512,
        r2.ru_majflt - r0.ru_majflt, r2.ru_minflt - r0.ru_minflt)


def trace_request(t_start: int, t_forward: int, num_tokens: int,
                  num_reqs: int) -> None:
    """Worker side: one request, from receipt to the published flag."""
    t_end = _now()
    _trace("request", _REQ_HDR).add(
        t_start, num_tokens, num_reqs, t_forward - t_start,
        t_end - t_forward, t_end - t_start)


def trace_launch(t_start: int, t_sent: int, num_tokens: int,
                 num_reqs: int) -> None:
    """GPU-worker side: one PLE launch, from the D2H copy to the done flag."""
    t_end = _now()
    _trace("launch", _LAUNCH_HDR).add(
        t_start, num_tokens, num_reqs, t_sent - t_start, t_end - t_sent,
        t_end - t_start)


now = _now

if TRACE_DIR:
    threading.Thread(target=_flush_loop, daemon=True,
                     name="ple_io_trace").start()
