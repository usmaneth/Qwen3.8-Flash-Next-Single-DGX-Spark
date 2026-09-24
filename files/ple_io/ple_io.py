# SPDX-License-Identifier: Apache-2.0
"""PLE row I/O for the CPU offload worker: the row gather, its trace and gates.

The container mounts this file as ``vllm.v1.ple_offload.ple_io``. The
generated ple_layer, worker and connector files call it through small hooks
that files/ple_io/patch_ple_io.py adds (see that file).

``gather()`` reads the packed PLE rows for one forward from the
memory-mapped table into the output buffer. The prefetch mode only tells
the kernel which pages to read. The bytes always come from the same
``torch.index_select`` over the mmap, so every mode gives the same bytes.

Modes (key ``mode``):
  fadvise  the shipped path: one posix_fadvise(WILLNEED) per unique 4 KiB
           page from a Python loop, then index_select.
  none     no prefetch: each missing page is a fault.
  batch    one batched call for all pages (key ``backend``):
             c   ple_io_native.c: a byte-map dedup, then ``threads``
                 pthreads of posix_fadvise(WILLNEED) above 4096 pages.
             pm  process_madvise(MADV_WILLNEED) with iovec batches of 1024
                 over the table mapping, ``threads`` pool threads above
                 4096 pages.
           An error in the backend uses the fadvise loop for that gather.
           After 3 errors the gather stays on the fadvise loop.

Control keys. The value order is: the ctl file (when VLLM_PLE_IO_TRACE_DIR
is set) over the env VLLM_PLE_IO_<KEY> over the default.
  mode=fadvise|none|batch   backend=c|pm   threads=N (1-64)
  defer=0|1 (ple_io_defer)  fast=0|1 (ple_io_fast)
  check=0|1                 gate G3: digests of the rows on both sides
  trace=0|1                 per-gather, per-request and per-launch records
  res=N                     pages to check with mincore per traced gather.
                            res=0 also skips the trace-only unique() count.
  id=TEXT                   echoed in the ctl record (the harness waits for it)

VLLM_PLE_IO_TRACE_DIR: empty (default) = no ctl file and no records. Else a
directory for one TSV file per process and record kind, and for the ctl file
<dir>/ctl ("key=value" lines). A daemon thread reads the ctl file each 0.25 s
and writes one "ctl" record with all keys each time the file changes.
"""

import atexit
import ctypes
import hashlib
import os
import random
import resource
import sys
import threading
import time

import numpy as np
import torch

_PAGE_SHIFT = 12
_PAGE = 1 << _PAGE_SHIFT
_now = time.perf_counter_ns
now = _now

TRACE_DIR = os.environ.get("VLLM_PLE_IO_TRACE_DIR", "").strip()

KEYS = ("mode", "backend", "threads", "defer", "fast", "check", "trace", "res")
DEFAULTS = {
    "mode": "fadvise",
    "backend": "c",
    "threads": "4",
    "defer": "0",
    "fast": "0",
    "check": "0",
    "trace": "1" if TRACE_DIR else "0",
    "res": "128",
}
_VALID = {
    "mode": lambda v: v in ("fadvise", "none", "batch"),
    "backend": lambda v: v in ("c", "pm"),
    "threads": lambda v: v.isdigit() and 1 <= int(v) <= 64,
    "defer": lambda v: v in ("0", "1"),
    "fast": lambda v: v in ("0", "1"),
    "check": lambda v: v in ("0", "1"),
    "trace": lambda v: v in ("0", "1"),
    "res": lambda v: v.isdigit(),
}
# Values from the ctl file. The ctl thread replaces the whole dict.
CTL: dict[str, str] = {}
CTL_ID = ""


def _env(key: str) -> str:
    val = os.environ.get("VLLM_PLE_IO_" + key.upper(), "").strip()
    if not val and key == "res":  # the name before rev 2
        val = os.environ.get("VLLM_PLE_IO_TRACE_RES", "").strip()
    return val


def ctl_get(key: str, default: str | None = None) -> str:
    """The current value of a control key (ctl file > env > default)."""
    val = CTL.get(key)
    if val is not None:
        return val
    val = _env(key)
    if val and (key not in _VALID or _VALID[key](val)):
        return val
    return DEFAULTS.get(key, "") if default is None else default


# Hot-path copies of the keys. _apply() sets them.
MODE = BACKEND = ""
THREADS = DEFER = FAST = CHECK = TRACE_RES = 0
TRACE = False


def _apply() -> None:
    global MODE, BACKEND, THREADS, DEFER, FAST, CHECK, TRACE, TRACE_RES
    MODE = ctl_get("mode")
    BACKEND = ctl_get("backend")
    THREADS = int(ctl_get("threads"))
    DEFER = int(ctl_get("defer"))
    FAST = int(ctl_get("fast"))
    CHECK = int(ctl_get("check")) if TRACE_DIR else 0
    TRACE = bool(TRACE_DIR) and ctl_get("trace") == "1"
    TRACE_RES = int(ctl_get("res"))


def state() -> dict[str, str]:
    return {k: ctl_get(k) for k in KEYS}


_apply()


# ---------------------------------------------------------------------------
# Trace files
# ---------------------------------------------------------------------------
class _TraceFile:
    """Buffered TSV records. The daemon thread writes the buffer each 0.25 s."""

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


def trace(kind: str, header: str) -> _TraceFile:
    """The trace file of one record kind in this process (made on first use)."""
    t = _traces.get(kind)
    if t is None:
        with _trace_lock:
            t = _traces.get(kind)
            if t is None:
                t = _traces[kind] = _TraceFile(kind, header)
    return t


_trace = trace  # the name before rev 2

_GATHER_HDR = ("t_ns mode rows uniq_rows pages res_checked res_hit "
               "pages_ns trace_ns advise_ns select_ns inblock_bytes_advise "
               "inblock_bytes_select majflt minflt backend threads")
_REQ_HDR = "t_ns num_tokens num_reqs forward_ns copy_ns total_ns seq fast"
_LAUNCH_HDR = ("t_ns num_tokens num_reqs send_ns wait_ns total_ns seq defer "
               "t_wait_start t_done gap_us")
_CTL_HDR = "t_ns " + " ".join(KEYS) + " id"
_CHECK_HDR = "t_ns seq side layer num_tokens width digest"
_NOTE_HDR = "t_ns what detail"

_ctl_seen: tuple = ()


def _read_ctl() -> bool:
    """Read <TRACE_DIR>/ctl if it changed. Return True when it was applied."""
    global CTL, CTL_ID, _ctl_seen
    path = os.path.join(TRACE_DIR, "ctl")
    try:
        st = os.stat(path)
    except OSError:
        return False
    sig = (st.st_mtime_ns, st.st_size, st.st_ino)
    if sig == _ctl_seen:
        return False
    try:
        text = open(path).read()
    except OSError:
        return False
    if text and not text.endswith("\n"):
        return False  # a partial write: read it again next time
    _ctl_seen = sig
    new: dict[str, str] = {}
    cid = ""
    for tok in text.split():
        key, _, val = tok.partition("=")
        if key == "id":
            cid = val
        elif key in _VALID and _VALID[key](val):
            new[key] = val
    CTL, CTL_ID = new, cid
    _apply()
    cur = state()
    trace("ctl", _CTL_HDR).add(_now(), *(cur[k] for k in KEYS), cid or "-")
    return True


def note(what: str, detail: str) -> None:
    """One diagnostic record (trace dir only) and one stderr line."""
    print(f"ple_io[{os.getpid()}]: {what}: {detail}", file=sys.stderr, flush=True)
    if TRACE_DIR:
        trace("note", _NOTE_HDR).add(_now(), what, detail.replace("\t", " "))


def _flush_loop() -> None:
    """Each 0.25 s: apply the ctl file, then write every trace buffer."""
    while True:
        time.sleep(0.25)
        try:
            _read_ctl()
        except Exception as exc:  # never stop the thread
            print(f"ple_io: ctl read failed: {exc}", file=sys.stderr)
        for t in list(_traces.values()):
            t.flush()


# ---------------------------------------------------------------------------
# Prefetch backends
# ---------------------------------------------------------------------------
_libc = ctypes.CDLL(None, use_errno=True)
_libc.syscall.restype = ctypes.c_long
_libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p)

_SYS_PIDFD_OPEN = 434  # asm-generic numbers (aarch64 and x86_64 agree here)
_SYS_PROCESS_MADVISE = 440
_MADV_WILLNEED = 3
_IOV_MAX = 1024
_POOL_MIN_PAGES = 4096


def _mincore_hits(base: int, pages, limit: int) -> tuple[int, int]:
    """Count the resident pages in a random sample of ``pages``."""
    if limit <= 0 or len(pages) == 0:
        return 0, 0
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


def _advise_safe(fd, ids: torch.Tensor, row_width: int) -> None:
    try:
        _advise(fd, _pages(ids, row_width))
    except Exception:  # advisory prefetch only
        pass


def _np_pages(ids_np: np.ndarray, row_width: int) -> np.ndarray:
    """Unique sorted page numbers of the rows (numpy, for the pm backend)."""
    off = ids_np * row_width
    p0 = off >> _PAGE_SHIFT
    p1 = (off + (row_width - 1)) >> _PAGE_SHIFT
    cross = p1 != p0
    if cross.any():
        p0 = np.concatenate((p0, p1[cross]))
    return np.unique(p0)


class _Native:
    """ctypes view of libple_io_native.so. ctypes releases the GIL per call."""

    def __init__(self) -> None:
        self.lib = None
        self.path = ""
        self.error = ""
        here = os.path.dirname(os.path.abspath(__file__))
        paths = [os.environ.get("VLLM_PLE_IO_NATIVE", "").strip(),
                 "/opt/ple_io/libple_io_native.so",
                 os.path.join(here, "build", "libple_io_native.so")]
        for path in paths:
            if path and os.path.exists(path):
                try:
                    lib = ctypes.CDLL(path)
                    lib.ple_prefetch.restype = ctypes.c_int64
                    lib.ple_prefetch.argtypes = (
                        ctypes.c_int, ctypes.c_void_p, ctypes.c_int64,
                        ctypes.c_int64, ctypes.c_int64, ctypes.c_int)
                    self.lib, self.path = lib, path
                    return
                except (OSError, AttributeError) as exc:
                    self.error = f"{path}: {exc}"
        self.error = self.error or "libple_io_native.so not found"


_native: _Native | None = None


def _native_lib() -> _Native:
    global _native
    if _native is None:
        _native = _Native()
        if _native.lib is None:
            note("native", "absent: " + _native.error)
        else:
            note("native", "loaded " + _native.path)
    return _native


class _PM:
    """process_madvise(MADV_WILLNEED) over the table mapping of this process."""

    def __init__(self) -> None:
        self.pidfd = -1
        self.error = ""
        self.iov = None  # uint64 [n, 2]: base, len
        self.pool = None
        self.pool_n = 0
        self.probed = False

    def probe(self, base: int) -> bool:
        """Advise one page of the table. Log the result one time."""
        self.probed = True
        fd = _libc.syscall(_SYS_PIDFD_OPEN, os.getpid(), 0)
        if fd < 0:
            self.error = f"pidfd_open errno {ctypes.get_errno()}"
            note("pm", "probe failed: " + self.error)
            return False
        self.pidfd = int(fd)
        iov = np.array([[base, _PAGE]], dtype=np.uint64)
        r = _libc.syscall(_SYS_PROCESS_MADVISE, self.pidfd,
                          ctypes.c_void_p(iov.ctypes.data), 1, _MADV_WILLNEED, 0)
        if r < 0:
            self.error = f"process_madvise errno {ctypes.get_errno()}"
            note("pm", "probe failed: " + self.error)
            os.close(self.pidfd)
            self.pidfd = -1
            return False
        note("pm", f"probe ok ({r} B)")
        return True

    def _run(self, iov: np.ndarray, lo: int, hi: int) -> int:
        """Advise iovec entries [lo, hi) in calls of at most 1024 entries."""
        base = iov.ctypes.data
        for s in range(lo, hi, _IOV_MAX):
            cnt = min(_IOV_MAX, hi - s)
            r = _libc.syscall(_SYS_PROCESS_MADVISE, self.pidfd,
                              ctypes.c_void_p(base + s * 16), cnt,
                              _MADV_WILLNEED, 0)
            if r < 0:
                raise OSError(ctypes.get_errno(), "process_madvise")
        return hi - lo

    def prefetch(self, table: torch.Tensor, ids: torch.Tensor,
                 threads: int) -> int:
        base = table.data_ptr()
        if not self.probed:
            self.probe(base)
        if self.pidfd < 0:
            raise OSError(self.error)
        ids_np = ids.numpy()
        if ids_np.dtype != np.int64:
            ids_np = ids_np.astype(np.int64)
        pages = _np_pages(ids_np, table.shape[-1])
        n = int(pages.size)
        if self.iov is None or self.iov.shape[0] < n:
            self.iov = np.empty((max(n, 1 << 18), 2), dtype=np.uint64)
            self.iov[:, 1] = _PAGE
        iov = self.iov
        iov[:n, 0] = (pages << _PAGE_SHIFT).astype(np.uint64) + np.uint64(base)
        if threads <= 1 or n <= _POOL_MIN_PAGES:
            return self._run(iov, 0, n)
        if self.pool is None or self.pool_n != threads:
            from concurrent.futures import ThreadPoolExecutor
            if self.pool is not None:
                self.pool.shutdown(wait=False)
            self.pool = ThreadPoolExecutor(threads, thread_name_prefix="ple_io_pm")
            self.pool_n = threads
        step = -(-n // threads)
        step = -(-step // _IOV_MAX) * _IOV_MAX
        futs = [self.pool.submit(self._run, iov, lo, min(n, lo + step))
                for lo in range(0, n, step)]
        return sum(f.result() for f in futs)


_pm = _PM()
_batch_errors = 0
_batch_off = False


def _prefetch_batch(fd, table: torch.Tensor, ids: torch.Tensor) -> int:
    """One batched prefetch. Return the pages advised, or -1 after an error."""
    global _batch_errors, _batch_off
    try:
        if BACKEND == "pm":
            return _pm.prefetch(table, ids, THREADS)
        nat = _native_lib()
        if nat.lib is None:
            raise OSError(nat.error)
        if ids.dtype != torch.int64 or not ids.is_contiguous():
            ids = ids.contiguous().to(torch.int64)
        total_pages = (table.shape[0] * table.shape[-1] + _PAGE - 1) >> _PAGE_SHIFT
        r = nat.lib.ple_prefetch(int(fd), ids.data_ptr(), ids.numel(),
                                 table.shape[-1], total_pages, THREADS)
        if r < 0:
            raise OSError(-r, os.strerror(-r))
        return int(r)
    except Exception as exc:
        _batch_errors += 1
        if _batch_errors >= 3 and not _batch_off:
            _batch_off = True
            note("batch", f"3 errors, stay on the fadvise loop (last: {exc})")
        elif _batch_errors < 3:
            note("batch", f"error, fadvise loop for this gather: {exc}")
        return -1


def prefetch(fd, table: torch.Tensor, ids: torch.Tensor) -> None:
    """Tell the kernel to read the pages of ``table[ids]`` (the current mode)."""
    if fd is None or MODE == "none" or not ids.numel():
        return
    if MODE == "batch" and not _batch_off:
        if _prefetch_batch(fd, table, ids) >= 0:
            return
    _advise_safe(fd, ids, table.shape[-1])


def gather(fd, table: torch.Tensor, ids: torch.Tensor,
           out: torch.Tensor) -> None:
    """Copy ``table[ids]`` into ``out`` (a uint8 [rows, row_width] view)."""
    if not TRACE:
        prefetch(fd, table, ids)
        torch.index_select(table, 0, ids, out=out)
        return

    row_width = table.shape[-1]
    mode = MODE
    n = ids.numel()
    t0 = _now()
    pages = None
    npages = 0
    if mode == "fadvise" or (mode == "batch" and _batch_off):
        pages = _pages(ids, row_width) if n else ids[:0]
        npages = pages.numel()
    t0b = _now()
    # Trace-only work (not in the untraced path): unique rows, residency.
    uniq_rows = -1
    res_checked = res_hit = 0
    if TRACE_RES > 0:
        uniq_rows = int(torch.unique(ids).numel())
        rp = pages if pages is not None else (_pages(ids, row_width) if n else ids[:0])
        res_checked, res_hit = _mincore_hits(table.data_ptr(), rp, TRACE_RES)
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    t1 = _now()
    tag = mode
    if fd is not None and n:
        if pages is not None:
            try:
                _advise(fd, pages)
            except Exception:
                pass
            if mode == "batch":
                tag = "batch-off"
        elif mode == "batch":
            r = _prefetch_batch(fd, table, ids)
            if r < 0:
                _advise_safe(fd, ids, row_width)
                tag = "batch-err"
            else:
                npages = r
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    t2 = _now()
    torch.index_select(table, 0, ids, out=out)
    t3 = _now()
    r2 = resource.getrusage(resource.RUSAGE_SELF)
    trace("gather", _GATHER_HDR).add(
        t0, tag, n, uniq_rows, npages, res_checked,
        res_hit, t0b - t0, t1 - t0b, t2 - t1, t3 - t2,
        (r1.ru_inblock - r0.ru_inblock) * 512,
        (r2.ru_inblock - r1.ru_inblock) * 512,
        r2.ru_majflt - r0.ru_majflt, r2.ru_minflt - r0.ru_minflt,
        BACKEND if mode == "batch" else "-", THREADS if mode == "batch" else 0)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def digest(t: torch.Tensor) -> str:
    """blake2b (8-byte digest) over the bytes of a CPU tensor."""
    t = t.contiguous().reshape(-1)
    if t.dtype != torch.uint8:
        t = t.view(torch.uint8)
    return hashlib.blake2b(memoryview(t.numpy()), digest_size=8).hexdigest()


def check_cpu(seq: int, layer: str, rows: torch.Tensor) -> None:
    """Worker side of G3: the digest of the output slice that it copied."""
    trace("check", _CHECK_HDR).add(_now(), seq, "cpu", layer, rows.shape[0],
                                   rows.shape[1] * rows.element_size(),
                                   digest(rows))


def _check_width(buf: torch.Tensor) -> int:
    """Elements per token that the worker writes into a GPU output buffer.

    NVFP4 packed rows (uint8 buffer): D/2 code bytes + D/16 scale bytes.
    Other dtypes: the full row.
    """
    d = buf.shape[1]
    if buf.dtype == torch.uint8:
        return d // 2 + d // 16
    return d


def check_gpu(conn, seq: int, num_tokens: int) -> None:
    """GPU side of G3: copy each output slice to the CPU after the wait."""
    for name, layer in conn._layers.items():
        buf = layer._gpu_output_buffer
        w = _check_width(buf)
        host = buf[:num_tokens, :w].to("cpu")
        trace("check", _CHECK_HDR).add(_now(), seq, "gpu", name, num_tokens,
                                       w * buf.element_size(), digest(host))


# ---------------------------------------------------------------------------
# Launch side (GPU worker): the wait, the GPU gap and the launch record
# ---------------------------------------------------------------------------
_event_pool: list = []
_launch_q: list = []  # [fields..., event A, event B], oldest first


def _event():
    if _event_pool:
        return _event_pool.pop()
    return torch.cuda.Event(enable_timing=True)


def _drain_launches() -> None:
    """Write the launch records whose GPU gap events are complete."""
    tf = trace("launch", _LAUNCH_HDR)
    while _launch_q:
        rec = _launch_q[0]
        a, b = rec[-2], rec[-1]
        if a is not None:
            if not b.query():
                if len(_launch_q) <= 16:
                    return
                b.synchronize()
            gap = round(a.elapsed_time(b) * 1000.0, 1)
            _event_pool.extend((a, b))
        else:
            gap = -1
        _launch_q.pop(0)
        tf.add(*rec[:-2], gap)


def launch_wait(conn, seq: int, num_tokens: int, num_reqs: int,
                t0: int, t_sent: int, defer: int = 0) -> None:
    """Block until the worker publishes ``seq``, then write the records.

    With the trace on, event A goes on the current stream just before the
    host blocks and event B just after. A.elapsed_time(B) is the GPU idle
    gap of this wait. The next launch reads it, so the host never waits
    for it.
    """
    if not TRACE:
        conn._wait_done(seq)
        if CHECK:
            check_gpu(conn, seq, num_tokens)
        return
    if _launch_q:
        _drain_launches()
    a = b = stream = None
    if conn.device.type == "cuda":
        stream = torch.cuda.current_stream(conn.device)
        a, b = _event(), _event()
        a.record(stream)
    t_ws = _now()
    conn._wait_done(seq)
    t_done = _now()
    if b is not None:
        b.record(stream)
    _launch_q.append([t0, num_tokens, num_reqs, t_sent - t0, t_done - t_ws,
                      t_done - t0, seq, defer, t_ws, t_done, a, b])
    if CHECK:
        check_gpu(conn, seq, num_tokens)


def trace_request(t_start: int, t_forward: int, num_tokens: int,
                  num_reqs: int, seq: int = -1, fast: int = 0) -> None:
    """Worker side: one request, from receipt to the published flag."""
    t_end = _now()
    trace("request", _REQ_HDR).add(
        t_start, num_tokens, num_reqs, t_forward - t_start,
        t_end - t_forward, t_end - t_start, seq, fast)


if TRACE_DIR:
    _read_ctl()
    threading.Thread(target=_flush_loop, daemon=True,
                     name="ple_io_trace").start()
