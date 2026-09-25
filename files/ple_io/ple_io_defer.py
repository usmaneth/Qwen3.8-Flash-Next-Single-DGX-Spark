# SPDX-License-Identifier: Apache-2.0
"""B2: the deferred PLE wait on eager steps (``defer=1``).

The container mounts this file as ``vllm.v1.ple_offload.ple_io_defer``.
files/ple_io/patch_defer.py adds the hooks.

The shipped connector sends the PLE request in ``prepare_forward`` and
spins on the done flag before the runner enqueues the forward, so the GPU
is idle during the whole CPU gather. The model reads the PLE output only at
layer index 1. With ``defer=1``:

1. ``launch()`` sends the request as before (the D2H sync of the inputs
   stays where it is) and keeps (conn, seq, ...) in ``pending``. It returns
   without the wait.
2. Eager steps: the ``ple_offload_wait`` placeholder (ple_offload_layer.py)
   calls ``wait_pending()`` before the first kernel that reads the output
   buffer. The embedding and layer 0 run on the GPU during the gather.
3. Graph steps: a FULL or PIECEWISE replay does not run the Python of the
   placeholder. ``install()`` wraps ``run_fullgraph`` and ``run_pw_graph``
   so that each one calls ``wait_pending()`` first. Decode (FULL graphs)
   therefore waits at the same point as the shipped path.
4. Replay guard: ``install()`` also wraps ``torch.cuda.CUDAGraph.replay``.
   Each CUDA graph replay in the process goes through it, so a replay path
   that a later vLLM adds (a new manager or a subclass that overrides
   ``run_fullgraph``) also waits before its kernels go on the stream. When
   the guard finds a pending record, it waits, counts it (``GUARD``) and
   writes one note. The result stays exact; only the overlap is less.
   Without the guard, ``install()`` does not allow the defer.
5. Backstops: ``before_launch()`` and ``after_forward()`` (release_outputs)
   find a ``pending`` record that no wait took. That forward used the
   buffer without a wait (an eager path without the placeholder). The
   backstop waits, counts it and writes a "backstop" record. Then it sets
   the defer off for the life of the process (``ALLOWED = False``), so at
   most one forward can use the buffer without the wait. With ``check=1``
   the backstop is also an error. The note goes to stderr each time.

Order: the worker calls copy_stream.synchronize() before it writes the flag.
The host enqueues the first kernel that reads the buffer only after it sees
the flag. The next request goes out only after the D2H stream waits for the
forward of this step, so the worker never writes a buffer that the GPU reads.

Each step reads ``defer`` one time (in ``launch()``) and keeps the value
with the pending record, so a ctl change never lands inside a step.
"""

import functools

from vllm.v1.ple_offload import ple_io as _ple_io

pending = None  # (conn, seq, num_tokens, num_reqs, t0, t_sent) or None
BACKSTOP = 0
GUARD = 0  # replays that the torch-level guard had to wait for
ALLOWED = False  # install() sets it when the replay methods and the guard are wrapped
_wrapped: list[str] = []
_BACKSTOP_HDR = "t_ns where seq count"


def launch(conn, seq: int, num_tokens: int, num_reqs: int, t0: int,
           t_sent: int) -> None:
    """After the request is sent: defer the wait, or wait now."""
    global pending
    if _ple_io.DEFER and ALLOWED:
        pending = (conn, seq, num_tokens, num_reqs, t0, t_sent)
        return
    _ple_io.launch_wait(conn, seq, num_tokens, num_reqs, t0, t_sent, 0)


def wait_pending() -> None:
    """Block until the pending request is done (no-op without one)."""
    global pending
    p = pending
    if p is None:
        return
    pending = None
    _ple_io.launch_wait(*p, defer=1)


def _backstop(where: str) -> None:
    global BACKSTOP, ALLOWED
    BACKSTOP += 1
    seq = pending[1] if pending is not None else -1
    msg = (f"PLE defer: a forward ran without the deferred wait (at {where}, "
           f"seq {seq}, count {BACKSTOP})")
    if _ple_io.TRACE_DIR:
        _ple_io.trace("backstop", _BACKSTOP_HDR).add(_ple_io.now(), where, seq, BACKSTOP)
    wait_pending()
    ALLOWED = False  # fail safe: no more deferred waits in this process
    _ple_io.note("defer-backstop", msg + "; the defer is now off")
    if _ple_io.CHECK:
        raise RuntimeError(msg)


def before_launch(conn) -> None:
    """At the start of _launch: an old pending record is a backstop case."""
    if pending is not None:
        _backstop("launch")


def after_forward(conn) -> None:
    """In release_outputs (after the forward): the backstop."""
    if pending is not None:
        _backstop("release_outputs")


def _guard_wait() -> None:
    """The replay guard found a pending record: wait before the replay."""
    global GUARD
    GUARD += 1
    seq = pending[1] if pending is not None else -1
    wait_pending()
    if GUARD == 1:
        _ple_io.note("defer-guard", f"a CUDA graph replay outside the wrapped "
                     f"methods found the pending wait (seq {seq}); it waited")


def _graph_class():
    """The CUDA graph class that vLLM replays (a test replaces this)."""
    import torch
    return torch.cuda.CUDAGraph


def _wrap_guard(cls) -> bool:
    fn = cls.__dict__.get("replay")
    if fn is None:
        return False
    if getattr(fn, "_ple_io_defer", False):
        return True

    @functools.wraps(fn)
    def replay(self, *args, **kwargs):
        if pending is not None:
            _guard_wait()
        return fn(self, *args, **kwargs)

    replay._ple_io_defer = True
    setattr(cls, "replay", replay)
    _wrapped.append(f"{cls.__name__}.replay")
    return True


def _wrap(cls, name: str) -> bool:
    fn = cls.__dict__.get(name)
    if fn is None:
        return False
    if getattr(fn, "_ple_io_defer", False):
        return True

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if pending is not None:
            wait_pending()
        return fn(*args, **kwargs)

    wrapper._ple_io_defer = True
    setattr(cls, name, wrapper)
    _wrapped.append(f"{cls.__name__}.{name}")
    return True


def install(conn) -> None:
    """Wrap the graph replay methods and the dummy-output path of ``conn``.

    Defer is allowed only when both replay methods of the MRV2 cudagraph
    manager and the torch replay guard are wrapped, and the connector
    stages CUDA inputs (MRV2).
    """
    global ALLOWED
    ok = False
    try:
        from vllm.v1.worker.gpu import cudagraph_utils as cgu
        ok = _wrap(cgu.CudaGraphManager, "run_fullgraph")
        ok = _wrap(cgu.CudaGraphManager, "run_pw_graph") and ok
        mcg = getattr(cgu, "ModelCudaGraphManager", None)
        if mcg is not None and "run_fullgraph" in mcg.__dict__:
            ok = _wrap(mcg, "run_fullgraph") and ok
        if not _wrap_guard(_graph_class()):
            _ple_io.note("defer", "no replay() on the CUDA graph class: no guard")
            ok = False
    except Exception as exc:  # the defer stays off
        _ple_io.note("defer", f"install failed: {exc}")
        ok = False
    orig = conn.signal_dummy_outputs

    def signal_dummy_outputs(num_tokens: int) -> None:
        if pending is not None:
            wait_pending()
        return orig(num_tokens)

    conn.signal_dummy_outputs = signal_dummy_outputs
    ALLOWED = bool(ok and getattr(conn, "_uses_cuda_inputs", False))
    _ple_io.note("defer", f"allowed={ALLOWED} wrapped={','.join(_wrapped)}")
