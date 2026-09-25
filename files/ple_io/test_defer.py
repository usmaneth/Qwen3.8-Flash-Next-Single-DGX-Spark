#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU test for B2 (ple_io_defer.py) with stand-ins for vLLM.

    python3 files/ple_io/test_defer.py

The test loads ple_io.py and ple_io_defer.py with stub vllm modules and a
stand-in connector, then checks the order of the waits:
  - defer=0: launch() waits at once.
  - defer=1: launch() returns without the wait. The replay wrappers
    (run_fullgraph, run_pw_graph), the eager placeholder path and
    signal_dummy_outputs wait before they run.
  - The replay guard on the CUDA graph class waits before a replay that
    does not go through a wrapped method, and counts it.
  - The backstops in before_launch() and after_forward() count a skipped
    wait and set the defer off for the process. With check=1 a skipped
    wait is an error.
  - install() allows the defer only for a connector with CUDA inputs, and
    only when the replay guard is in place.
It also applies the hooks to the generated files when they exist
(files/ple_offload/*.py) and compiles the result.
"""
import importlib.util
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    os.environ["VLLM_PLE_IO_TRACE_DIR"] = tempfile.mkdtemp(prefix="ple_io_defer_")
    for name in ("vllm", "vllm.v1", "vllm.v1.ple_offload", "vllm.v1.worker",
                 "vllm.v1.worker.gpu"):
        sys.modules.setdefault(name, types.ModuleType(name))
    ple_io = load("vllm.v1.ple_offload.ple_io", os.path.join(HERE, "ple_io.py"))
    sys.modules["vllm.v1.ple_offload"].ple_io = ple_io

    log: list[str] = []
    cgu = types.ModuleType("vllm.v1.worker.gpu.cudagraph_utils")

    class CudaGraphManager:
        def run_fullgraph(self, desc):
            log.append("replay-full")

        def run_pw_graph(self, model, inputs):
            log.append("replay-pw")

    class ModelCudaGraphManager(CudaGraphManager):
        def run_fullgraph(self, desc):
            super().run_fullgraph(desc)
            log.append("model-full")

    cgu.CudaGraphManager = CudaGraphManager
    cgu.ModelCudaGraphManager = ModelCudaGraphManager
    sys.modules[cgu.__name__] = cgu
    d = load("vllm.v1.ple_offload.ple_io_defer", os.path.join(HERE, "ple_io_defer.py"))

    class StubGraph:  # stands in for torch.cuda.CUDAGraph
        def replay(self):
            log.append("graph-replay")

    d._graph_class = lambda: StubGraph

    class Conn:
        device = types.SimpleNamespace(type="cpu")
        _layers: dict = {}
        _uses_cuda_inputs = True

        def _wait_done(self, seq):
            log.append(f"wait{seq}")

        def signal_dummy_outputs(self, n):
            log.append("dummy")

    fails = 0

    def expect(label, want):
        nonlocal fails
        if log != want:
            fails += 1
            print(f"FAIL {label}: {log} != {want}")
        log.clear()

    conn = Conn()
    d.install(conn)
    if not d.ALLOWED:
        fails += 1
        print("FAIL install did not allow the defer")
    ple_io.TRACE = False

    ple_io.DEFER = 0
    d.before_launch(conn)
    d.launch(conn, 1, 7, 1, 0, 0)
    log.append("forward")
    d.after_forward(conn)
    expect("defer=0", ["wait1", "forward"])

    ple_io.DEFER = 1
    d.before_launch(conn)
    d.launch(conn, 2, 7, 1, 0, 0)
    log.append("sent")
    ModelCudaGraphManager().run_fullgraph(None)  # FULL decode step
    d.after_forward(conn)
    expect("defer=1 full", ["sent", "wait2", "replay-full", "model-full"])

    d.launch(conn, 3, 8192, 1, 0, 0)
    log.append("layer0")
    # the eager placeholder: the same code that patch_defer.py adds
    _defer = sys.modules.get("vllm.v1.ple_offload.ple_io_defer")
    if _defer is not None and _defer.pending is not None:
        _defer.wait_pending()
    log.append("layer1")
    d.after_forward(conn)
    expect("defer=1 eager", ["layer0", "wait3", "layer1"])

    d.launch(conn, 4, 64, 2, 0, 0)
    CudaGraphManager().run_pw_graph(None, {})
    expect("defer=1 piecewise", ["wait4", "replay-pw"])

    d.launch(conn, 5, 7, 1, 0, 0)
    conn.signal_dummy_outputs(7)
    expect("defer=1 dummy", ["wait5", "dummy"])

    # A replay that no wrapped method covers (a new vLLM path): the guard
    # waits before the graph runs. Inside a wrapped method it does nothing.
    d.launch(conn, 50, 7, 1, 0, 0)
    StubGraph().replay()
    d.after_forward(conn)
    expect("guard", ["wait50", "graph-replay"])
    if d.GUARD != 1 or d.BACKSTOP != 0 or not d.ALLOWED:
        fails += 1
        print(f"FAIL guard count {d.GUARD} backstop {d.BACKSTOP} allowed {d.ALLOWED}")
    d.launch(conn, 51, 7, 1, 0, 0)
    CudaGraphManager().run_fullgraph(None)
    StubGraph().replay()
    expect("guard in a wrapped method", ["wait51", "replay-full", "graph-replay"])
    if d.GUARD != 1:
        fails += 1
        print(f"FAIL guard count {d.GUARD} != 1")

    d.launch(conn, 6, 7, 1, 0, 0)
    d.after_forward(conn)  # the forward took no wait
    expect("backstop release", ["wait6"])
    if d.ALLOWED:
        fails += 1
        print("FAIL the backstop did not set the defer off")
    d.launch(conn, 60, 7, 1, 0, 0)  # the defer is off: launch waits at once
    expect("defer off after a backstop", ["wait60"])
    d.ALLOWED = True
    d.launch(conn, 7, 7, 1, 0, 0)
    d.before_launch(conn)  # the next launch finds the old record
    expect("backstop launch", ["wait7"])
    if d.BACKSTOP != 2 or d.ALLOWED:
        fails += 1
        print(f"FAIL backstop count {d.BACKSTOP} != 2 or allowed {d.ALLOWED}")

    d.ALLOWED = True
    ple_io.CHECK = 1
    d.launch(conn, 8, 7, 1, 0, 0)
    try:
        d.after_forward(conn)
        fails += 1
        print("FAIL check=1 backstop did not raise")
    except RuntimeError:
        pass
    ple_io.CHECK = 0
    log.clear()
    if d.pending is not None:
        fails += 1
        print("FAIL pending not clear")

    # A connector without CUDA inputs (MRV1) never defers.
    class Conn1(Conn):
        _uses_cuda_inputs = False
    c1 = Conn1()
    d.install(c1)
    d.launch(c1, 9, 7, 1, 0, 0)
    expect("MRV1 no defer", ["wait9"])

    # A CUDA graph class without replay(): no guard, so no defer.
    class NoReplay:
        pass
    d._graph_class = lambda: NoReplay
    d.install(conn)
    if d.ALLOWED:
        fails += 1
        print("FAIL install allowed the defer without the replay guard")
    d.launch(conn, 10, 7, 1, 0, 0)
    expect("no guard no defer", ["wait10"])
    ple_io.DEFER = 0

    # The hooks on the generated files (when start.sh made them).
    files = os.path.dirname(HERE)
    orig = os.path.join(files, "ple_offload", "orig")
    if os.path.isdir(orig) and os.path.exists(os.path.join(files, "ple_layer_patched.py")):
        # Generate fresh copies in a temporary tree (the worktree stays as it is).
        tmp = tempfile.mkdtemp(prefix="ple_io_hooks_")
        tf = os.path.join(tmp, "files")
        shutil.copytree(orig, os.path.join(tf, "ple_offload", "orig"))
        shutil.copy(os.path.join(files, "patch_ple_offload.py"), tf)
        shutil.copytree(HERE, os.path.join(tf, "ple_io"),
                        ignore=shutil.ignore_patterns("build", "__pycache__"))
        shutil.copy(os.path.join(files, "ple_layer_patched.py"), tf)
        subprocess.check_call([sys.executable, os.path.join(tf, "patch_ple_offload.py")],
                              stdout=subprocess.DEVNULL)
        r = subprocess.run([sys.executable, os.path.join(tf, "ple_io", "patch_ple_io.py")],
                           capture_output=True, text=True)
        if r.returncode != 0:
            fails += 1
            print("FAIL hooks:", r.stdout, r.stderr)
        for name in ("connector.py", "ple_offload_layer.py", "worker.py"):
            q = os.path.join(tf, "ple_offload", name)
            src = open(q).read()
            if name != "worker.py" and "ple_io_defer" not in src:
                fails += 1
                print("FAIL no defer hook in", q)
            py_compile.compile(q, doraise=True)
        py_compile.compile(os.path.join(tf, "ple_layer_patched.py"), doraise=True)
        # A second run finds the marks and changes nothing.
        r2 = subprocess.run([sys.executable, os.path.join(tf, "ple_io", "patch_ple_io.py")],
                            capture_output=True, text=True)
        if r2.returncode != 0 or "hooked " in r2.stdout.replace("already hooked", ""):
            fails += 1
            print("FAIL second hook run:", r2.stdout, r2.stderr)
        print("hooks:", " | ".join(l for l in r.stdout.splitlines()))
        shutil.rmtree(tmp)
    print(f"ple_io_defer order test: {'PASS' if not fails else 'FAIL'} ({fails} failures)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
