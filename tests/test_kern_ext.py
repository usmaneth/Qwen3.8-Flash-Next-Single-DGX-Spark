#!/usr/bin/env python3
"""CPU tests for files/kern: the generator (gen_kern.py) and the worker
extension (kd_ext.py) with stand-in runner objects and a fake CUDA event.

    python3 -m pytest -q tests/test_kern_ext.py
"""
import importlib.util
import json
import os
import subprocess
import sys
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KERN = os.path.join(os.path.dirname(HERE), "files", "kern")
SRC = os.environ.get("VLLM_SRC", "/models/usman/qwen38-tune/vllm-src/vllm")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ generator
@pytest.fixture(scope="module")
def gen_out(tmp_path_factory):
    orig = tmp_path_factory.mktemp("orig")
    out = tmp_path_factory.mktemp("out")
    for rel, name in (("models/qwen3_8_flash_next/nvidia/model.py", "nvidia_model.py"),
                      ("v1/attention/backends/short_conv_attn.py", "short_conv_attn.py")):
        with open(os.path.join(SRC, rel)) as f, open(orig / name, "w") as g:
            g.write(f.read())
    r = subprocess.run([sys.executable, os.path.join(KERN, "gen_kern.py"), "--orig", str(orig),
                        "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return orig, out


def test_gen_writes_all(gen_out):
    _, out = gen_out
    for n in ("nvidia_model.py", "short_conv_attn.py", "lm_head_fp8.py", "kd_ext.py", "SHA256.json"):
        assert (out / n).exists(), n


def test_gen_model_knob_off_is_image(gen_out):
    orig, out = gen_out
    a = (orig / "nvidia_model.py").read_text()
    b = (out / "nvidia_model.py").read_text()
    # Only additions: "import os" and the guarded call.
    added = [ln for ln in b.splitlines() if ln not in a.splitlines()]
    assert "import os" in added
    assert any("VLLM_QWEN38_LM_HEAD_FP8" in ln for ln in added)
    assert all(ln in b.splitlines() for ln in a.splitlines())


def test_gen_short_conv_default_blocking(gen_out):
    _, out = gen_out
    s = (out / "short_conv_attn.py").read_text()
    assert '_ASYNC_H2D = os.environ.get("VLLM_SHORTCONV_ASYNC_H2D", "0") == "1"' in s
    assert "_idx_cpu.to(" not in s
    assert s.count("_index_to_device(") == 6  # def + 5 call sites


def test_gen_refuses_patched_input(gen_out, tmp_path):
    _, out = gen_out
    for n in ("nvidia_model.py", "short_conv_attn.py"):
        (tmp_path / n).write_text((out / n).read_text())
    r = subprocess.run([sys.executable, os.path.join(KERN, "gen_kern.py"), "--orig", str(tmp_path),
                        "--out", str(tmp_path / "o")], capture_output=True, text=True)
    assert r.returncode == 1 and "already patched" in r.stderr


# ------------------------------------------------------------ fake CUDA events
class FakeClock:
    t = 0.0


class FakeEvent:
    def __init__(self, enable_timing=True):
        self.t = None

    def record(self, stream=None):
        self.t = FakeClock.t

    def query(self):
        return self.t is not None

    def elapsed_time(self, other):
        return other.t - self.t


@pytest.fixture()
def kd(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    return _load("kd_ext_test", os.path.join(KERN, "kd_ext.py"))


class Desc:
    def __init__(self, n):
        self.num_tokens = n


def make_runner():
    calls = []
    cgm = types.SimpleNamespace(captured_token_counts=lambda: [7, 14])
    cgm.run_fullgraph = lambda desc: (calls.append("v"), setattr(FakeClock, "t", FakeClock.t + 50.0))

    class Spec:
        def propose(self, *a, **k):
            FakeClock.t += 2.0  # draft pass 0
            self.on_multi_step_decode_begin(1)
            FakeClock.t += 10.0  # 1-token passes
            return "drafts"

        def on_multi_step_decode_begin(self, n):
            calls.append("dm")

    spec = Spec()
    spec.decode_cudagraph_manager = types.SimpleNamespace(
        graphs={1: object()}, _graphs_captured=True, captured_token_counts=lambda: [1, 2, 3, 4])
    spec.prefill_cudagraph_manager = types.SimpleNamespace(captured_token_counts=lambda: [7])

    class Runner:
        def sample(self, *a, **k):
            FakeClock.t += 5.0
            return "sampled"

    r = Runner()
    r.cudagraph_manager = cgm
    r.speculator = spec
    r.model = torch.nn.Linear(2, 2)
    return r, calls


def step(r, n=7, host=0.5, tail=3.0):
    r.cudagraph_manager.run_fullgraph(Desc(n))
    FakeClock.t += host
    r.sample()
    FakeClock.t += host
    r.speculator.propose()
    FakeClock.t += tail


def test_timer_parts_sum_to_step(kd):
    r, _ = make_runner()
    t = kd._StepTimer(r, tokens=7)
    t.wrap()
    t.on = True
    for _ in range(10):
        step(r)
    rep = t.report()
    assert rep["n"] == 9  # the last step has no closing verify
    parts = ("verify", "g_vs", "sample", "g_sd", "draft0", "draftN", "tail")
    assert sum(rep[p]["median"] for p in parts) == pytest.approx(rep["step"]["median"])
    assert rep["verify"]["median"] == pytest.approx(50.0)
    assert rep["sample"]["median"] == pytest.approx(5.0)
    assert rep["draft0"]["median"] == pytest.approx(2.0)
    assert rep["draftN"]["median"] == pytest.approx(10.0)
    assert rep["tail"]["median"] == pytest.approx(3.0)
    assert rep["step"]["median"] == pytest.approx(71.0)


def test_timer_drops_other_widths_and_prefill_steps(kd):
    r, _ = make_runner()
    t = kd._StepTimer(r, tokens=7)
    t.wrap()
    t.on = True
    step(r, n=14)          # another width: not counted
    step(r)
    r.sample()             # a second sampler call (prefill step) in this step
    step(r)
    step(r)
    rep = t.report()
    assert rep["n"] == 1
    assert rep["dropped"] >= 1


def test_timer_off_is_passthrough(kd):
    r, calls = make_runner()
    t = kd._StepTimer(r, tokens=7)
    t.wrap()
    assert r.sample() == "sampled"
    assert r.speculator.propose() == "drafts"
    step(r)
    assert t.report()["n"] == 0


def test_kd_set_draft_graph_and_attr(kd):
    r, _ = make_runner()
    w = kd.KDExt()
    w.model_runner = r
    res = w.kd_set(json.dumps({"draft_graph": 0}))
    assert r.speculator.decode_cudagraph_manager._graphs_captured is False
    assert res["info"]["draft_decode_on"] is False
    w.kd_set(json.dumps({"draft_graph": 1}))
    assert r.speculator.decode_cudagraph_manager._graphs_captured is True
    mod = types.ModuleType("kd_fake_flags")
    mod.FLAG = False
    sys.modules["kd_fake_flags"] = mod
    w.kd_set(json.dumps({"attr": {"kd_fake_flags:FLAG": True}}))
    assert mod.FLAG is True
    with pytest.raises(RuntimeError):
        w.kd_set(json.dumps({"attr": {"kd_fake_flags:NOPE": 1}}))
    with pytest.raises(RuntimeError):
        w.kd_set(json.dumps({"lm_fp8": 1}))  # no FP8 head in the stand-in model


def test_kd_set_lm_fp8_toggles_max_rows(kd):
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class S:
        max_rows: int = 64
        topk: int = 64

    class Qwen38Fp8LMHeadMethod:
        def __init__(self):
            self.w8 = object()
            self.settings = S()

    r, _ = make_runner()
    r.model.quant_method = Qwen38Fp8LMHeadMethod()
    w = kd.KDExt()
    w.model_runner = r
    w.kd_set(json.dumps({"lm_fp8": 0}))
    assert r.model.quant_method.settings.max_rows == 0
    w.kd_set(json.dumps({"lm_fp8": 1}))
    assert r.model.quant_method.settings.max_rows == 64
