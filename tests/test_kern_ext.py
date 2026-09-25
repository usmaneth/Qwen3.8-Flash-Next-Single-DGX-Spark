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
                      ("v1/attention/backends/short_conv_attn.py", "short_conv_attn.py"),
                      ("models/qwen3_8_flash_next/common/qsa_cache.py", "qsa_cache.py")):
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
    for n in ("nvidia_model.py", "short_conv_attn.py", "qsa_cache.py"):
        (tmp_path / n).write_text((out / n).read_text())
    r = subprocess.run([sys.executable, os.path.join(KERN, "gen_kern.py"), "--orig", str(tmp_path),
                        "--out", str(tmp_path / "o")], capture_output=True, text=True)
    assert r.returncode == 1 and "already patched" in r.stderr


def test_gen_qsa_cache_default_off(gen_out):
    orig, out = gen_out
    a = (orig / "qsa_cache.py").read_text()
    b = (out / "qsa_cache.py").read_text()
    assert 'os.environ.get("VLLM_QSA_FUSED_DRAFT", "0") == "1"' in b
    assert "def update_draft_decode_metadata" in b
    # The flag-off build() makes the same build_qsa_metadata call: the same
    # keyword values, now passed through one dict.
    for kw in ("storage_block_size=self.storage_block_size,",
               "compress_ratio=self.compress_ratio,",
               "k_work_metadata_buffer=k_work_metadata if build_k_work else None,",
               "request_capacity=request_capacity,"):
        assert kw in a and kw in b
    assert b.count("build_qsa_metadata(") == a.count("build_qsa_metadata(") + 1


def test_gen_qsa_cache_builder_flag(gen_out, monkeypatch):
    """The patched builder: the flag follows the env var, build() stores the
    rebuild arguments only when the flag is on, and the update call launches
    the metadata kernel with exactly the arguments of the last build()."""
    _, out = gen_out
    src = (out / "qsa_cache.py").read_text()
    tree = __import__("ast").parse(src)
    cls = next(n for n in tree.body if getattr(n, "name", "") == "QSAMetadataBuilder")
    names = [f.name for f in cls.body if hasattr(f, "name")]
    assert "update_draft_decode_metadata" in names
    upd = next(f for f in cls.body if getattr(f, "name", "") == "update_draft_decode_metadata")
    txt = __import__("ast").unparse(upd)
    assert "self._draft_rebuild" in txt and "**rebuild_kwargs" in txt


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


# --------------------------------------------------------------- R4 (CPU part)
def test_patch_mtp_kern_idempotent(tmp_path):
    src = os.path.join(os.path.dirname(HERE), "files", "mtp_patched.py")
    if not os.path.exists(src):
        pytest.skip("files/mtp_patched.py is generated by start.sh")
    p = tmp_path / "mtp_patched.py"
    text = open(src).read()
    if "enable_mtp_w8a16" in text:
        pytest.skip("already patched in place")
    p.write_text(text)
    gen = os.path.join(KERN, "patch_mtp_kern.py")
    r1 = subprocess.run([sys.executable, gen, str(p)], capture_output=True, text=True)
    assert r1.returncode == 0 and "applied" in r1.stdout
    once = p.read_text()
    r2 = subprocess.run([sys.executable, gen, str(p)], capture_output=True, text=True)
    assert "already applied" in r2.stdout and p.read_text() == once
    assert once.count('os.environ.get("VLLM_MTP_DENSE_W8A16", "0") == "1"') == 1


def test_w8a16_quantize_and_reference():
    m = _load("mtp_w8a16_test", os.path.join(KERN, "mtp_w8a16.py"))
    torch.manual_seed(0)
    w = (torch.randn(96, 320) * 0.05).to(torch.bfloat16)
    w8, s = m.quantize_rows(w, chunk_rows=40)
    assert w8.dtype == torch.float8_e4m3fn and s.shape == (96,)
    # per-row amax maps to 448 and the dequantized weight is close to BF16
    deq = w8.float() * s[:, None]
    assert torch.allclose(deq, w.float(), rtol=0.07, atol=1e-3)
    x = torch.randn(3, 320).to(torch.bfloat16)
    ref = m.w8a16_reference(x, w8, s)
    assert ref.dtype == torch.bfloat16 and ref.shape == (3, 96)
    exact = ((x.float() @ w8.float().t()) * s[None, :]).to(torch.bfloat16)
    assert torch.equal(ref, exact)


def test_w8a16_method_falls_back():
    m = _load("mtp_w8a16_test2", os.path.join(KERN, "mtp_w8a16.py"))

    class Inner:
        def apply(self, layer, x, bias=None):
            return "bf16"

    meth = m.MtpW8A16Method(Inner(), "x")
    layer = types.SimpleNamespace(weight=torch.zeros(64, 32, dtype=torch.bfloat16))
    m._ON = True
    assert meth.apply(layer, torch.zeros(1, 32, dtype=torch.bfloat16)) == "bf16"  # no FP8 copy (CPU)
    meth.w8 = torch.zeros(64, 32, dtype=torch.float8_e4m3fn)
    meth.scale = torch.ones(64)
    assert meth.apply(layer, torch.zeros(40, 32, dtype=torch.bfloat16)) == "bf16"  # M > MAX_M
    m._ON = False
    assert meth.apply(layer, torch.zeros(1, 32, dtype=torch.bfloat16)) == "bf16"  # knob off


def test_w4_quantize_roundtrip_and_exact_bf16_weights():
    m = _load("w4a16_test", os.path.join(KERN, "w4a16.py"))
    torch.manual_seed(1)
    w = (torch.randn(48, 256) * 0.03).to(torch.bfloat16)
    w[3, :32] = 0  # an all-zero group
    packed, gs, rs = m.quantize_w4(w, chunk_rows=20)
    assert packed.shape == (48, 128) and gs.shape == (48, 8) and rs.shape == (48,)
    deq = m.dequant_w4(packed, gs, rs).view(48, 256)
    err = (deq - w.float()).abs()
    step = (rs[:, None] * gs.float()).repeat_interleave(32, dim=1)
    assert bool((err <= 0.5 * step * 1.001 + 1e-12).all())  # round-to-nearest
    # q x gs is exact in BF16 (the kernel feeds it to tl.dot as BF16)
    lo = (packed & 0xF).to(torch.int16) - 8
    hi = (packed >> 4).to(torch.int16) - 8
    q = torch.stack((lo, hi), -1).view(48, 256).float()
    wq = (q.view(48, 8, 32) * gs.float()[..., None]).view(48, 256)
    assert torch.equal(wq.to(torch.bfloat16).float(), wq)
    x = torch.randn(2, 256).to(torch.bfloat16)
    ref = m.w4a16_reference(x, packed, gs, rs)
    alt = ((x.float() @ deq.t())).to(torch.bfloat16)
    assert (ref.float() - alt.float()).abs().max() <= 0.02 * alt.float().abs().max()


def test_patch_ple_kern_applies_and_is_idempotent(tmp_path):
    src = os.path.join(os.path.dirname(HERE), "files", "ple_offload")
    names = ("ple_offload_layer.py", "connector.py", "worker.py", "protocol.py")
    if not all(os.path.exists(os.path.join(src, n)) for n in names):
        pytest.skip("files/ple_offload is generated by start.sh")
    for n in names:
        text = open(os.path.join(src, n)).read()
        if "VLLM_PLE_GPU_WAIT" in text:
            pytest.skip("already patched in place")
        (tmp_path / n).write_text(text)
    gen = os.path.join(KERN, "patch_ple_kern.py")
    r1 = subprocess.run([sys.executable, gen, str(tmp_path)], capture_output=True, text=True)
    assert r1.returncode == 0 and "applied to" in r1.stdout, r1.stdout + r1.stderr
    once = {n: (tmp_path / n).read_text() for n in names}
    r2 = subprocess.run([sys.executable, gen, str(tmp_path)], capture_output=True, text=True)
    assert "already applied" in r2.stdout
    assert all((tmp_path / n).read_text() == once[n] for n in names)
    # knob off: every change sits behind _GPU_WAIT
    assert '_GPU_WAIT = os.environ.get("VLLM_PLE_GPU_WAIT", "0") == "1"' in once["worker.py"]
    assert "staging_bufs=self._kd_make_staging() if _GPU_WAIT else None" in once["connector.py"]


def test_kd_set_off_knob_without_build_is_noop(kd):
    r, _ = make_runner()
    w = kd.KDExt()
    w.model_runner = r
    res = w.kd_set(json.dumps({"mtp_w4": 0, "skinny": 0, "ple_gpu_wait": 0, "mtp_norm": 0,
                               "attr": {"no_such_module_x:FLAG": False}}))
    assert set(res["done"]["absent_off"]) >= {"mtp_w4", "skinny", "ple_gpu_wait", "mtp_norm"}
    with pytest.raises(RuntimeError):
        w.kd_set(json.dumps({"skinny": 1}))


@pytest.mark.parametrize("plan,builds", [
    ("/models/usman/kern-decode/plans/l4-r3a2-r8-r6.json", ("w8", "head", "ple", "skinny", "norm")),
    ("/models/usman/kern-decode/plans/l3-gates-det.json", ("w8", "head", "ple", "skinny")),
])
def test_plan_arms_apply(kd, monkeypatch, plan, builds):
    """Every arm of a queued plan passes kd_set with the builds of its launch."""
    if not os.path.exists(plan):
        pytest.skip("plan not present")
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
    r.capture_model = lambda: r.speculator.decode_cudagraph_manager.graphs.update({1: object()})
    r.cudagraph_manager.graphs = {}
    r.speculator.prefill_cudagraph_manager.graphs = {}
    mods = {"vllm.v1.attention.backends.short_conv_attn": {"_ASYNC_H2D": False}}
    if "w8" in builds:
        mods["vllm.models.qwen3_8_flash_next.nvidia.mtp_w8a16"] = {"_ON": True, "_W4_ON": False, "_BUILD_W4": False, "_NORM_ON": False}
    if "head" in builds:
        mods["vllm.models.qwen3_8_flash_next.nvidia.mtp"] = {"_KERN_HEAD_W4_ON": True}
    if "ple" in builds:
        mods["vllm.v1.ple_offload.connector"] = {"_GPU_WAIT": True, "_GPU_WAIT_ON": True}
    if "skinny" in builds:
        mods["vllm.models.qwen3_8_flash_next.nvidia.skinny_bf16"] = {"_ON": True}
    for name, attrs in mods.items():
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        monkeypatch.setitem(sys.modules, name, m)
    monkeypatch.setenv("VLLM_MTP_FUSED_NORM", "1" if "norm" in builds else "0")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    w = kd.KDExt()
    w.model_runner = r
    arms = json.load(open(plan))["arms"]
    for name, knobs in arms.items():
        w.kd_set(json.dumps(knobs))


def test_kd_set_fused_draft(kd, monkeypatch):
    r, _ = make_runner()
    ok = types.SimpleNamespace(supports_draft_decode_metadata_update=True,
                               backend=types.SimpleNamespace(get_name=lambda: "QSA"))
    r.speculator.attn_groups = [[ok]]
    r.speculator.num_speculative_steps = 6
    r.speculator.use_fused_multi_step_decode = False
    w = kd.KDExt()
    w.model_runner = r
    monkeypatch.delenv("VLLM_QSA_FUSED_DRAFT", raising=False)
    # off without the build: a no-op; on without the build: an error
    assert "fused_draft" in w.kd_set(json.dumps({"fused_draft": 0}))["done"]["absent_off"]
    with pytest.raises(RuntimeError):
        w.kd_set(json.dumps({"fused_draft": 1}))
    monkeypatch.setenv("VLLM_QSA_FUSED_DRAFT", "1")
    res = w.kd_set(json.dumps({"fused_draft": 1}))
    assert r.speculator.use_fused_multi_step_decode is True
    assert res["info"]["fused_draft"] is True
    w.kd_set(json.dumps({"fused_draft": 0}))
    assert r.speculator.use_fused_multi_step_decode is False
    bad = types.SimpleNamespace(supports_draft_decode_metadata_update=False,
                                backend=types.SimpleNamespace(get_name=lambda: "OTHER"))
    r.speculator.attn_groups = [[ok, bad]]
    with pytest.raises(RuntimeError, match="OTHER"):
        w.kd_set(json.dumps({"fused_draft": 1}))
