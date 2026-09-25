#!/usr/bin/env python3
"""CPU tests for the L7 kernels (branch kern-l7). No GPU.

Run in the pinned image (it has torch, triton and ml_dtypes):
  docker run --rm --entrypoint python3 -e TRITON_INTERPRET=1 \
      -v <worktree>:/w:ro vllm/vllm-openai:qwen38-flash-next -m pytest -q -p no:cacheprovider /w/tests/test_kern_l7.py

Parts:
  1. skinny_mx: the swizzled scale index equals the vllm swizzle; the numpy
     quantizer twin follows the FlashInfer formula on edge blocks; the Triton
     GEMV (interpreter, FP32 dot, see DOT_F32) equals the FP64 reference of
     the dequantized operands within the FP32 sum error, for split 1/2/4,
     M = 1/7, a partial last tile, unfused and fused quantization; the fused
     in-kernel quantizer equals the numpy twin bitwise.
  2. sgate: the row dot equals the FP64 reference rounded to BF16 (within 1
     BF16 ulp), M = 1/7.
  3. l7 tiles: load, merge and restore of the module tile tables.
  4. hostalloc: collect() groups views by storage; a CPU move rebinds every
     view to the new storage with the same values, offsets and strides.
  5. offline compile: every L7 Triton kernel compiles to an sm_121a cubin
     (the GB10 target) without a GPU.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KERN = os.path.join(os.path.dirname(HERE), "files", "kern")
sys.path.insert(0, KERN)

INTERP = os.environ.get("TRITON_INTERPRET") == "1"
ml_dtypes = pytest.importorskip("ml_dtypes")
triton = pytest.importorskip("triton")

import skinny_mx as mx  # noqa: E402
import sgate  # noqa: E402
import l7  # noqa: E402
import hostalloc  # noqa: E402


def _fp8(a):
    return torch.from_numpy(np.ascontiguousarray(a).view(np.uint8)).view(torch.float8_e4m3fn)


def _rand_mx(rows, k, rng, scale_lo=118, scale_hi=130):
    """Random E4M3 codes (finite) and E8M0 scales, as numpy."""
    codes = rng.integers(0, 256, size=(rows, k), dtype=np.uint8)
    codes = np.where((codes & 0x7F) == 0x7F, codes ^ 0x01, codes)  # no NaN codes
    q = codes.view(ml_dtypes.float8_e4m3fn)
    s = rng.integers(scale_lo, scale_hi, size=(rows, k // 32), dtype=np.uint8)
    return q, s


# ------------------------------------------------------------------ 1. skinny_mx
def test_swizzle_index_matches_swizzle():
    rng = np.random.default_rng(1)
    for rows, kb in ((7, 80), (300, 20), (128, 4), (1, 192)):
        sf = rng.integers(0, 255, size=(rows, kb), dtype=np.uint8)
        flat = mx.swizzle_np(sf)
        r, c = np.meshgrid(np.arange(rows), np.arange(kb), indexing="ij")
        idx = mx.swizzled_sf_index(r, c, kb)
        assert np.array_equal(flat[idx], sf)


def test_swizzle_np_matches_vllm_function():
    try:
        spec = importlib.util.find_spec("vllm")
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        pytest.skip("vllm not installed")
    try:
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import swizzle_mxfp8_scale
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"vllm mxfp8_utils import: {e!r}")
    rng = np.random.default_rng(2)
    for rows, k in ((7, 2560), (300, 640), (16384 // 64, 6144)):
        sf = rng.integers(0, 255, size=(rows, k // 32), dtype=np.uint8)
        ref = swizzle_mxfp8_scale(torch.from_numpy(sf), M=rows, K=k).numpy()
        assert np.array_equal(ref, mx.swizzle_np(sf))


def test_quant_twin_edges():
    x = np.zeros((1, 32 * 6), np.float32)
    x[0, 0:32] = 448.0          # amax/448 = 1 -> code 127, values exact
    x[0, 32:64] = 450.0         # just above (450 is exact in BF16) -> code 128 (round up)
    x[0, 64:96] = 0.0           # zero block -> code 0; ftz rcp gives inf, 0 * inf = NaN
    x[0, 96:128] = 1e-3
    x[0, 128:160] = -3.0
    x[0, 160:192] = np.linspace(-7, 7, 32)
    xb = x.astype(ml_dtypes.bfloat16).astype(np.float32)
    q, code = mx.mx_quant_twin(xb)
    assert code[0, 0] == 127 and code[0, 1] == 128
    assert np.all(q[0, 0:32].astype(np.float32) == 448.0)
    assert code[0, 2] == 0
    deq = q.astype(np.float32).reshape(1, 6, 32) * mx.e8m0_to_f32_np(code)[:, :, None]
    blk = xb.reshape(1, 6, 32)
    for b in (0, 1, 3, 4, 5):
        rel = np.abs(deq[0, b] - blk[0, b]) / np.maximum(np.abs(blk[0, b]), 1e-30)
        assert rel.max() <= 2 ** -3, (b, rel.max())  # E4M3 has 3 mantissa bits


def _check_gemv(n, k, m, tile, fused, seed):
    rng = np.random.default_rng(seed)
    wq, ws = _rand_mx(n, k, rng)
    if fused:
        xf = (rng.standard_normal((m, k)) * np.exp(rng.uniform(-3, 3, size=(m, k // 32))).repeat(32, 1))
        xb = xf.astype(ml_dtypes.bfloat16)
        xq, xs = mx.mx_quant_twin(xb.astype(np.float32))
        x_t = torch.from_numpy(xb.view(np.int16)).view(torch.bfloat16)
    else:
        xq, xs = _rand_mx(m, k, rng)
        x_t = _fp8(xq)
    ref = mx.reference_np(xq, xs, wq, ws)
    w_t = _fp8(wq)
    wsf = torch.from_numpy(mx.swizzle_np(ws))
    xsf = torch.from_numpy(mx.swizzle_np(xs))
    kw = dict(tile=tile, fused=fused, _dot_f32=True)
    if fused:
        y = mx.skinny_mx_linear(x_t, w_t, wsf, **kw)
    else:
        y = mx.skinny_mx_linear(x_t, w_t, wsf, x_q=x_t, x_sf=xsf, **kw)
    yf = y.float().numpy().astype(np.float64)
    # bound: one BF16 rounding of the exact value, plus the FP32 sum error of K terms.
    # The interpreter converts FP32 -> BF16 by truncation (error up to 1 ulp, 2^-7
    # relative; with small integer operands the sums are exact and the error is 0),
    # so the rounding term is 1 ulp here. The GPU rounds to nearest (half an ulp).
    absdot = np.abs(mx.reference_np(np.abs(xq.astype(np.float32)).astype(ml_dtypes.float8_e4m3fn), xs,
                                    np.abs(wq.astype(np.float32)).astype(ml_dtypes.float8_e4m3fn), ws))
    bound = np.abs(ref) * 2.0 ** -7 + absdot * k * 2.0 ** -24 + 1e-30
    err = np.abs(yf - ref)
    assert np.all(err <= bound), (n, k, m, tile, fused, float((err / bound).max()))
    y2 = mx.skinny_mx_linear(x_t, w_t, wsf, **kw) if fused else \
        mx.skinny_mx_linear(x_t, w_t, wsf, x_q=x_t, x_sf=xsf, **kw)
    assert torch.equal(y.view(torch.int16), y2.view(torch.int16))
    return y


@pytest.mark.skipif(not INTERP, reason="needs TRITON_INTERPRET=1 (CPU)")
@pytest.mark.parametrize("n,k,m,tile", [
    (64, 512, 7, (32, 128, 1, 4, 1)),
    (64, 512, 1, (32, 128, 2, 4, 1)),
    (48, 640, 7, (16, 128, 4, 4, 1)),    # split 4 with a partial last split, N not a tile multiple
    (160, 384, 7, (64, 128, 1, 4, 1)),   # 2 row tiles of 128-row scale groups
    (40, 256, 16, (32, 64, 2, 4, 1)),
])
@pytest.mark.parametrize("fused", [False, True])
def test_skinny_mx_matches_reference(n, k, m, tile, fused):
    _check_gemv(n, k, m, tile, fused, seed=n * 31 + k + m + int(fused))


@pytest.mark.skipif(not INTERP, reason="needs TRITON_INTERPRET=1 (CPU)")
def test_fused_equals_unfused_on_twin_quantized_input():
    # With the same MXFP8 activation, the fused and unfused kernels give the same bits.
    rng = np.random.default_rng(7)
    n, k, m = 64, 512, 7
    wq, ws = _rand_mx(n, k, rng)
    xb = (rng.standard_normal((m, k)) * 3).astype(ml_dtypes.bfloat16)
    xq, xs = mx.mx_quant_twin(xb.astype(np.float32))
    w_t, wsf = _fp8(wq), torch.from_numpy(mx.swizzle_np(ws))
    tile = (32, 128, 2, 4, 1)
    a = mx.skinny_mx_linear(torch.from_numpy(xb.view(np.int16)).view(torch.bfloat16), w_t, wsf,
                            tile=tile, fused=True, _dot_f32=True)
    b = mx.skinny_mx_linear(_fp8(xq), w_t, wsf, x_q=_fp8(xq), x_sf=torch.from_numpy(mx.swizzle_np(xs)),
                            tile=tile, fused=False, _dot_f32=True)
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.skipif(not INTERP, reason="needs TRITON_INTERPRET=1 (CPU)")
def test_in_kernel_quantizer_equals_twin():
    rng = np.random.default_rng(11)
    m, k = 7, 1024
    xf = rng.standard_normal((m, k)) * np.exp(rng.uniform(-8, 8, size=(m, k // 32))).repeat(32, 1)
    xf[0, :32] = 0.0
    xf[1, 32:64] = 448.0
    xb = xf.astype(ml_dtypes.bfloat16)
    x_t = torch.from_numpy(xb.view(np.int16)).view(torch.bfloat16)
    q, s = mx.mx_quant_rows(x_t, bk=256)
    tq, ts = mx.mx_quant_twin(xb.astype(np.float32))
    assert np.array_equal(s.numpy(), ts)
    tqf = tq.astype(np.float32)
    qn = q.numpy()
    same = (qn == tqf) | (np.isnan(qn) & np.isnan(tqf))
    assert same.all()


# ------------------------------------------------------------------ 2. sgate
@pytest.mark.skipif(not INTERP, reason="needs TRITON_INTERPRET=1 (CPU)")
@pytest.mark.parametrize("m", [1, 7, 16])
def test_sgate_rowdot(m):
    g = torch.Generator().manual_seed(m)
    x = torch.randn(m, 2560, generator=g).to(torch.bfloat16)
    w = (torch.randn(1, 2560, generator=g) / 50).to(torch.bfloat16)
    y = sgate.rowdot(x, w)
    ref = (x.double() @ w.double().t())
    ulp = torch.pow(2.0, torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7)
    absdot = x.double().abs() @ w.double().abs().t()
    assert ((y.double() - ref).abs() <= ulp + absdot * 2560 * 2.0 ** -24).all()
    assert torch.equal(y.view(torch.int16), sgate.rowdot(x, w).view(torch.int16))


def test_sgate_usable_rules():
    sgate._ON = True
    try:
        w = torch.zeros(1, 2560, dtype=torch.bfloat16)
        x = torch.zeros(7, 2560, dtype=torch.bfloat16)
        assert not sgate._usable(x, w, None)  # CPU weight: never the kernel path in the server
    finally:
        sgate._ON = False


# ------------------------------------------------------------------ 3. l7 tiles
def test_l7_tiles_merge_and_restore(tmp_path, monkeypatch):
    import types

    fake = types.ModuleType("fake_tiles_mod")
    fake.TILES = {(1, 2): (16, 128, 1, 4, 3)}
    monkeypatch.setitem(sys.modules, "fake_tiles_mod", fake)
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"fake_tiles_mod": {"3,4": [32, 256, 4, 4, 3], "1,2": [64, 128, 2, 8, 3]},
                             "_note": "ignored"}))
    table = l7.load_table(str(p))
    l7._STATE.update(table=None, orig={}, on=False)
    l7.set_tiles(False, table=table)
    assert fake.TILES == {(1, 2): (16, 128, 1, 4, 3)}
    l7.set_tiles(True)
    assert fake.TILES[(3, 4)] == (32, 256, 4, 4, 3) and fake.TILES[(1, 2)] == (64, 128, 2, 8, 3)
    l7.set_tiles(False)
    assert fake.TILES == {(1, 2): (16, 128, 1, 4, 3)}
    bad = tmp_path / "b.json"
    bad.write_text(json.dumps({"fake_tiles_mod": {"3,4": [32, 250, 4, 4, 3]}}))
    with pytest.raises(ValueError):
        l7.load_table(str(bad))


# ------------------------------------------------------------------ 4. hostalloc
def test_hostalloc_collect_and_cpu_move():
    import contextlib

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            base = torch.arange(3 << 18, dtype=torch.float32)  # 3 MiB
            self.a = torch.nn.Parameter(base[: 1 << 18].view(512, 512), requires_grad=False)
            self.b = torch.nn.Parameter(base[1 << 18:].view(1024, 512), requires_grad=False)
            self.w8 = torch.arange(1 << 21, dtype=torch.int16).view(2048, 1024)  # a plain attribute
            self.small = torch.nn.Parameter(torch.ones(10), requires_grad=False)
            self.register_buffer("buf", torch.full((1 << 19,), 2.0))

    m = M()
    groups, sizes = hostalloc.collect(m, device_type="cpu")
    assert len(groups) == 3  # a+b share one storage; w8; buf. small is below MIN_BYTES
    assert sorted(len(v) for v in groups.values()) == [1, 1, 2]
    before = {k: v.clone() for k, v in (("a", m.a), ("b", m.b), ("w8", m.w8), ("buf", m.buf))}
    old_ptr = m.a.untyped_storage().data_ptr()
    res = hostalloc.move_weights(m, to="host", _pool_ctx=contextlib.nullcontext)
    assert res["storages"] == 3
    assert m.a.untyped_storage().data_ptr() != old_ptr
    assert m.a.untyped_storage().data_ptr() == m.b.untyped_storage().data_ptr()
    assert m.b.storage_offset() == 1 << 18 and m.b.stride() == (512, 1)
    for name, t in before.items():
        assert torch.equal(getattr(m, name), t), name
    assert isinstance(m.a, torch.nn.Parameter)
    res2 = hostalloc.move_weights(m, to="device", _pool_ctx=contextlib.nullcontext)
    assert res2["storages"] == 3
    for name, t in before.items():
        assert torch.equal(getattr(m, name), t), name
    hostalloc._MOVED.clear()


def test_hostalloc_dense_scope_and_floor():
    """kern3-h1 fix: "dense" skips the routed-expert modules, and a move to the
    pool stops at the MemAvailable line; a move back returns what moved."""
    import contextlib

    class FusedMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w13_weight = torch.nn.Parameter(torch.ones(1 << 19), requires_grad=False)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = FusedMoE()
            self.proj = torch.nn.Parameter(torch.full((1 << 19,), 3.0), requires_grad=False)
            self.o = torch.nn.Parameter(torch.full((1 << 19,), 4.0), requires_grad=False)

    m = Layer()
    g_all, _ = hostalloc.collect(m, device_type="cpu")
    g_dense, _ = hostalloc.collect(m, device_type="cpu", scope="dense")
    assert len(g_all) == 3 and len(g_dense) == 2
    exp_ptr = m.experts.w13_weight.untyped_storage().data_ptr()
    res = hostalloc.move_weights(m, to="host", _pool_ctx=contextlib.nullcontext, scope="dense",
                                 _mem=lambda: 100.0)
    assert res["storages"] == 2 and res["aborted"] is None
    assert m.experts.w13_weight.untyped_storage().data_ptr() == exp_ptr
    back = hostalloc.move_weights(m, to="device", _pool_ctx=contextlib.nullcontext)
    assert back["storages"] == 2
    hostalloc._MOVED.clear()
    # MemAvailable falls 2 GiB per storage from 20: the line is max(12, 17) = 17,
    # so the second storage (18 GiB) moves and the third (16 GiB) does not.
    mem = iter([20.0, 20.0, 18.0, 16.0])
    res = hostalloc.move_weights(m, to="host", _pool_ctx=contextlib.nullcontext,
                                 _mem=lambda: next(mem))
    assert res["aborted"] and res["storages"] == 2, res
    back = hostalloc.move_weights(m, to="device", _pool_ctx=contextlib.nullcontext)
    assert back["storages"] == 2
    assert torch.equal(m.proj, torch.full((1 << 19,), 3.0))
    assert torch.equal(m.o, torch.full((1 << 19,), 4.0))
    hostalloc._MOVED.clear()


def test_inlaunch_l7_reads_nested_call_result(monkeypatch, tmp_path):
    """kern3-h1 fix: the freed-share and abort checks read res["done"]."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools", "l7"))
    inl = pytest.importorskip("inlaunch_l7")
    key = "call:vllm.models.qwen3_8_flash_next.nvidia.hostalloc:kd_call"
    for val, want in (({"to": "host", "bytes": 10, "freed_share": 0.9, "aborted": None}, "freed share"),
                      ({"to": "host", "bytes": 10, "freed_share": 1.0, "aborted": "x"}, "aborted"),
                      ({"to": "host", "bytes": 10, "freed_share": 0.99, "aborted": None}, None)):
        monkeypatch.setattr(inl, "_orig_rpc", lambda *a, _v=val, **k: {"done": {key: _v}, "info": {}})
        monkeypatch.setattr(inl, "mem_avail_gib", lambda: 50.0)
        knobs = json.dumps({"call": {key[5:]: "host:dense"}})
        if want is None:
            inl.rpc(8888, "kd_set", knobs)
        else:
            with pytest.raises(RuntimeError, match=want):
                inl.rpc(8888, "kd_set", knobs)


def test_hostalloc_so_exports():
    so = os.path.join(KERN, "hostalloc", "kern_hostalloc.so")
    if not os.path.exists(so):
        pytest.skip("kern_hostalloc.so not built (files/kern/hostalloc/build.sh)")
    import ctypes

    try:
        lib = ctypes.CDLL(so)
    except OSError as e:
        pytest.skip(f"cannot load without libcuda: {e}")
    for sym in ("kern_host_malloc", "kern_host_free", "kern_host_stats"):
        assert hasattr(lib, sym)


# ------------------------------------------------------------------ 5. offline compile
def _compile(fn, sig, consts, warps=4):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    src = ASTSource(fn=fn, signature=sig, constexprs=consts)
    return triton.compile(src, target=GPUTarget("cuda", 121, 32), options={"num_warps": warps})


@pytest.mark.skipif(INTERP, reason="offline compile needs TRITON_INTERPRET unset")
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("bn,bk,split", [(16, 256, 1), (64, 256, 4), (128, 128, 1), (32, 128, 8)])
def test_compile_skinny_mx_sm121(fused, bn, bk, split):
    sig = {"X": "*bf16" if fused else "*fp8e4nv", "XS": "*u8", "W": "*fp8e4nv", "WS": "*u8", "O": "*bf16",
           "P": "*fp32", "M": "i32", "N": "i32", "K": "i32", "stride_xm": "i32", "K_PER": "i32",
           "KB_X": "i32", "KB_W": "i32", "RCP448": "fp32", "BM": "constexpr", "BN": "constexpr",
           "BK": "constexpr", "SPLIT": "constexpr", "STAGES": "constexpr", "FUSED": "constexpr",
           "DOT_F32": "constexpr"}
    k = _compile(mx._skinny_mx_kernel, sig, {"BM": 16, "BN": bn, "BK": bk, "SPLIT": split, "STAGES": 3,
                                             "FUSED": fused, "DOT_F32": False})
    assert "mma.sync.aligned" in k.asm["ptx"] and len(k.asm["cubin"]) > 1000


@pytest.mark.skipif(INTERP, reason="offline compile needs TRITON_INTERPRET unset")
def test_compile_other_l7_kernels_sm121():
    _compile(mx._splitk_reduce, {"P": "*fp32", "O": "*bf16", "M": "i32", "N": "i32", "SPLIT": "constexpr",
                                 "BLOCK": "constexpr"}, {"SPLIT": 4, "BLOCK": 1024})
    _compile(mx._rcp_approx_kernel, {"X": "*fp32", "Y": "*fp32"}, {})
    _compile(mx._mx_quant_debug_kernel, {"X": "*bf16", "Q": "*fp32", "S": "*u8", "M": "i32", "K": "i32",
                                         "stride_xm": "i32", "RCP448": "fp32", "BM": "constexpr",
                                         "BK": "constexpr"}, {"BM": 16, "BK": 256})
    _compile(sgate._rowdot_kernel, {"X": "*bf16", "W": "*bf16", "O": "*bf16", "M": "i32", "K": "i32",
                                    "stride_xm": "i32", "BM": "constexpr", "BK": "constexpr"},
             {"BM": 8, "BK": 512})


# ------------------------------------------------------------------ 6. micro tools (CPU parts)
def test_make_tiles_uses_only_passing_sections(tmp_path):
    tools = os.path.join(os.path.dirname(HERE), "tools", "l7")
    sys.path.insert(0, tools)
    import make_tiles

    (tmp_path / "r6t.json").write_text(json.dumps({"gate": {"pass": True}, "tiles": {"512,2560": [16, 256, 4, 4, 3]}}))
    (tmp_path / "s2.json").write_text(json.dumps({"gate": {"pass": False}, "tiles": {"2560,6144": [64, 256, 1, 8, 3]}}))
    (tmp_path / "s1.json").write_text(json.dumps({"gate": {"S1_pass": True}, "shapes": {
        "16384x2560": {"m": {"7": {"best_unfused": [16, 256, 1, 4, 3]}}}}}))
    make_tiles.main(str(tmp_path))
    t = json.load(open(tmp_path / "l7_tiles.json"))
    assert t["skinny_bf16"] == {"512,2560": [16, 256, 4, 4, 3]}
    assert "mtp_w8a16" not in t and t["_note"]["s2"].startswith("gate failed")
    assert t["skinny_mx"] == {"16384,2560": [16, 256, 1, 4, 3]}
    table = l7.load_table(str(tmp_path / "l7_tiles.json"))
    assert table["skinny_mx"][(16384, 2560)] == (16, 256, 1, 4, 3)


def test_gen_kern_l7_hook_is_guarded(tmp_path):
    src = os.environ.get("VLLM_SRC", "/models/usman/qwen38-tune/vllm-src/vllm")
    path = os.path.join(src, "models/qwen3_8_flash_next/nvidia/model.py")
    if not os.path.exists(path):
        pytest.skip("no vllm source tree")
    gk = importlib.util.spec_from_file_location("gen_kern", os.path.join(KERN, "gen_kern.py"))
    mod = importlib.util.module_from_spec(gk)
    gk.loader.exec_module(mod)
    out = mod.patch_model(open(path).read())
    assert 'if os.environ.get("VLLM_KERN_L7", "0") == "1":' in out
    assert "from .l7 import enable_l7" in out
    assert all(f in mod.COPIES for f in ("l7.py", "skinny_mx.py", "sgate.py", "hostalloc.py"))


@pytest.mark.skipif(INTERP, reason="offline compile needs TRITON_INTERPRET unset")
def test_compile_hc12_kernel_sm121():
    """Item 3 reuses the fx-hc12 kernel (tools/hc12/hc12_gemv.py); it must build for sm_121a."""
    h = os.environ.get("HC12_TOOLS", "/models/usman/qwen38-flash-wt/fx-hc12/tools/hc12")
    if not os.path.exists(os.path.join(h, "hc12_gemv.py")):
        pytest.skip("fx-hc12 tools not mounted (HC12_TOOLS)")
    sys.path.insert(0, h)
    import hc12_gemv as g

    sig = {"X": "*bf16", "SM": "*u8", "EC": "*u8", "BASE": "*u8", "EOFF": "*i32", "ESC": "*u8", "O": "*bf16",
           "P": "*fp32", "M": "i32", "N": "i32", "N_OUT": "i32", "K": "i32", "stride_xm": "i32", "K_PER": "i32",
           "TGK": "i32", "NT": "i32", "TOTAL": "i32", "BM": "constexpr", "BN": "constexpr", "BK": "constexpr",
           "SPLIT": "constexpr", "STAGES": "constexpr", "FAST": "constexpr"}
    for bn, bk, fast in ((32, 256, True), (16, 128, False)):
        k = _compile(g._hc12_kernel, sig, {"BM": 16, "BN": bn, "BK": bk, "SPLIT": 4, "STAGES": 3, "FAST": fast})
        assert len(k.asm["cubin"]) > 1000
