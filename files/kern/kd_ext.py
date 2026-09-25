# SPDX-License-Identifier: Apache-2.0
"""kern-decode worker extension: runtime knobs and a per-part step timer.

vLLM loads this class with --worker-extension-cls kd_ext.KDExt and adds its
methods to the GPU worker. With VLLM_SERVER_DEV_MODE=1 the API server calls
them through POST /collective_rpc {"method": "kd_set", "args": ["<json>"]}.
The server must bind to 127.0.0.1 (the recipe default).

Purpose. A decode A/B between two launches has 3-4% launch-to-launch noise
and each launch takes 10-18 minutes. Most decode rungs change a code path
that the host picks per call (an eager kernel, a graph lookup, a flag). This
extension switches those paths while the server runs, so that the arms of
one rung alternate inside one launch (A B B A ...) on the same weights,
memory layout and GPU state.

Knobs (kd_set). Every knob keeps the math of its arm exactly as a launch
with the matching env var would run it:
  lm_fp8       R5. 1: the FP8 screen with the BF16 rescore of the target
               lm_head (lm_head_fp8.py) runs for eligible batches. 0: the
               BF16 head runs (max_rows 0 makes _usable() refuse the FP8
               path). Needs VLLM_QWEN38_LM_HEAD_FP8=1 at launch (the copy).
  draft_graph  R1. 1: the drafter replays its 1-token decode graphs. 0: the
               drafter runs the 1-token passes eager (the dispatch of an
               uncaptured manager returns CUDAGraphMode.NONE). Needs the
               sizes 1..MAX_NUM_SEQS in CUDAGRAPH_CAPTURE_SIZES at launch.
  sc_async     R2. The module flag _ASYNC_H2D of the patched
               short_conv_attn.py (VLLM_SHORTCONV_ASYNC_H2D).
  ple_gpu_wait R3a. The connector flag _GPU_WAIT_ON (VLLM_PLE_GPU_WAIT=1 at
               launch): 1 = no host wait before the forward, the GPU kernel
               in the PLE layer waits; 0 = the recipe host wait.
  mtp_w8       R4. mtp_w8a16._ON (VLLM_MTP_DENSE_W8A16=1 at launch). The
               drafter graphs bake the path in: set recapture with it.
  fused_draft  R21. speculator.use_fused_multi_step_decode: 1 = the 1-token
               draft passes 1..K-1 run as ONE decode graph (the QSA builder
               refreshes its metadata inside the graph); 0 = one graph per
               pass. Needs VLLM_QSA_FUSED_DRAFT=1 at launch. The graphs bake
               the path in: set recapture with it.
  attr         {"module:attr": value}: set a module attribute of a kern
               patch (the flags of later rungs).
  call         {"module:function": arg}: L7. Call function(models, arg)
               with an nn.ModuleList of the target and the drafter model
               (l7:kd_call switches the tile table, hostalloc:kd_call
               moves the weights). Runs before recapture.
  recapture    true: capture all CUDA graphs again (target and drafter).
               For knobs that a graph bakes in at capture time.

Timer (kd_timer). CUDA events on the current stream around the verify
graph replay, the target sample (lm_head, rejection sampler), the drafter
propose call and the start of its 1-token passes. The events resolve with
Event.query() on later calls, so the timer adds no host sync. Per decode
step it gives, on the GPU time line:
  verify   v1 - v0          (the target verify graph replay)
  g_vs     s0 - v1          (host gap before the sampler)
  sample   s1 - s0          (lm_head + logits + rejection sampler)
  g_sd     d0 - s1          (host gap before the drafter)
  draft0   dm - d0          (draft prefill pass, graph)
  draftN   d1 - dm          (the 1-token passes)
  tail     v0' - d1         (GPU idle until the next verify replay)
  step     v0' - v0         (= the sum of the parts)
Only steps with a FULL verify replay of kd_timer_tokens tokens and a
drafter call count. A step longer than max_step_ms (a request boundary) is
dropped.
"""
import collections
import dataclasses
import importlib
import json
import os
import statistics
import sys
import time

import torch

MARKS = ("v0", "v1", "s0", "s1", "d0", "dm", "d1")


class _StepTimer:
    def __init__(self, runner, tokens=7, max_step_ms=250.0, pool=64):
        self.runner = runner
        self.tokens = tokens
        self.max_step_ms = max_step_ms
        self.on = False
        self.installed = False
        self.cur = None  # dict mark -> Event for the open step
        self.cur_tokens = None
        self.pending = collections.deque()
        self.rows = []
        self.dropped = 0
        self._free = [torch.cuda.Event(enable_timing=True) for _ in range(pool * len(MARKS))]

    # event pool
    def _ev(self):
        return self._free.pop() if self._free else torch.cuda.Event(enable_timing=True)

    def _mark(self, name):
        if self.cur is None:
            return
        if name in self.cur:
            # A second sampler or drafter call before the next verify replay
            # (a prefill step): the step is not a pure decode step.
            self.cur["bad"] = True
            return
        ev = self._ev()
        ev.record()
        self.cur[name] = ev

    def _resolve(self):
        while self.pending:
            step, nxt = self.pending[0]
            if not nxt.query():
                break
            self.pending.popleft()
            bad = step.pop("bad", False)
            try:
                if bad or not all(m in step for m in MARKS):
                    self.dropped += 1
                    continue
                v0 = step["v0"]
                t = {m: v0.elapsed_time(step[m]) for m in MARKS[1:]}
                stp = v0.elapsed_time(nxt)
                if stp > self.max_step_ms:
                    self.dropped += 1
                    continue
                self.rows.append({
                    "verify": t["v1"], "g_vs": t["s0"] - t["v1"],
                    "sample": t["s1"] - t["s0"], "g_sd": t["d0"] - t["s1"],
                    "draft0": t["dm"] - t["d0"], "draftN": t["d1"] - t["dm"],
                    "tail": stp - t["d1"], "step": stp,
                })
            finally:
                for ev in step.values():
                    self._free.append(ev)

    # hooks
    def on_verify(self, desc, run):
        if not self.on:
            return run(desc)
        ntok = getattr(desc, "num_tokens", None)
        ev0 = self._ev()
        ev0.record()
        if self.cur is not None:
            # close the previous step at this verify start
            if self.cur_tokens == self.tokens:
                self.pending.append((self.cur, ev0))
            else:
                self.cur.pop("bad", None)
                for ev in self.cur.values():
                    self._free.append(ev)
        self._resolve()
        self.cur = {"v0": ev0}
        self.cur_tokens = ntok
        out = run(desc)
        self._mark("v1")
        return out

    def wrap(self):
        if self.installed:
            return
        r = self.runner
        cgm = r.cudagraph_manager
        run_full = cgm.run_fullgraph
        cgm.run_fullgraph = lambda desc, _r=run_full: self.on_verify(desc, _r)
        sample = r.sample

        def _sample(*a, _s=sample, **k):
            if self.on:
                self._mark("s0")
            out = _s(*a, **k)
            if self.on:
                self._mark("s1")
            return out

        r.sample = _sample
        spec = r.speculator
        if spec is not None:
            propose = spec.propose

            def _propose(*a, _p=propose, **k):
                if self.on:
                    self._mark("d0")
                out = _p(*a, **k)
                if self.on:
                    self._mark("d1")
                return out

            spec.propose = _propose
            begin = spec.on_multi_step_decode_begin

            def _begin(*a, _b=begin, **k):
                if self.on:
                    self._mark("dm")
                return _b(*a, **k)

            spec.on_multi_step_decode_begin = _begin
        self.installed = True

    def reset(self):
        self.rows = []
        self.dropped = 0

    def report(self):
        torch.cuda.synchronize()
        self._resolve()
        rows = self.rows
        out = {"n": len(rows), "dropped": self.dropped, "tokens": self.tokens}
        if not rows:
            return out
        for k in rows[0]:
            v = sorted(x[k] for x in rows)
            n = len(v)
            lo, hi = int(0.1 * n), max(int(0.1 * n) + 1, n - int(0.1 * n))
            out[k] = {
                "median": round(statistics.median(v), 4),
                "mean": round(statistics.fmean(v), 4),
                "tmean": round(statistics.fmean(v[lo:hi]), 4),
                "p10": round(v[int(0.1 * (n - 1))], 4),
                "p90": round(v[int(0.9 * (n - 1))], 4),
            }
        return out


def _fused_draft_built():
    return os.environ.get("VLLM_QSA_FUSED_DRAFT", "0") == "1"


def _find_fp8_head(model):
    for m in model.modules():
        qm = getattr(m, "quant_method", None)
        if type(qm).__name__ == "Qwen38Fp8LMHeadMethod":
            return qm
    return None


class KDExt:
    """Methods added to the vLLM GPU worker (prefix kd_ to avoid clashes)."""

    def _kd_runner(self):
        return self.model_runner

    def _kd_state(self):
        st = getattr(self, "_kd_st", None)
        if st is None:
            st = {"timer": None, "fp8_settings": None}
            self._kd_st = st
        return st

    def kd_ping(self):
        return {"ok": True, "pid": __import__("os").getpid(), "t": time.time()}

    def kd_info(self):
        r = self._kd_runner()
        info = {"runner": type(r).__module__ + "." + type(r).__name__}
        qm = _find_fp8_head(r.model)
        info["fp8_head"] = None if qm is None else {
            "w8": qm.w8 is not None, "max_rows": qm.settings.max_rows,
            "topk": qm.settings.topk}
        spec = getattr(r, "speculator", None)
        if spec is not None and getattr(spec, "decode_cudagraph_manager", None) is not None:
            dm = spec.decode_cudagraph_manager
            info["draft_decode_graphs"] = dm.captured_token_counts()
            info["draft_decode_on"] = dm._graphs_captured
            pm = spec.prefill_cudagraph_manager
            info["draft_prefill_graphs"] = pm.captured_token_counts()
        info["verify_graphs"] = r.cudagraph_manager.captured_token_counts()
        sc = sys.modules.get("vllm.v1.attention.backends.short_conv_attn")
        info["sc_async"] = getattr(sc, "_ASYNC_H2D", None)
        con = sys.modules.get("vllm.v1.ple_offload.connector")
        info["ple_gpu_wait"] = getattr(con, "_GPU_WAIT_ON", None) if getattr(con, "_GPU_WAIT", False) else None
        info["fused_draft"] = (getattr(spec, "use_fused_multi_step_decode", None)
                               if _fused_draft_built() else None)
        w8 = sys.modules.get("vllm.models.qwen3_8_flash_next.nvidia.mtp_w8a16")
        info["mtp_w8"] = getattr(w8, "_ON", None)
        info["mtp_w4"] = getattr(w8, "_W4_ON", None) if getattr(w8, "_BUILD_W4", False) else None
        return info

    def kd_set(self, spec_json):
        """Apply knobs from a JSON object; return kd_info() after the change.

        A knob set to 0/false whose build is absent in this launch is a
        no-op (that path cannot run); a knob set to 1 without its build
        raises.
        """
        knobs = json.loads(spec_json) if isinstance(spec_json, str) else dict(spec_json)
        absent = []
        builds = {
            "ple_gpu_wait": lambda: getattr(sys.modules.get("vllm.v1.ple_offload.connector"), "_GPU_WAIT", False),
            "mtp_w8": lambda: "vllm.models.qwen3_8_flash_next.nvidia.mtp_w8a16" in sys.modules,
            "mtp_w4": lambda: getattr(sys.modules.get("vllm.models.qwen3_8_flash_next.nvidia.mtp_w8a16"),
                                      "_BUILD_W4", False),
            "mtp_norm": lambda: os.environ.get("VLLM_MTP_FUSED_NORM", "0") == "1",
            "skinny": lambda: "vllm.models.qwen3_8_flash_next.nvidia.skinny_bf16" in sys.modules,
            "fused_draft": _fused_draft_built,
        }
        for k, built in builds.items():
            if k in knobs and not knobs[k] and not built():
                knobs.pop(k)
                absent.append(k)
        for key in list((knobs.get("attr") or {})):
            mname, aname = key.split(":")
            if not knobs["attr"][key] and not hasattr(sys.modules.get(mname), aname):
                knobs["attr"].pop(key)
                absent.append(key)
        r = self._kd_runner()
        st = self._kd_state()
        done = {}
        if "lm_fp8" in knobs:
            qm = _find_fp8_head(r.model)
            if qm is None or qm.w8 is None:
                raise RuntimeError("lm_fp8: no FP8 lm_head copy (VLLM_QWEN38_LM_HEAD_FP8=1 at launch)")
            if st["fp8_settings"] is None:
                st["fp8_settings"] = qm.settings
            on = bool(knobs["lm_fp8"])
            qm.settings = st["fp8_settings"] if on else dataclasses.replace(
                st["fp8_settings"], max_rows=0)
            done["lm_fp8"] = on
        if "draft_graph" in knobs:
            dm = r.speculator.decode_cudagraph_manager
            on = bool(knobs["draft_graph"])
            if on and not dm.graphs:
                raise RuntimeError("draft_graph: no 1-token draft graph was captured")
            dm._graphs_captured = on
            done["draft_graph"] = on
        if "sc_async" in knobs:
            mod = importlib.import_module("vllm.v1.attention.backends.short_conv_attn")
            if not hasattr(mod, "_ASYNC_H2D"):
                raise RuntimeError("sc_async: short_conv_attn.py is not the patched file")
            mod._ASYNC_H2D = bool(knobs["sc_async"])
            done["sc_async"] = mod._ASYNC_H2D
        if "ple_gpu_wait" in knobs:
            mod = sys.modules.get("vllm.v1.ple_offload.connector")
            if mod is None or not getattr(mod, "_GPU_WAIT", False):
                raise RuntimeError("ple_gpu_wait: needs VLLM_PLE_GPU_WAIT=1 at launch")
            mod._GPU_WAIT_ON = bool(knobs["ple_gpu_wait"])
            done["ple_gpu_wait"] = mod._GPU_WAIT_ON
        if "mtp_w8" in knobs:
            mod = sys.modules.get("vllm.models.qwen3_8_flash_next.nvidia.mtp_w8a16")
            if mod is None:
                raise RuntimeError("mtp_w8: needs VLLM_MTP_DENSE_W8A16=1 at launch")
            mod._ON = bool(knobs["mtp_w8"])
            done["mtp_w8"] = mod._ON
        if "mtp_norm" in knobs:
            mod = sys.modules.get("vllm.models.qwen3_8_flash_next.nvidia.mtp_w8a16")
            if mod is None or os.environ.get("VLLM_MTP_FUSED_NORM", "0") != "1":
                raise RuntimeError("mtp_norm: needs VLLM_MTP_FUSED_NORM=1 at launch")
            mod._NORM_ON = bool(knobs["mtp_norm"])
            done["mtp_norm"] = mod._NORM_ON
        if "mtp_w4" in knobs:
            mod = sys.modules.get("vllm.models.qwen3_8_flash_next.nvidia.mtp_w8a16")
            if mod is None or not mod._BUILD_W4:
                raise RuntimeError("mtp_w4: needs VLLM_MTP_DENSE_W4A16=1 at launch")
            mod._W4_ON = bool(knobs["mtp_w4"])
            done["mtp_w4"] = mod._W4_ON
        if "skinny" in knobs:
            mod = sys.modules.get("vllm.models.qwen3_8_flash_next.nvidia.skinny_bf16")
            if mod is None:
                raise RuntimeError("skinny: needs VLLM_KERN_SKINNY=1 at launch")
            mod._ON = bool(knobs["skinny"])
            done["skinny"] = mod._ON
        if "fused_draft" in knobs:
            spec = getattr(r, "speculator", None)
            if spec is None or not _fused_draft_built():
                raise RuntimeError("fused_draft: needs VLLM_QSA_FUSED_DRAFT=1 at launch")
            on = bool(knobs["fused_draft"])
            if on:
                bad = sorted({g.backend.get_name() for gs in spec.attn_groups for g in gs
                              if not g.supports_draft_decode_metadata_update})
                if bad:
                    raise RuntimeError(f"fused_draft: backends without the draft update: {bad}")
                if spec.num_speculative_steps < 2:
                    raise RuntimeError("fused_draft: needs at least 2 speculative steps")
            spec.use_fused_multi_step_decode = on
            done["fused_draft"] = on
        if "fi_tactic" in knobs:
            done["fi_tactic"] = self._kd_fi_tactic(knobs["fi_tactic"])
        for key, val in (knobs.get("attr") or {}).items():
            mname, aname = key.split(":")
            mod = importlib.import_module(mname)
            if not hasattr(mod, aname):
                raise RuntimeError(f"attr: {mname} has no attribute {aname}")
            setattr(mod, aname, val)
            done[key] = val
        for key, arg in (knobs.get("call") or {}).items():
            # L7: {"module:function": arg}; function(models, arg) gets an
            # nn.ModuleList of the target model and the drafter model.
            mname, fname = key.split(":")
            mod = importlib.import_module(mname)
            if not hasattr(mod, fname):
                raise RuntimeError(f"call: {mname} has no function {fname}")
            models = [r.model]
            dm = getattr(getattr(r, "speculator", None), "model", None)
            if isinstance(dm, torch.nn.Module) and dm is not r.model:
                models.append(dm)
            done["call:" + key] = getattr(mod, fname)(torch.nn.ModuleList(models), arg)
        if knobs.get("recapture"):
            done["recapture_s"] = self._kd_recapture()
        done["absent_off"] = absent
        return {"done": done, "info": self.kd_info()}

    def _kd_fi_tactic(self, rules):
        """R7: set FlashInfer autotuner tactics of the loaded config file.

        rules: [{"op": "mxfp8_gemm", "match": "(6144, 2560)", "max_m": 64,
        "tactic": 3}]; tactic null restores the file value. The graphs keep
        the old kernel until a recapture.
        """
        from flashinfer.autotuner import AutoTuner

        tuner = AutoTuner.get()
        st = self._kd_state()
        orig = st.setdefault("fi_orig", {})
        changed = 0
        for rule in rules or []:
            for key in list(tuner._file_configs):
                if rule["op"] not in key or rule["match"] not in key:
                    continue
                try:
                    m = int(key.split("((", 1)[1].split(",", 1)[0])
                except (IndexError, ValueError):
                    continue
                if m > rule.get("max_m", 64):
                    continue
                orig.setdefault(key, tuner._file_configs[key])
                runner, _ = tuner._file_configs[key]
                t = rule.get("tactic")
                tuner._file_configs[key] = orig[key] if t is None else (runner, t)
                changed += 1
        # live-tuning entries win over the file; drop the matching ones
        for ck in list(tuner.profiling_cache):
            if any(r["op"] in str(ck) and r["match"] in str(ck) for r in rules or []):
                tuner.profiling_cache.pop(ck, None)
        return changed

    def _kd_recapture(self):
        r = self._kd_runner()
        t0 = time.time()
        torch.cuda.synchronize()
        mgrs = [r.cudagraph_manager]
        spec = getattr(r, "speculator", None)
        if spec is not None:
            mgrs += [m for m in (getattr(spec, "prefill_cudagraph_manager", None),
                                 getattr(spec, "decode_cudagraph_manager", None)) if m is not None]
        keep_draft = None
        if spec is not None and spec.decode_cudagraph_manager is not None:
            keep_draft = spec.decode_cudagraph_manager._graphs_captured or not spec.decode_cudagraph_manager.graphs
        for m in mgrs:
            m.graphs.clear()
            m._graphs_captured = False
        torch.cuda.synchronize()
        r.capture_model()
        if keep_draft is False:
            spec.decode_cudagraph_manager._graphs_captured = False
        torch.cuda.synchronize()
        return round(time.time() - t0, 2)

    def kd_timer(self, cmd="report", tokens="7"):
        st = self._kd_state()
        if st["timer"] is None:
            st["timer"] = _StepTimer(self._kd_runner(), tokens=int(tokens))
            st["timer"].wrap()
        t = st["timer"]
        t.tokens = int(tokens)
        if cmd == "start":
            t.reset()
            t.cur = None
            t.on = True
            return {"on": True}
        if cmd == "stop":
            t.on = False
            rep = t.report()
            t.cur = None
            return rep
        if cmd == "report":
            return t.report()
        raise ValueError(cmd)
