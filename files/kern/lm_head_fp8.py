# SPDX-License-Identifier: Apache-2.0
"""FP8 screen and BF16 rescore for the Qwen3.8-Flash-Next target lm_head.

Problem (trace prof/dp0_pp0_tp0_dcp0_ep0_rank0.*, 42 decode steps at K=3):
the target lm_head (248320 x 2560, BF16, 1.27 GB) runs once per decode step
in eager mode and takes 6.42 ms (median) of a 66-70 ms step. It is the
largest single BF16 GEMM of the step (data/gemm_breakdown_225.txt).

Method. This module replaces the quant method of the target lm_head:
  1. Screen: an FP8 (e4m3, one scale per row) copy of the head, 0.64 GB,
     runs on torch._scaled_mm with rowwise scales. The MTP draft head uses
     the same kernel in the reference trace: 227 GB/s inside the draft CUDA
     graphs (530-532 us) and 215 GB/s eager (562.6 us). The target head runs
     eager, so the screen is expected at about 215 GB/s: 2.8-3.0 ms.
  2. Rescore: for each row of logits, the TOPK best screen candidates are
     computed again from the BF16 head (TOPK x 2560 x 2 bytes per row) and
     are written back into the logits.
The result has the BF16 values at the candidate positions and FP8 values at
all other positions.

Where the FP8 path runs (an allow-list). apply() uses the FP8 path only in
the sampler scope: inside the V2 GPUModelRunner.sample call
(v1/worker/gpu/model_runner.py:1384-1400), and only when batch_bf16_reason
accepts the batch. install_sampler_hook() wraps that method when the FP8
copy is built. Every other call of the head uses the BF16 head (inner):
  - prompt logprobs. model_runner.py:1828-1835 calls compute_logits after
    sample(), in chunks of up to 1024 rows (sample/prompt_logprob.py:207-223),
    and compute_topk_scores reads the logit of the real prompt token, which
    is often not a candidate. So prompt_logprobs and driftgate.py stay BF16.
  - the MTP drafter. load_eagle_model gives the drafter the target lm_head
    module (spec_decode/eagle/utils.py:83-101). speculator.propose runs
    after sample() (model_runner.py:1893), so a drafter call through the
    shared head (MTP_DRAFT_VOCAB empty, or probabilistic draft sampling)
    uses BF16, also during draft CUDA graph capture.
  - the memory profile run (_dummy_sampler_run, model_runner.py:812-815).
The batch check sends the whole batch to the BF16 head when any request of
the batch uses one of these. They change logits before top_k, so tokens
that are not candidates can enter the sampling set with FP8 values:
  - a grammar bitmask (model_runner.py:1391-1400; for example a named
    tool_choice, which the pulse router forwards)
  - logit_bias, allowed_token_ids or min_tokens, penalties, bad_words or
    min_p (sampler.py:198-244, rejection_sampler.py:146-170)
  - logprobs or logprob_token_ids (the log-softmax sums over all tokens)
  - top_k above SAMPLE_TOPK (top_k off included) with a temperature that is
    not 0. Greedy rows use only the maximum, so any top_k is accepted.
The thinking budget writes 1e9 into the forced token (sample/
thinking_budget.py:369), so the forced output does not depend on the other
values; it needs no fallback.

Correctness condition. For one row let T be the set of the 20 best BF16
logits and v20 the 20th BF16 value. The conditions are:
  (a) every token of T is a screen candidate, and
  (b) every value that is not a candidate (an FP8 value) is below v20.
(a) alone is not sufficient. When (a) and (b) hold and the rescore rounds
as F.linear does (below), the output is distributionally identical to the
BF16 head for greedy decoding and for top_k <= 20 sampling (then top_p).
It is bit-identical for greedy rows, for rows with an explicit seed, and
for the rows of the rejection sampler (rejection_sampler.py:146-170): top_k
masks every value outside the top k, and the random numbers come from
(seed, position). It is not bit-identical for unseeded rows on the FlashInfer
sampler of the path without spec decode (sampler.py:272-284): the top-k-only
branch of flashinfer_sample takes a softmax over the whole vocabulary
(topk_topp_sampler.py:500-504), and that softmax includes the FP8 values
outside the candidates. The probabilities then differ by a common factor,
so with the same random numbers another token can come out in rare cases.
exact_topk of tools/fp8head_offline_eval.py and the audit counters
topk_values and outside_ok test (a) and (b) together; the audit counter
contain tests (a) only.

Rounding. The rescore multiplies in FP32 (the products of two BF16 values
are exact), sums in FP32 and rounds once to BF16. It uses no BF16 GEMM, so
no reduced-precision reduction of cuBLAS can occur in it. The BF16 head sums
in FP32 in the cuBLAS GEMM of F.linear, in another order, so a candidate
value can differ from the BF16 head by one BF16 rounding step.
tools/fp8head_microbench.py measures that rate on the GPU against the
base-vs-base rate (F.linear at M = 4 against M = 8 and M = 1), and stops
the session when the rescore rate is more than TOPK_WARN below that floor.

Risk counter (the ship configuration, no host sync per call). For each row
the rescore also gives s_k, the smallest screen value of the candidates, and
v_s, the SAMPLE_TOPK-th best rescored value. Every screen value outside the
candidates is at most s_k. A row is at risk when s_k + RISK_MARGIN >= v_s.
Proof of the rule: a token j outside the candidates has the output value
s_j <= s_k < v_s, which is (b), and the BF16 value b_j <= s_j + e_j. When
every screen error e_j is below RISK_MARGIN, b_j < v_s, so the SAMPLE_TOPK
best BF16 tokens are all candidates, which is (a). So a row that is not at
risk satisfies (a) and (b) when the true screen error is below the margin.
The count of rows at risk is a device tensor; one log line every RISK_EVERY
FP8 calls reads it (one .item() per RISK_EVERY calls). The audit counter
missed counts the rows that fail (a) or (b) and are not at risk: then
RISK_MARGIN is too small. The share of rows at risk is a baseline, not an
alarm: on 4089 real completion rows (tools/fp8head_offline_eval.py, data/
fp8head_offline_risk_s0.json and _s1.json) the gap v20 - s_64 has a median
of 2.1 and a 1st percentile of 1.1-1.2 logits, so at TOPK 64 about 11% of
the rows are at risk at margin 1.5 and 44% at margin 2.0; at TOPK 128 0.1%
and 2%. No row fails (a) or (b) at any TOPK >= 32, so these rows cannot
test the margin itself; the audit arm and the microbench count missed rows.

The FP8 copy is made once in process_weights_after_loading, in chunks of
BUILD_ROWS rows, so the build needs about 0.1 GB of temporary memory and
never runs during CUDA graph capture. The BF16 head stays in memory for the
rescore, so the model uses 0.64 GB more memory.

Environment (read once, when the target model is built):
  VLLM_QWEN38_LM_HEAD_FP8=1             enable (model.py calls
                                        enable_fp8_lm_head() only then)
  VLLM_QWEN38_LM_HEAD_FP8_TOPK=64       candidates per row that get the
                                        BF16 rescore. 0 = screen only (the
                                        numerics of files/ours/
                                        patch_lm_head_fp8.py), for bisection
  VLLM_QWEN38_LM_HEAD_FP8_SAMPLE_TOPK=20
                                        the largest top_k of a sampled
                                        request that keeps the FP8 path.
                                        The offline eval measured the
                                        containment of the top 20 only
  VLLM_QWEN38_LM_HEAD_FP8_MAX_ROWS=64   rows above this use the BF16 head.
                                        A decode batch has MAX_NUM_SEQS x
                                        (1 + K) = 16 rows at most
  VLLM_QWEN38_LM_HEAD_FP8_CHUNK=128     rows per rescore chunk. The gather
                                        holds CHUNK x TOPK x 2560 values;
                                        CHUNK x TOPK must be at most
                                        MAX_GATHER_ROWS (84 MB in BF16 and
                                        168 MB in FP32). At the decode size
                                        (16 x 64 rows) that is 5 + 10 MB
  VLLM_QWEN38_LM_HEAD_FP8_AUDIT=0       N > 0: each FP8 call with at most
                                        AUDIT_MAX_ROWS rows also runs the
                                        BF16 head and compares. One log line
                                        every N calls. It adds the BF16 time,
                                        so use it only in an audit arm. The
                                        rows of batches with a sampled
                                        request (temperature not 0) are also
                                        counted on their own (sampled_*)
  VLLM_QWEN38_LM_HEAD_FP8_AUDIT_K=20    the top-k that the audit compares
  VLLM_QWEN38_LM_HEAD_FP8_RISK_EVERY=1000
                                        one risk log line every N FP8 calls
                                        (RISK_LOG). 0 = no risk counter
  VLLM_QWEN38_LM_HEAD_FP8_RISK_MARGIN=1.5
                                        the margin of the risk counter, in
                                        logits. The rule needs a margin above
                                        the screen error; 1.5 is 1.6 x the
                                        largest screen error of the offline
                                        eval (0.94 over all tokens of 4089
                                        rows). The audit arm and the
                                        microbench check it (missed)
  VLLM_QWEN38_LM_HEAD_FP8_ALTERNATE=0   1: a timing arm only. Every second
                                        FP8 call uses the BF16 head, so one
                                        profiled request gives BF16 steps
                                        and FP8 steps in the same process
                                        (tools/headstat.py, alternate). Both
                                        paths give the same argmax when (a)
                                        and (b) hold and the rescore rounds
                                        as F.linear; the audit top1 rate
                                        measures that
"""

import contextlib
import functools
import os
import sys
import threading
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

logger = init_logger(__name__)

ENV = "VLLM_QWEN38_LM_HEAD_FP8"
TOPK_ENV = "VLLM_QWEN38_LM_HEAD_FP8_TOPK"
SAMPLE_TOPK_ENV = "VLLM_QWEN38_LM_HEAD_FP8_SAMPLE_TOPK"
MAX_ROWS_ENV = "VLLM_QWEN38_LM_HEAD_FP8_MAX_ROWS"
CHUNK_ENV = "VLLM_QWEN38_LM_HEAD_FP8_CHUNK"
AUDIT_ENV = "VLLM_QWEN38_LM_HEAD_FP8_AUDIT"
AUDIT_K_ENV = "VLLM_QWEN38_LM_HEAD_FP8_AUDIT_K"
ALTERNATE_ENV = "VLLM_QWEN38_LM_HEAD_FP8_ALTERNATE"
RISK_EVERY_ENV = "VLLM_QWEN38_LM_HEAD_FP8_RISK_EVERY"
RISK_MARGIN_ENV = "VLLM_QWEN38_LM_HEAD_FP8_RISK_MARGIN"

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0
# torch._scaled_mm needs N % 16 == 0. The pad rows are zero.
ROW_ALIGN = 16
# Rows per step of the FP8 build: 8192 x 2560 FP32 values = 84 MB.
BUILD_ROWS = 8192
# The largest CHUNK x TOPK: 16384 x 2560 BF16 values = 84 MB per gather.
MAX_GATHER_ROWS = 16384
# The audit runs only on decode-sized calls.
AUDIT_MAX_ROWS = 64
# The log line of a successful build. ab/run_arm.sh looks for it.
BUILD_LOG = "Qwen3.8 FP8 lm_head: screen"
AUDIT_LOG = "Qwen3.8 FP8 lm_head audit:"
RISK_LOG = "Qwen3.8 FP8 lm_head risk:"
# The integer counters of the audit, in log order.
_ROW_COUNTS = ("rows", "top1", "topk_values", "contain", "outside_ok")
AUDIT_COUNTS = (
    _ROW_COUNTS
    + ("screen_top1", "at_risk", "missed")
    + tuple("sampled_" + name for name in _ROW_COUNTS)
)

# The sampler scope. The V2 model runner of the pinned image (vLLM
# v1/worker/gpu/model_runner.py). gpu_worker.py:529 imports it in
# init_device, before load_model builds the model.
RUNNER_MODULE = "vllm.v1.worker.gpu.model_runner"
RUNNER_CLASS = "GPUModelRunner"
HOOK_ATTR = "_qwen38_fp8_lm_head_scope"
NO_LOGPROBS = -1  # v1/worker/gpu/sample/states.py:12


@dataclass(frozen=True)
class Fp8HeadSettings:
    topk: int = 64
    max_rows: int = 64
    chunk_rows: int = 128
    audit_every: int = 0
    audit_k: int = 20
    sample_topk: int = 20
    alternate: bool = False
    risk_every: int = 1000
    risk_margin: float = 1.5


def _int_env(environ, name: str, default: int, low: int, high: int) -> int:
    text = environ.get(name, "").strip()
    if not text:
        return default
    value = int(text)
    if not low <= value <= high:
        raise ValueError(f"{name}={value} is not in {low}..{high}")
    return value


def _float_env(environ, name: str, default: float, low: float, high: float) -> float:
    text = environ.get(name, "").strip()
    if not text:
        return default
    value = float(text)
    if not low <= value <= high:  # also refuses nan
        raise ValueError(f"{name}={value} is not in {low}..{high}")
    return value


def read_settings(environ=None) -> Fp8HeadSettings | None:
    """Return the settings, or None when VLLM_QWEN38_LM_HEAD_FP8 is not 1."""
    environ = os.environ if environ is None else environ
    if environ.get(ENV, "0").strip() != "1":
        return None
    s = Fp8HeadSettings(
        topk=_int_env(environ, TOPK_ENV, 64, 0, 4096),
        max_rows=_int_env(environ, MAX_ROWS_ENV, 64, 1, 1 << 20),
        chunk_rows=_int_env(environ, CHUNK_ENV, 128, 1, 1 << 16),
        audit_every=_int_env(environ, AUDIT_ENV, 0, 0, 1 << 30),
        audit_k=_int_env(environ, AUDIT_K_ENV, 20, 1, 4096),
        sample_topk=_int_env(environ, SAMPLE_TOPK_ENV, 20, 1, 4096),
        alternate=_int_env(environ, ALTERNATE_ENV, 0, 0, 1) == 1,
        risk_every=_int_env(environ, RISK_EVERY_ENV, 1000, 0, 1 << 30),
        risk_margin=_float_env(environ, RISK_MARGIN_ENV, 1.5, 0.0, 1e4),
    )
    if s.chunk_rows * s.topk > MAX_GATHER_ROWS:
        raise ValueError(
            f"{CHUNK_ENV} x {TOPK_ENV} = {s.chunk_rows} x {s.topk} is above "
            f"{MAX_GATHER_ROWS}: one rescore gather would hold "
            f"{s.chunk_rows * s.topk * 2560 * 2 / 2**30:.2f} GiB"
        )
    if s.topk > 0 and s.sample_topk > s.topk:
        raise ValueError(
            f"{SAMPLE_TOPK_ENV}={s.sample_topk} is above {TOPK_ENV}={s.topk}"
        )
    return s


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


# ---------------------------------------------------------------------------
# Sampler scope
# ---------------------------------------------------------------------------
_SCOPE = threading.local()
# Decisions of the hooked GPUModelRunner.sample: "fp8" or the fallback reason.
SCOPE_COUNTS: Counter = Counter()
_HOOK = {"sample_topk": 20}


def in_fp8_scope() -> bool:
    return getattr(_SCOPE, "allow", False)


def in_sampled_batch() -> bool:
    """True inside the scope of a batch with a sampled request (the audit)."""
    return getattr(_SCOPE, "sampled", False)


@contextlib.contextmanager
def fp8_scope(allow: bool = True, sampled: bool = False):
    """Let apply() use the FP8 path in this block (allow=True) or not.

    sampled marks a batch with at least one request whose temperature is not
    0. Only the audit reads it, to count those rows on their own.
    """
    prev = (getattr(_SCOPE, "allow", False), getattr(_SCOPE, "sampled", False))
    _SCOPE.allow, _SCOPE.sampled = allow, sampled
    try:
        yield
    finally:
        _SCOPE.allow, _SCOPE.sampled = prev


def batch_bf16_reason(runner, input_batch, grammar_output, sample_topk: int) -> str | None:
    """None when every request of the batch can use the FP8 path.

    Otherwise the reason for the BF16 head. The state arrays are the NumPy
    views that the V2 sampler reads itself (sample/sampler.py:74-85). A
    missing attribute gives "unknown", so a change of the runner falls back
    to the BF16 head.
    """
    if grammar_output is not None:
        return "grammar"
    try:
        sampler = runner.sampler
        idx = input_batch.idx_mapping_np
        st = sampler.sampling_states
        if np.any(sampler.logit_bias_state.use_logit_bias[idx]):
            return "logit_bias"
        if np.any(sampler.penalties_state.use_penalty[idx]):
            return "penalty"
        if np.any(sampler.bad_words_state.num_bad_words.np[idx] > 0):
            return "bad_words"
        if np.any(st.min_p.np[idx] != 0.0):
            return "min_p"
        if np.any(st.num_logprobs[idx] != NO_LOGPROBS) or np.any(
            sampler.logprob_token_ids_state.num_token_ids.np[idx] > 0
        ):
            return "logprobs"
        sampled = st.temperature.np[idx] != 0.0
        if np.any(sampled & (st.top_k.np[idx] > sample_topk)):
            return "top_k"
    except (AttributeError, IndexError, TypeError, ValueError):
        return "unknown"
    return None


def batch_sampled(runner, input_batch) -> bool:
    """True when a request of the batch has a temperature that is not 0."""
    try:
        idx = input_batch.idx_mapping_np
        return bool(np.any(runner.sampler.sampling_states.temperature.np[idx] != 0.0))
    except (AttributeError, IndexError, TypeError, ValueError):
        return False


def install_sampler_hook(sample_topk: int) -> str:
    """Wrap GPUModelRunner.sample so that it opens the FP8 scope.

    Returns "installed", "present" (installed before) or "missing" (the V2
    runner is not loaded in this process, so the FP8 path would never run).
    It does not import the runner module: the worker has imported it before
    the model is built, and other processes must not import it.
    """
    _HOOK["sample_topk"] = sample_topk
    mod = sys.modules.get(RUNNER_MODULE)
    cls = getattr(mod, RUNNER_CLASS, None) if mod is not None else None
    orig = getattr(cls, "sample", None) if cls is not None else None
    if orig is None:
        return "missing"
    if getattr(orig, HOOK_ATTR, False):
        return "present"

    @functools.wraps(orig)
    def sample(self, *args, **kwargs):
        # The call site is self.sample(hidden_states, input_batch,
        # grammar_output) (model_runner.py:1814).
        input_batch = kwargs.get("input_batch", args[1] if len(args) > 1 else None)
        grammar = kwargs.get("grammar_output", args[2] if len(args) > 2 else None)
        if input_batch is None:
            reason = "unknown"
        else:
            reason = batch_bf16_reason(self, input_batch, grammar, _HOOK["sample_topk"])
        SCOPE_COUNTS["fp8" if reason is None else reason] += 1
        sampled = reason is None and batch_sampled(self, input_batch)
        with fp8_scope(reason is None, sampled):
            return orig(self, *args, **kwargs)

    setattr(sample, HOOK_ATTR, True)
    setattr(cls, "sample", sample)
    return "installed"


# ---------------------------------------------------------------------------
# FP8 screen and BF16 rescore
# ---------------------------------------------------------------------------
def quantize_rows_fp8(
    weight: torch.Tensor, build_rows: int = BUILD_ROWS
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a [N, K] weight to e4m3 with one FP32 scale per row.

    Returns w8 [Npad, K] and scale [1, Npad], Npad = N rounded up to
    ROW_ALIGN. The pad rows are zero. The loop quantizes build_rows rows at a
    time, so the FP32 temporaries stay small. The result does not depend on
    build_rows, because each row is quantized independently.
    """
    n, k = weight.shape
    n_pad = -(-n // ROW_ALIGN) * ROW_ALIGN
    w8 = torch.zeros((n_pad, k), dtype=FP8_DTYPE, device=weight.device)
    scale = torch.ones((n_pad, 1), dtype=torch.float32, device=weight.device)
    for r0 in range(0, n, build_rows):
        r1 = min(n, r0 + build_rows)
        wf = weight[r0:r1].float()
        s = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_MAX
        w8[r0:r1] = (wf / s).to(FP8_DTYPE)
        scale[r0:r1] = s
        del wf, s
    return w8, scale.t().contiguous()


def quantize_act_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [M, K] activations to e4m3 with one FP32 scale per row."""
    xf = x.float()
    xs = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / FP8_MAX
    return (xf / xs).to(FP8_DTYPE), xs


def screen_logits_scaled_mm(
    x: torch.Tensor, w8: torch.Tensor, w8_scale: torch.Tensor, rows: int
) -> torch.Tensor:
    """FP8 screen on the GPU: rowwise-scaled torch._scaled_mm, BF16 output."""
    x8, xs = quantize_act_fp8(x)
    out = torch._scaled_mm(
        x8, w8.t(), scale_a=xs, scale_b=w8_scale, out_dtype=torch.bfloat16
    )
    return out[:, :rows]


def screen_logits_reference(
    x: torch.Tensor, w8: torch.Tensor, w8_scale: torch.Tensor, rows: int
) -> torch.Tensor:
    """The same values as screen_logits_scaled_mm, computed in FP32.

    Products of two e4m3 values are exact in FP32, so this differs from the
    GPU kernel only in the FP32 summation order. The CPU tests and
    tools/fp8head_offline_eval.py use it.
    """
    x8, xs = quantize_act_fp8(x)
    acc = x8.float() @ w8.float().t()
    return (acc * xs * w8_scale).to(torch.bfloat16)[:, :rows]


def rescore_(
    logits: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    topk: int,
    chunk_rows: int,
    rank: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Write BF16 logits into the topk best screen positions of each row.

    logits [M, N] is changed in place. x [M, K] and weight [N, K] are BF16.
    The candidate values are computed in FP32: the gathered rows and x are
    converted to FP32, multiplied (exact products) and summed, then rounded
    once to BF16. The FP32 temporaries hold CHUNK x topk x K values.

    Returns (idx, bounds). idx [M, topk] holds the candidate indices. bounds
    is None when rank is 0. Else it is [M, 2] FP32: column 0 is the smallest
    screen value of the candidates (every screen value outside the
    candidates is at most this value), column 1 is the rank-th best rescored
    value of the row. The risk counter compares the two.
    """
    m = logits.shape[0]
    k = min(topk, logits.shape[1])
    rank = min(rank, k)
    parts, bparts = [], []
    for r0 in range(0, m, chunk_rows):
        r1 = min(m, r0 + chunk_rows)
        vals, idx = logits[r0:r1].topk(k, dim=-1, sorted=False)
        w = weight.index_select(0, idx.reshape(-1)).view(r1 - r0, k, weight.shape[1])
        wf = w.float()
        del w
        wf.mul_(x[r0:r1].float().unsqueeze(1))
        exact = wf.sum(dim=-1).to(logits.dtype)
        del wf
        logits[r0:r1].scatter_(1, idx, exact)
        parts.append(idx)
        if rank > 0:
            s_k = vals.float().amin(dim=-1)
            v_s = exact.float().topk(rank, dim=-1, sorted=False).values.amin(dim=-1)
            bparts.append(torch.stack((s_k, v_s), dim=-1))
    idx = parts[0] if len(parts) == 1 else torch.cat(parts)
    if not bparts:
        return idx, None
    return idx, (bparts[0] if len(bparts) == 1 else torch.cat(bparts))


def rows_at_risk(bounds: torch.Tensor, margin: float) -> torch.Tensor:
    """[M] bool: s_k + margin >= v_s (the risk rule of the docstring)."""
    return bounds[:, 0] + margin >= bounds[:, 1]


class Qwen38Fp8LMHeadMethod(UnquantizedEmbeddingMethod):
    """Quant method of the target lm_head: FP8 screen plus BF16 rescore.

    It keeps the method that it replaces (inner) and calls it for every case
    that it does not handle: a call outside the sampler scope, no FP8 copy,
    a bias, batch-invariant mode, a dtype other than BF16, or more than
    max_rows rows.
    """

    # The devices on which process_weights_after_loading builds the copy.
    # The PLE offload worker builds the model on the meta device and must
    # not build it. The CPU tests add "cpu".
    build_devices: tuple[str, ...] = ("cuda",)
    # The build needs the sampler hook, because without it the FP8 path
    # never runs. The microbench and the CPU tests set it to False and open
    # the scope themselves with fp8_scope().
    hook_required: bool = True

    def __init__(self, inner: UnquantizedEmbeddingMethod, settings: Fp8HeadSettings):
        super().__init__()
        self.inner = inner
        self.settings = settings
        self.w8: torch.Tensor | None = None
        self.w8_scale: torch.Tensor | None = None
        self.rows = 0
        self.screen_fn = screen_logits_scaled_mm
        self.audit_calls = 0
        self.alternate_calls = 0
        self.hook_status = "not needed"
        self._audit: dict[str, torch.Tensor] | None = None
        # Risk counter: calls and rows on the host, rows at risk on the GPU.
        self.risk_calls = 0
        self.risk_rows = 0
        self._risk_at: torch.Tensor | None = None

    # Weight creation and embedding lookups stay with the inner method.
    def create_weights(self, layer, *args, **kwargs):
        return self.inner.create_weights(layer, *args, **kwargs)

    def embedding(self, layer, input_):
        return self.inner.embedding(layer, input_)

    def tie_weights(self, layer, embed_tokens):
        return self.inner.tie_weights(layer, embed_tokens)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        self.inner.process_weights_after_loading(layer)
        weight = getattr(layer, "weight", None)
        if weight is None or weight.dim() != 2 or weight.dtype != torch.bfloat16:
            logger.warning("Qwen3.8 FP8 lm_head: no 2-D BF16 weight; BF16 head kept.")
            return
        if weight.device.type not in self.build_devices:
            return
        if self.hook_required:
            self.hook_status = install_sampler_hook(self.settings.sample_topk)
            if self.hook_status == "missing":
                logger.warning(
                    "Qwen3.8 FP8 lm_head: %s.%s.sample is not loaded; BF16 head kept.",
                    RUNNER_MODULE, RUNNER_CLASS,
                )
                return
        self.w8, self.w8_scale = quantize_rows_fp8(weight.data)
        self.rows = weight.shape[0]
        logger.info(
            "%s %dx%d FP8 (%.2f GiB extra), rescore top %d in BF16, sampled "
            "top_k <= %d, rows <= %d, chunk %d, audit every %d, alternate %d, "
            "risk every %d margin %.3g, sampler scope %s",
            BUILD_LOG,
            weight.shape[0],
            weight.shape[1],
            self.w8.numel() / 2**30,
            self.settings.topk,
            self.settings.sample_topk,
            self.settings.max_rows,
            self.settings.chunk_rows,
            self.settings.audit_every,
            int(self.settings.alternate),
            self.settings.risk_every,
            self.settings.risk_margin,
            self.hook_status,
        )

    def _usable(self, layer: nn.Module, x: torch.Tensor, bias) -> bool:
        weight = layer.weight
        return (
            self.w8 is not None
            and bias is None
            and not envs.VLLM_BATCH_INVARIANT
            and x.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and x.device == self.w8.device
            and x.shape[-1] == weight.shape[1]
            and 0 < x.numel() // x.shape[-1] <= self.settings.max_rows
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not in_fp8_scope() or not self._usable(layer, x, bias):
            return self.inner.apply(layer, x, bias)
        if self.settings.alternate:
            # Timing arm: calls 2, 4, 6, ... use the BF16 head.
            self.alternate_calls += 1
            if self.alternate_calls % 2 == 0:
                return self.inner.apply(layer, x, bias)
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        logits = self.screen_fn(x2, self.w8, self.w8_scale, self.rows)
        capturing = _capturing()
        audit = (
            self.settings.audit_every > 0
            and x2.shape[0] <= AUDIT_MAX_ROWS
            and not capturing
        )
        screen = logits.clone() if audit else None
        cand = risk = None
        if self.settings.topk > 0:
            want_risk = (self.settings.risk_every > 0 or audit) and not capturing
            cand, bounds = rescore_(
                logits,
                x2,
                layer.weight,
                self.settings.topk,
                self.settings.chunk_rows,
                rank=self.settings.sample_topk if want_risk else 0,
            )
            if bounds is not None:
                risk = rows_at_risk(bounds, self.settings.risk_margin)
                if self.settings.risk_every > 0:
                    self._risk_step(x2.shape[0], risk)
        if audit:
            self._audit_step(layer, x2, logits, screen, cand, risk)
        # With pad rows the screen output is a column slice. The sampler and
        # the grammar bitmask kernels expect packed rows, so pack it (no copy
        # when N % 16 == 0, as for 248320).
        return logits.contiguous().reshape(*lead, self.rows)

    def _risk_step(self, rows: int, risk: torch.Tensor) -> None:
        """Add the rows at risk on the GPU; log every RISK_EVERY calls."""
        if self._risk_at is None:
            self._risk_at = torch.zeros((), dtype=torch.int64, device=risk.device)
        self._risk_at.add_(risk.sum())
        self.risk_calls += 1
        self.risk_rows += rows
        if self.risk_calls % self.settings.risk_every == 0:
            at_risk = int(self._risk_at)  # the one host sync of the counter
            logger.info(
                "%s calls %d rows %d at_risk %d margin %.3g rank %d",
                RISK_LOG,
                self.risk_calls,
                self.risk_rows,
                at_risk,
                self.settings.risk_margin,
                self.settings.sample_topk,
            )

    def _audit_step(self, layer, x, out, screen, cand, risk=None) -> None:
        """Compare the returned logits with the BF16 head (counters on the GPU).

        The rows of a batch with a sampled request are also counted in the
        sampled_* counters. missed counts the rows that fail (a) or (b) and
        that the risk counter did not flag; it has a meaning when AUDIT_K is
        SAMPLE_TOPK (both 20 by default).
        """
        ref = self.inner.apply(layer, x, None)
        k = min(self.settings.audit_k, ref.shape[-1])
        ref_v, ref_i = ref.topk(k, dim=-1)
        out_v = out.topk(k, dim=-1).values
        ref_top1 = ref.argmax(dim=-1)
        if self._audit is None:
            zero = torch.zeros((), dtype=torch.int64, device=ref.device)
            fzero = torch.zeros((), dtype=torch.float32, device=ref.device)
            self._audit = {name: zero.clone() for name in AUDIT_COUNTS}
            self._audit.update(max_diff=fzero.clone(), screen_max_diff=fzero.clone())
        a = self._audit
        rows = torch.ones(x.shape[0], dtype=torch.bool, device=ref.device)
        per_row = {
            "rows": rows,
            "top1": out.argmax(dim=-1) == ref_top1,
            "topk_values": (out_v == ref_v).all(dim=-1),
        }
        a["screen_top1"] += (screen.argmax(dim=-1) == ref_top1).sum()
        if cand is not None:
            # (a): the audit-k best BF16 tokens are candidates.
            hit = (ref_i.unsqueeze(-1) == cand.unsqueeze(-2)).any(dim=-1).all(dim=-1)
            # (b): every value that is not a candidate is below the k-th
            # BF16 value.
            is_cand = torch.zeros_like(out, dtype=torch.bool).scatter_(1, cand, True)
            outside = out.masked_fill(is_cand, float("-inf")).amax(dim=-1)
            out_ok = outside < ref_v[:, -1]
            per_row.update(contain=hit, outside_ok=out_ok)
            if risk is not None:
                a["at_risk"] += risk.sum()
                a["missed"] += (~(hit & out_ok) & ~risk).sum()
        for name, ok in per_row.items():
            a[name] += ok.sum()
        if in_sampled_batch():
            for name, ok in per_row.items():
                a["sampled_" + name] += ok.sum()
        ref_f = ref_v.float()
        a["max_diff"] = torch.maximum(
            a["max_diff"], (out.gather(1, ref_i).float() - ref_f).abs().amax()
        )
        a["screen_max_diff"] = torch.maximum(
            a["screen_max_diff"], (screen.gather(1, ref_i).float() - ref_f).abs().amax()
        )
        self.audit_calls += 1
        if self.audit_calls % self.settings.audit_every == 0:
            scope = ",".join(f"{r}={n}" for r, n in sorted(SCOPE_COUNTS.items())) or "none"
            logger.info(
                "%s calls %d rows %d top1 %d screen_top1 %d top%d_values %d "
                "contain %d topk %d max_diff %.6g screen_max_diff %.6g "
                "outside_ok %d at_risk %d missed %d sampled_rows %d sampled_top1 %d "
                "sampled_topk_values %d sampled_contain %d sampled_outside_ok %d "
                "scope %s",
                AUDIT_LOG,
                self.audit_calls,
                int(a["rows"]),
                int(a["top1"]),
                int(a["screen_top1"]),
                k,
                int(a["topk_values"]),
                int(a["contain"]),
                self.settings.topk,
                float(a["max_diff"]),
                float(a["screen_max_diff"]),
                int(a["outside_ok"]),
                int(a["at_risk"]),
                int(a["missed"]),
                int(a["sampled_rows"]),
                int(a["sampled_top1"]),
                int(a["sampled_topk_values"]),
                int(a["sampled_contain"]),
                int(a["sampled_outside_ok"]),
                scope,
            )


def enable_fp8_lm_head(model: nn.Module, environ=None) -> bool:
    """Swap the quant method of model.lm_head. Return True when it swapped.

    model.py calls this at the end of Qwen3_8FlashNextForCausalLM.__init__.
    The MTP drafter gets the same lm_head module later (load_eagle_model,
    spec_decode/eagle/utils.py:83-101), because Qwen3_8FlashNextMTP has no
    has_own_lm_head flag. Its calls run outside the sampler scope and use
    the BF16 head, with or without MTP_DRAFT_VOCAB.
    """
    settings = read_settings(environ)
    if settings is None:
        return False
    head = getattr(model, "lm_head", None)
    method = getattr(head, "quant_method", None)
    if isinstance(method, Qwen38Fp8LMHeadMethod):
        return False
    if not isinstance(method, UnquantizedEmbeddingMethod):
        logger.warning(
            "Qwen3.8 FP8 lm_head: quant method %s is not unquantized; BF16 head kept.",
            type(method).__name__,
        )
        return False
    if getattr(head, "tp_size", 1) != 1:
        logger.warning("Qwen3.8 FP8 lm_head: TP > 1 is not supported; BF16 head kept.")
        return False
    head.quant_method = Qwen38Fp8LMHeadMethod(method, settings)
    return True
