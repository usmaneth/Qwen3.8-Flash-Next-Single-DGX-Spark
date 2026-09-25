# SPDX-License-Identifier: Apache-2.0
"""Real token streams for the PLE studies, and the decode step replay.

    python3 streams.py --out DIR [--src JSONL ...]

Source: the datagen records of the served model (distill/gen.*.jsonl). Each
record has the prompt_token_ids of a real request and the token_ids that the
model generated for it. The script writes DIR/streams.npz:
  tokens  int32 [N]      prompt + output of each record, back to back
  start   int64 [R + 1]  record r is tokens[start[r]:start[r + 1]]
  plen    int64 [R]      prompt length of record r
  shape   uint8 [R]      0 = plain, 1 = codex

The replay (class Replay) turns the streams into the batches that the V2
model runner gives to the PLE layer, with the rules of the served profile
(spark1-best.env: MTP K=6, MAX_NUM_SEQS=4, MAX_NUM_BATCHED_TOKENS=8192):
  - A prefill chunk is prompt[n:n + c]. A prefix-cache hit starts n at a
    multiple of 16 inside the prompt.
  - A decode request sends [last sampled token, d1 .. dK]. The first a drafts
    are the true next tokens, a comes from the measured per-position
    acceptance (accept.jsonl, tag codex-default-r1). Draft a + 1 is a wrong
    token, and the drafts after it are wrong tokens too. A wrong token is a
    token of the same stream at a random position (a real token id).
  - After the step, num_computed grows by a + 1. Rejected drafts never enter
    the context.
  - ngram_context of a request is the two tokens before num_computed (EOS
    before position 0), as Qwen3_8FlashNextModelState._prepare_ngram_context
    makes it. Padded request rows are EOS.
  - The runner buffers are persistent: a padding token keeps the value of an
    earlier step (the stale value), and the padded query_start_loc entries
    equal the unpadded num_tokens (model_runner.py:1229).
The replay does not decide the CUDA graph padding. The caller adds it.
"""
import argparse
import json
import os

import numpy as np

EOS = 248044
# Cumulative acceptance per draft position at K=6 (accept.jsonl,
# tag codex-default-r1, 2026-09-24): P(a >= i) for i = 1..6.
ACCEPT_CUM = (0.797, 0.631, 0.504, 0.403, 0.335, 0.281)
SRC = ("/models/usman/distill/gen.snapshot2.jsonl",)


def build(srcs, out: str) -> str:
    toks, start, plen, shape = [], [0], [], []
    n = 0
    for src in srcs:
        with open(src) as f:
            for line in f:
                d = json.loads(line)
                p = d.get("prompt_token_ids") or []
                o = d.get("token_ids") or []
                if len(p) < 1 or len(o) < 1:
                    continue
                s = np.asarray(p + o, dtype=np.int64)
                if s.min() < 0 or s.max() >= 1 << 31:
                    continue
                toks.append(s.astype(np.int32))
                n += s.size
                start.append(n)
                plen.append(len(p))
                shape.append(1 if d.get("shape") == "codex" else 0)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "streams.npz")
    np.savez(path, tokens=np.concatenate(toks), start=np.asarray(start, np.int64),
             plen=np.asarray(plen, np.int64), shape=np.asarray(shape, np.uint8),
             src=np.asarray([",".join(srcs)]))
    return path


class Streams:
    def __init__(self, path: str) -> None:
        z = np.load(path)
        self.tokens, self.start = z["tokens"], z["start"]
        self.plen, self.shape = z["plen"], z["shape"]

    def __len__(self) -> int:
        return self.plen.size

    def seq(self, r: int) -> np.ndarray:
        return self.tokens[self.start[r]:self.start[r + 1]]


class _Req:
    __slots__ = ("rid", "seq", "lp", "n")

    def __init__(self, rid: int, seq: np.ndarray, lp: int, n: int) -> None:
        self.rid, self.seq, self.lp, self.n = rid, seq, lp, n


class Replay:
    """Yields the scheduler steps of a set of records, as lists of segments.

    A segment is (rid, kind, tokens int32, n, true_len): kind is "prefill"
    or "decode", n is num_computed before the step, and true_len is the
    number of tokens of the segment that are the stream itself (the rest are
    rejected drafts). step() also returns the ngram_context rows.
    """

    def __init__(self, st: Streams, rids, rng: np.random.Generator, k: int = 6,
                 max_seqs: int = 4, budget: int = 8192,
                 seq_hook=None) -> None:
        self.st, self.rids, self.rng = st, list(rids), rng
        self.k, self.max_seqs, self.budget = k, max_seqs, budget
        self.seq_hook = seq_hook
        self.next = 0
        self.run: list[_Req] = []
        self.target = 1

    def _admit(self) -> None:
        while len(self.run) < self.target and self.next < len(self.rids):
            rid = self.rids[self.next]
            self.next += 1
            seq = self.st.seq(rid)
            if self.seq_hook is not None:
                seq = self.seq_hook(rid, seq)
            lp = int(self.st.plen[rid])
            n = 0
            if lp > 32 and self.rng.random() < 0.5:  # prefix-cache hit
                n = int(self.rng.integers(1, lp // 16)) * 16
            self.run.append(_Req(rid, seq, lp, n))

    def _accept(self) -> int:
        u = self.rng.random()
        a = 0
        for p in ACCEPT_CUM[:self.k]:
            if u < p:
                a += 1
            else:
                break
        return a

    def _wrong(self, seq: np.ndarray, true_tok: int) -> int:
        while True:
            t = int(seq[int(self.rng.integers(0, seq.size))])
            if t != true_tok:
                return t

    @staticmethod
    def ctx(seq: np.ndarray, n: int) -> tuple[int, int]:
        a = int(seq[n - 2]) if n - 2 >= 0 else EOS
        b = int(seq[n - 1]) if n - 1 >= 0 else EOS
        return a, b

    def done(self) -> bool:
        return not self.run and self.next >= len(self.rids)

    def step(self):
        if self.rng.random() < 0.02 or not self.run:
            self.target = int(self.rng.choice([1, 2, 3, 4], p=[0.5, 0.25, 0.15, 0.1]))
            self.target = min(self.target, self.max_seqs)
        self._admit()
        if not self.run:
            return None
        order = list(self.run)
        self.rng.shuffle(order)
        # Decode requests first, as the scheduler serves running requests.
        order.sort(key=lambda q: 0 if q.n >= q.lp else 1)
        left = self.budget
        segs, ctxs = [], []
        for q in order:
            if left <= 0:
                break
            if q.n < q.lp:
                c = min(q.lp - q.n, left)
                toks = q.seq[q.n:q.n + c].astype(np.int32)
                segs.append((q.rid, "prefill", toks, q.n, c))
                ctxs.append(self.ctx(q.seq, q.n))
                q.n += c
                left -= c
                continue
            kk = min(self.k, left - 1)
            a = min(self._accept(), kk)
            s = q.seq
            toks = [int(s[q.n])]
            for i in range(1, kk + 1):
                j = q.n + i
                true_tok = int(s[j]) if j < s.size else -2
                if i <= a and j < s.size:
                    toks.append(true_tok)
                else:
                    toks.append(self._wrong(s, true_tok))
            true_len = 1 + min(a, max(0, s.size - 1 - q.n))
            segs.append((q.rid, "decode", np.asarray(toks, np.int32), q.n, true_len))
            ctxs.append(self.ctx(s, q.n))
            q.n = min(q.n + a + 1, s.size - 1)
            left -= 1 + kk
        self.run = [q for q in self.run if not (q.n >= q.lp and q.n >= q.seq.size - 1)]
        return segs, ctxs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", nargs="*", default=list(SRC))
    a = ap.parse_args()
    path = build(a.src, a.out)
    st = Streams(path)
    lp = st.plen.sum()
    print(f"{path}: {len(st)} records, {st.tokens.size} tokens "
          f"({lp} prompt, {st.tokens.size - lp} output), EOS {int((st.tokens == EOS).sum())}")


if __name__ == "__main__":
    main()
