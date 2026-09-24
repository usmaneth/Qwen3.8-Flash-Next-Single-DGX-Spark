"""CPU check for the align-mode chunk split of files/ours/scheduler.py.

The test takes _mamba_block_aligned_split and _mamba_split_grid from the
generated scheduler.py (patch_block_drop.py, then patch_mamba_grid.py) and runs
them on a stub scheduler. The expected chunk ends come from the replay
/models/usman/prefill-ttft/poison_sim.py, which reproduces the hit counts that
the live servers showed.

    python3 patch_block_drop.py && python3 patch_mamba_grid.py   (in files/ours)
    python3 -m unittest tests/test_split.py
"""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "files" / "ours" / "scheduler.py"
WANT = ("_mamba_block_aligned_split", "_mamba_split_grid")


def load_methods():
    tree = ast.parse(SRC.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    funcs = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in WANT]
    if len(funcs) != len(WANT):
        raise unittest.SkipTest("scheduler.py is not patched: run patch_mamba_grid.py")
    mod = ast.Module(body=funcs, type_ignores=[])
    flags = SimpleNamespace(value="mamba")
    ns = {
        "Request": object,
        "logger": logging.getLogger("test_split"),
        "_pt_flags": SimpleNamespace(get=lambda name, default: flags.value
                                     if name == "split_grid" else default),
    }
    exec(compile(mod, str(SRC), "exec"), ns)
    return ns, flags


NS, FLAGS = load_methods()


class Sched:
    """The attributes that the split reads."""
    _mamba_block_aligned_split = NS["_mamba_block_aligned_split"]
    _mamba_split_grid = NS["_mamba_split_grid"]

    def __init__(self, mamba_block, budget, hash_block=None, lookahead=1):
        self.cache_config = SimpleNamespace(block_size=12)   # the QSA ring (engine core.py:322)
        self.mamba_state_block_size = mamba_block
        self.block_size = mamba_block
        self.hash_block_size = hash_block or mamba_block
        self.mamba_partial_cache_hit = hash_block is not None and hash_block < mamba_block
        self.use_eagle_block_drop = False                     # vllm#53388 backport on
        self.max_num_scheduled_tokens = budget
        self.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0)
        self.num_prefill_lookahead = lookahead
        self._pt_debug_split = False


def chunk_ends(sched, prompt, start=0, junction=0, grid="mamba"):
    """Run one prefill from start to the prompt end; return the chunk ends."""
    FLAGS.value = grid
    req = SimpleNamespace(request_id="r", num_computed_tokens=start, num_prompt_tokens=prompt,
                          num_tokens=prompt, shared_prefix_boundary=junction)
    ends = []
    while req.num_computed_tokens < prompt:
        n = min(prompt - req.num_computed_tokens, sched.max_num_scheduled_tokens)
        n = sched._mamba_block_aligned_split(req, n)
        # _reserve_prefill_lookahead runs after the split in schedule().
        remaining = req.num_tokens - req.num_computed_tokens - n
        if 0 < remaining < sched.num_prefill_lookahead:
            n -= sched.num_prefill_lookahead - remaining
        assert n > 0, (req.num_computed_tokens, ends)
        req.num_computed_tokens += n
        ends.append(req.num_computed_tokens)
    return ends


class LegacyGrid(unittest.TestCase):
    """The 12-token grid: the chunk plans that poison_sim.py replays."""

    def test_codex_prime_k6(self):
        self.assertEqual(chunk_ends(Sched(1728, 8192), 11226, grid="legacy"), [8184, 11220, 11226])

    def test_64k_prime_k6(self):
        ends = chunk_ends(Sched(1728, 8192), 65536, grid="legacy")
        self.assertEqual(ends[:3], [8184, 16368, 24552])
        self.assertTrue(any(e % 1728 for e in ends[:-1]))


class MambaGrid(unittest.TestCase):
    def assert_on_grid(self, ends, block):
        for e in ends[:-1]:
            self.assertEqual(e % block, 0, ends)

    def test_codex_prime_k6(self):
        # poison_sim FIX: states at 6912 and 10368; the new session hits 10368.
        self.assertEqual(chunk_ends(Sched(1728, 8192), 11226), [6912, 10368, 11226])

    def test_codex_prime_k4(self):
        self.assertEqual(chunk_ends(Sched(1680, 8192), 11226), [6720, 10080, 11226])

    def test_block_multiple_budget(self):
        ends = chunk_ends(Sched(1728, 8640), 65536)
        self.assertEqual(ends, [8640 * i for i in range(1, 8)] + [63936, 65536])

    def test_long_prompts_stay_on_grid(self):
        for block, budget in ((1728, 8192), (1728, 10368), (1728, 15552), (1680, 8400)):
            for prompt in (1727, 1728, 1729, 11226, 65536, 200000, 262144, 524288):
                ends = chunk_ends(Sched(block, budget), prompt)
                self.assert_on_grid(ends, block)
                self.assertLessEqual(max(b - a for a, b in zip([0] + ends, ends)), budget)
                # The last full boundary is a chunk end: a new turn hits there.
                floor = prompt // block * block
                if 0 < floor < prompt:
                    self.assertIn(floor, ends)

    def test_hit_start_mid_block_realigns(self):
        # A fine (48-token) hit starts inside a block; the first chunk stops at
        # the next Mamba boundary.
        ends = chunk_ends(Sched(1728, 8640, hash_block=48), 20000, start=1728 * 3 + 480)
        self.assertEqual(ends[0], 1728 * 4)
        self.assert_on_grid(ends[:-1], 1728)

    def test_partial_tail_stop(self):
        # prefix-match-unit 48: the chunk also ends at the prompt's last
        # 48-token boundary, after the last Mamba boundary.
        prompt = 11226
        ends = chunk_ends(Sched(1728, 8640, hash_block=48), prompt)
        self.assertEqual(ends, [8640, 10368, prompt // 48 * 48, prompt])

    def test_junction_is_block_floored(self):
        ends = chunk_ends(Sched(1728, 8640), 30000, junction=12000)
        self.assertIn(12000 // 1728 * 1728, ends)
        self.assert_on_grid(ends, 1728)

    def test_lookahead_never_moves_an_end_off_grid(self):
        # Multi-module MTP reserves num_prefill_lookahead tokens before the
        # end. A prompt 3 tokens past a boundary must not end a chunk there.
        prompt = 1728 * 6 + 3
        for budget in (8192, 8640, 10368):
            ends = chunk_ends(Sched(1728, budget, lookahead=6), prompt)
            self.assert_on_grid(ends, 1728)
            self.assertNotIn(1728 * 6, ends)
        # Lookahead 1 (this model: one MTP module) keeps the boundary stop.
        self.assertIn(1728 * 6, chunk_ends(Sched(1728, 8640), prompt))


if __name__ == "__main__":
    unittest.main()
