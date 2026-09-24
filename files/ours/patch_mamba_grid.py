#!/usr/bin/env python3
"""Put the align-mode prefill split on the Mamba state grid (prefill-ttft B1).

Problem. The EngineCore sets cache_config.block_size to the smallest KV cache
group block (engine/core.py:322). On Qwen3.8-Flash-Next that is the QSA ring,
12 tokens. _mamba_block_aligned_split reads that value, so chunk ends land on
a 12-token grid and not on the Mamba block (1728 at K=6, 1680 at K=4). The
worker writes a GDN state only into the running block of a step. An old
running block keeps the state of the last step end inside it, and
cache_blocks later hashes that block as the entry for its full block end. A
hit on that entry resumes the GDN layers from a state up to one block short of
the prefix (the vllm#43559 class). This is the cause of F4 in TTFT.md.

Change (the grid part of vllm#54076):
  * The split reads the Mamba group block size (MambaSpec.block_size).
  * Runtime flag split_grid (pt_flags): "mamba" (default) or "legacy" (the old
    cache_config.block_size grid, for the A/B only).
  * Runtime flag budget (pt_flags): the step token budget, at most the boot
    value of --max-num-batched-tokens. 0 or absent keeps the boot value.
  * The split never ends a non-final chunk within num_prefill_lookahead of
    the prompt end. The lookahead reservation would move that end off the grid.
  * PT_DEBUG_SPLIT=1 logs each chunk (start, end, stop reason) and each Mamba
    hash registration (block, claimed end, state position, verdict).

Input: files/ours/scheduler.py as patch_block_drop.py writes it. The script
edits that file in place and refuses to run twice.

    python3 patch_block_drop.py && python3 patch_mamba_grid.py
"""
import os
import sys

OUT = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(OUT, "scheduler.py")
MARK = "prefill-ttft B1"

OLD_SPLIT_HEAD = '''        block_size = self.cache_config.block_size
        # The last block-aligned position whose state can be cached. With
        # Eagle, FullAttn prunes the last matching block, so back off one
        # block to avoid a Mamba cache miss.
        last_cache_position = request.num_tokens - request.num_tokens % block_size
        if self.use_eagle_block_drop:
            last_cache_position = max(last_cache_position - block_size, 0)

        end = start + num_new_tokens
'''
NEW_SPLIT_HEAD = '''        # prefill-ttft B1: the Mamba state grid, not the smallest group block.
        block_size = self._mamba_split_grid()
        # The last block-aligned position whose state can be cached. With
        # Eagle, FullAttn prunes the last matching block, so back off one
        # block to avoid a Mamba cache miss.
        last_cache_position = request.num_tokens - request.num_tokens % block_size
        if self.use_eagle_block_drop:
            last_cache_position = max(last_cache_position - block_size, 0)
        # _reserve_prefill_lookahead shortens a chunk that ends within
        # num_prefill_lookahead of the prompt end. That would move a block
        # aligned end off the grid, so such a position is never a chunk end.
        lookahead = self.num_prefill_lookahead

        def leaves_lookahead(pos: int) -> bool:
            return not 0 < request.num_tokens - pos < lookahead

        end = start + num_new_tokens
        reason = "final" if end >= prefill_end else "budget"
'''

OLD_ALIGN = '''            aligned_end = end // block_size * block_size
            if aligned_end > start or block_size <= max_prefill_tokens:
                end = aligned_end
'''
NEW_ALIGN = '''            aligned_end = end // block_size * block_size
            if aligned_end > start or block_size <= max_prefill_tokens:
                if aligned_end != end:
                    reason = "aligned"
                end = aligned_end
'''

OLD_STOPS = '''            # Never run past the last cacheable block boundary mid-chunk.
            last_cache_position,
'''
NEW_STOPS = '''            # Never run past the last cacheable block boundary mid-chunk.
            last_cache_position if leaves_lookahead(last_cache_position) else 0,
'''

OLD_TAIL = '''        # Stop at the earliest mandatory position strictly inside the chunk.
        end = min((s for s in stops if start < s < end), default=end)
        return max(end - start, 0)
'''
NEW_TAIL = '''        # Stop at the earliest mandatory position strictly inside the chunk.
        names = ("block", "last_cache", "tail", "junction")
        inside = [(s, n) for s, n in zip(stops, names) if start < s < end]
        if inside:
            end, reason = min(inside)
        if (
            end < request.num_tokens
            and not leaves_lookahead(end)
            and end % block_size == 0
            and end - block_size > start
        ):
            end -= block_size
            reason += "+lookahead"
        if self._pt_debug_split:
            self._pt_split_end[request.request_id] = end
            logger.info(
                "PTDBG split req=%s start=%d end=%d n=%d prompt=%d grid=%d "
                "reason=%s",
                request.request_id,
                start,
                end,
                end - start,
                request.num_prompt_tokens,
                block_size,
                reason,
            )
        return max(end - start, 0)

    def _mamba_split_grid(self) -> int:
        """prefill-ttft B1: the token grid of the align-mode chunk split."""
        if (
            _pt_flags is not None
            and _pt_flags.get("split_grid", "mamba") == "legacy"
        ):
            return self.cache_config.block_size
        return self.mamba_state_block_size

    def _pt_step_budget(self) -> int:
        """prefill-ttft B1: the step token budget (runtime flag budget)."""
        if _pt_flags is None:
            return self.max_num_scheduled_tokens
        budget = _pt_flags.get_int("budget", 0)
        if budget <= 0:
            return self.max_num_scheduled_tokens
        return min(budget, self.max_num_scheduled_tokens)

    def _pt_install_mamba_debug(self) -> None:
        """prefill-ttft B1: log each Mamba hash registration of one group.

        A registration is valid when the block holds the state at exactly its
        claimed position. The wrapper records, for each prefill step, the
        position whose state the step writes into its running block.
        """
        managers = [
            m
            for m in self.kv_cache_manager.coordinator.single_type_managers
            if type(m).__name__ == "MambaManager"
        ]
        if not managers:
            return
        mgr = managers[0]
        mamba_block = mgr.block_size
        state_pos: dict[int, tuple[str, int]] = {}
        totals = {"valid": 0, "poisoned": 0, "unknown": 0, "decode": 0}
        orig_cache_blocks = mgr.cache_blocks
        orig_partial = mgr._cache_partial_tail_block

        def verdict_of(block, claimed: int) -> tuple[str, str]:
            kind, pos = state_pos.get(block.block_id, ("?", -1))
            if kind == "d":
                verdict = "decode"
            elif pos == claimed:
                verdict = "valid"
            elif pos < 0:
                verdict = "unknown"
            else:
                verdict = "poisoned"
            totals[verdict] += 1
            return verdict, f"{kind}{pos}"

        def cache_blocks(request, num_tokens, retention_interval=None):
            # The coordinator rounds num_tokens down to the scheduler block,
            # so take the real step end from the split of this step.
            blocks = mgr.req_to_blocks.get(request.request_id, [])
            end = self._pt_split_end.pop(request.request_id, None)
            if end is None:
                end, kind = num_tokens, "d"
            else:
                kind = "p"
            idx = (end - 1) // mamba_block
            if end > 0 and idx < len(blocks) and not blocks[idx].is_null:
                state_pos[blocks[idx].block_id] = (kind, end)
            before = mgr.num_cached_block.get(request.request_id, 0)
            orig_cache_blocks(
                request, num_tokens, retention_interval=retention_interval
            )
            after = mgr.num_cached_block.get(request.request_id, 0)
            for k in range(before, min(after, len(blocks))):
                block = blocks[k]
                if block.is_null or block.block_hash is None:
                    continue
                claimed = (k + 1) * mamba_block
                verdict, state = verdict_of(block, claimed)
                logger.info(
                    "PTDBG mamba_reg grid=%d req=%s idx=%d claimed=%d state=%s "
                    "verdict=%s totals=%s",
                    self._mamba_split_grid(),
                    request.request_id,
                    k,
                    claimed,
                    state,
                    verdict,
                    totals,
                )

        def cache_partial(request, num_tokens):
            partial_hash = orig_partial(request, num_tokens)
            if partial_hash is not None:
                block = mgr.req_to_blocks[request.request_id][
                    num_tokens // mamba_block
                ]
                verdict, state = verdict_of(block, num_tokens)
                logger.info(
                    "PTDBG mamba_partial grid=%d req=%s claimed=%d state=%s "
                    "verdict=%s totals=%s",
                    self._mamba_split_grid(),
                    request.request_id,
                    num_tokens,
                    state,
                    verdict,
                    totals,
                )
            return partial_hash

        mgr.cache_blocks = cache_blocks
        mgr._cache_partial_tail_block = cache_partial
        logger.info("PTDBG Mamba registration log on (group block %d)", mamba_block)
'''

OLD_INIT = '''        # A finer prefix_match_unit is configured: a mamba partial tail entry
'''
NEW_INIT = '''        # prefill-ttft B1: the grid of the align split is the Mamba group
        # block. cache_config.block_size is the smallest group block here.
        self.mamba_state_block_size = next(
            (
                g.kv_cache_spec.block_size
                for g in kv_cache_config.kv_cache_groups
                if isinstance(g.kv_cache_spec, MambaSpec)
            ),
            self.cache_config.block_size,
        )
        self._pt_debug_split = os.environ.get("PT_DEBUG_SPLIT") == "1"
        self._pt_split_end: dict[str, int] = {}
        if self.need_mamba_block_aligned_split:
            logger.info(
                "prefill-ttft B1: align split grid %d (Mamba group block), "
                "cache_config.block_size %d, scheduler block %d, hash block %d, "
                "boot budget %d, flags %s",
                self.mamba_state_block_size,
                self.cache_config.block_size,
                self.block_size,
                self.hash_block_size,
                self.max_num_scheduled_tokens,
                "on" if _pt_flags is not None else "off",
            )
            if self._pt_debug_split:
                self._pt_install_mamba_debug()
        # A finer prefix_match_unit is configured: a mamba partial tail entry
'''

EDITS = [
    ("import itertools\n", "import itertools\nimport os\n"),
    ("from vllm.v1.kv_cache_interface import KVCacheConfig\n",
     "from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec\n"),
    ("from vllm.v1.utils import record_function_or_nullcontext\n",
     "from vllm.v1.utils import record_function_or_nullcontext\n\n"
     "try:  # prefill-ttft B1: runtime A/B flags (files/ours/pt_flags.py)\n"
     "    from vllm import pt_flags as _pt_flags\n"
     "except ImportError:\n"
     "    _pt_flags = None\n"),
    (OLD_SPLIT_HEAD, NEW_SPLIT_HEAD),
    (OLD_ALIGN, NEW_ALIGN),
    (OLD_STOPS, NEW_STOPS),
    (OLD_TAIL, NEW_TAIL),
    (OLD_INIT, NEW_INIT),
    ("        token_budget = self.max_num_scheduled_tokens\n",
     "        token_budget = self._pt_step_budget()\n"),
]


def main() -> None:
    s = open(PATH).read()
    if MARK in s:
        sys.exit(f"{PATH}: already patched ({MARK})")
    if "use_eagle_block_drop" not in s:
        sys.exit(f"{PATH}: run patch_block_drop.py first")
    for old, new in EDITS:
        if s.count(old) != 1:
            sys.exit(f"scheduler.py: anchor found {s.count(old)} times: {old[:70]!r}")
        s = s.replace(old, new, 1)
    open(PATH, "w").write(s)
    print("patched scheduler.py (Mamba grid split)")


if __name__ == "__main__":
    main()
