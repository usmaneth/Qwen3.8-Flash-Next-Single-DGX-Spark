# SPDX-License-Identifier: Apache-2.0
"""L7 hooks: the tile tables of the micro sweep and the L7 kernels.

enable_l7(model) runs at the end of Qwen3_8FlashNextForCausalLM.__init__
when VLLM_KERN_L7=1 (gen_kern.py writes the call). It
  - loads the tile table VLLM_KERN_L7_TILES (a JSON file of the L7 micro
    sweep) and keeps it OFF until set_tiles(True) (kd_ext "call" knob
    l7:set_tiles) or VLLM_KERN_L7_TILES_ON=1;
  - patches the FlashInfer MXFP8 linear kernel (skinny_mx.py, S1) when
    VLLM_KERN_SKINNY_MX=1;
  - swaps the shared-expert gate linears (sgate.py, S3) when VLLM_KERN_SGATE=1.

Tile JSON format:
  {"skinny_bf16": {"512,2560": [bn, bk, split, warps, stages], ...},
   "mtp_w8a16":   {"2560,6144": [...], ...},
   "skinny_mx":   {"16384,2560": [...], ...}}
"skinny_bf16" (R6t) retunes the R6 table (kern-decode-2 owns R6: this only
adds entries to its TILES dict at run time, it does not change its file).
"mtp_w8a16" (S2) retunes the R4 table. set_tiles(False) restores the
tables as they were before; a recapture is needed after each change.
"""
import importlib
import json
import os

ENV = "VLLM_KERN_L7"
PKG = "vllm.models.qwen3_8_flash_next.nvidia."
_STATE = {"table": None, "orig": {}, "on": False}


def _mod(name):
    try:
        return importlib.import_module(PKG + name)
    except ImportError:
        return importlib.import_module(name)


def load_table(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    table = {}
    for mod, entries in raw.items():
        if mod.startswith("_"):
            continue
        t = {}
        for key, tile in entries.items():
            n, k = (int(v) for v in key.split(","))
            tile = tuple(int(v) for v in tile)
            if len(tile) != 5 or tile[1] % 16 or tile[2] < 1:
                raise ValueError(f"l7 tiles: bad tile {mod} {key} {tile}")
            t[(n, k)] = tile
        table[mod] = t
    return table


def set_tiles(on: bool = True, table: dict | None = None) -> dict:
    """Merge the L7 tiles into the module TILES dicts (on) or restore them (off)."""
    if table is not None:
        _STATE["table"] = table
    table = _STATE["table"] or {}
    done = {}
    for name, entries in table.items():
        mod = _mod(name)
        tiles = mod.TILES
        orig = _STATE["orig"].setdefault(name, dict(tiles))
        tiles.clear()
        tiles.update(orig)
        if on:
            tiles.update(entries)
        done[name] = len(entries) if on else 0
    _STATE["on"] = bool(on)
    return done


def kd_call(model, arg) -> dict:
    """kd_ext "call" entry: arg true/false switches the L7 tile table."""
    return set_tiles(bool(arg))


def enable_l7(model, logger=None) -> dict:
    if os.environ.get(ENV, "0") != "1":
        return {}
    out = {}
    path = os.environ.get("VLLM_KERN_L7_TILES")
    if path:
        _STATE["table"] = load_table(path)
        out["tiles"] = set_tiles(os.environ.get("VLLM_KERN_L7_TILES_ON", "0") == "1")
    if os.environ.get("VLLM_KERN_SKINNY_MX", "0") == "1":
        out["skinny_mx"] = _mod("skinny_mx").patch_flashinfer_mxfp8(logger)
    if os.environ.get("VLLM_KERN_SGATE", "0") == "1":
        out["sgate"] = len(_mod("sgate").enable_sgate(model, logger))
    if logger is None:
        try:
            from vllm.logger import init_logger

            logger = init_logger(__name__)
        except ImportError:  # pragma: no cover
            logger = None
    if logger is not None:
        logger.info("kern L7: %s", out)
    return out
