"""Runtime A/B flags for the prefill-ttft experiments.

get(name, default) reads the file /pt-flags/<name> (the host flag directory,
mounted read-only) and keeps the value for 1 s. When the file is absent, the
environment variable PT_<NAME> gives the value. When that is absent too, the
default gives the value. The patched code reads a flag once per scheduler step
or once per forward, so one server can switch between A/B arms.

start.sh mounts this file at vllm/pt_flags.py. Without the mount, the patched
code falls back to the defaults.
"""
import os
import time

FLAG_DIR = os.environ.get("PT_FLAG_DIR", "/pt-flags")
TTL_S = 1.0
_cache: dict[str, tuple[float, str | None]] = {}


def _read(name: str) -> str | None:
    try:
        with open(os.path.join(FLAG_DIR, name)) as f:
            value = f.read().strip()
    except OSError:
        return None
    return value or None


def get(name: str, default: str) -> str:
    now = time.monotonic()
    hit = _cache.get(name)
    if hit is not None and now - hit[0] < TTL_S:
        value = hit[1]
    else:
        value = _read(name)
        _cache[name] = (now, value)
    if value is None:
        value = os.environ.get("PT_" + name.upper())
    return default if value is None else value


def get_int(name: str, default: int) -> int:
    try:
        return int(get(name, str(default)))
    except ValueError:
        return default
