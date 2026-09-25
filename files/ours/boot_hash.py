"""Verification hash dump for the boot-fast loader work (gate G1).

start.sh mounts this file as vllm/boot_hash.py only with BOOT_HASH=1, and the
hooks call it only when VLLM_BOOT_HASH_DIR is set. A time-only boot does not
mount it.

For each process that owns weights, the dump writes one manifest line per
tensor: each parameter, each buffer (non-persistent too), each tensor
attribute of a module, and the tensor fields of the objects that a module
holds (two levels, for example quant_method.moe_quant_config). A line holds
the name, the kind, the dtype, the shape, the stride, the storage offset and a
blake2b-128 hash of the bytes in logical (row-major) order. A tensor with
more than one name is recorded under each name and hashed once.

  * GPU worker: dump_runner(target, drafter) at the end of
    GPUModelRunner.load_model (after both loads, both process steps and the
    embed/lm_head share).
  * PLE offload worker: dump_offload(layers, packed_tables) at the end of
    _load_weights (after the packed table is attached and the process step
    ran). For a packed table, the dump records path, size and mtime only.

The hash copies each tensor to the CPU in 64 MiB chunks. Eight threads hash
tensors in parallel (hashlib releases the GIL).
"""
import dataclasses
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch

CHUNK = 64 << 20
KNOBS = ("VLLM_ST_WINDOW", "VLLM_ST_WINDOW_METHOD", "VLLM_ST_WINDOW_THREADS",
         "VLLM_ST_DROP_DONE", "VLLM_ST_SKIP_PLE_ONLY", "VLLM_MTP_FILE_GLOB",
         "VLLM_PLE_OFFLOAD_FILE_GLOB", "VLLM_EXPERT_LOOKUP", "VLLM_SKIP_MM_WARMUP")


def _chunks(t):
    """CPU uint8 arrays of the tensor bytes, in logical order."""
    t = t.detach()
    if t.numel() == 0:
        return
    if t.is_contiguous():
        flat = t.reshape(-1).view(torch.uint8)
        for s in range(0, flat.numel(), CHUNK):
            yield flat[s : s + CHUNK].cpu().numpy()
        return
    if t.dim() == 0:
        yield t.contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
        return
    row = max(1, t[0].numel() * t.element_size())
    step = max(1, CHUNK // row)
    for s in range(0, t.shape[0], step):
        yield t[s : s + step].contiguous().reshape(-1).view(torch.uint8).cpu().numpy()


def _hash(t):
    if t.device.type == "meta":
        return "meta"
    h = hashlib.blake2b(digest_size=16)
    for arr in _chunks(t):
        h.update(memoryview(arr))
    return h.hexdigest()


def _objects(obj, depth, seen):
    """(suffix, tensor) for the tensor fields of a non-module object."""
    if depth == 0 or id(obj) in seen:
        return
    seen.add(id(obj))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        fields = {f.name: getattr(obj, f.name, None) for f in dataclasses.fields(obj)}
    else:
        fields = getattr(obj, "__dict__", None)
        if not isinstance(fields, dict):
            return
    for k, v in list(fields.items()):
        if isinstance(v, torch.Tensor):
            yield k, v
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                if isinstance(x, torch.Tensor):
                    yield f"{k}[{i}]", x
        elif isinstance(v, torch.nn.Module) or v is None:
            continue
        elif type(v).__module__.split(".")[0] in ("builtins", "torch", "numpy", "enum"):
            continue
        else:
            for k2, x in _objects(v, depth - 1, seen):
                yield f"{k}.{k2}", x


def collect(model, prefix=""):
    """(name, kind, tensor) for every tensor that the model holds."""
    out = []
    for n, p in model.named_parameters(remove_duplicate=False):
        out.append((prefix + n, "param", p))
    for n, b in model.named_buffers(remove_duplicate=False):
        out.append((prefix + n, "buffer", b))
    for mn, m in model.named_modules(remove_duplicate=False):
        base = prefix + (mn + "." if mn else "")
        for k, v in list(vars(m).items()):
            if k.startswith("_") and k in ("_parameters", "_buffers", "_modules"):
                continue
            if isinstance(v, torch.Tensor):
                out.append((base + k, "attr", v))
            elif isinstance(v, (list, tuple)):
                for i, x in enumerate(v):
                    if isinstance(x, torch.Tensor):
                        out.append((f"{base}{k}[{i}]", "attr", x))
            elif isinstance(v, torch.nn.Module) or v is None:
                continue
            elif type(v).__module__.split(".")[0] in ("builtins", "torch", "numpy", "enum"):
                continue
            else:
                for k2, x in _objects(v, 2, set()):
                    out.append((f"{base}{k}.{k2}", "obj", x))
    return out


def _key(t):
    try:
        ptr = t.data_ptr()
    except Exception:
        ptr = id(t)
    return (t.device.type, ptr, str(t.dtype), tuple(t.shape), tuple(t.stride()),
            t.storage_offset())


def _dump(role, items, extra=None):
    out_dir = os.environ["VLLM_BOOT_HASH_DIR"]
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    uniq = {}
    for _, _, t in items:
        uniq.setdefault(_key(t), t)
    lock = threading.Lock()
    hashes = {}

    def work(k):
        h = _hash(uniq[k])
        with lock:
            hashes[k] = h

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(work, list(uniq)))
    path = os.path.join(out_dir, f"manifest-{role}-{os.getpid()}.jsonl")
    nbytes = sum(t.numel() * t.element_size() for t in uniq.values())
    with open(path, "w") as fh:
        fh.write(json.dumps({"header": True, "role": role, "pid": os.getpid(),
                             "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "count": len(items), "unique": len(uniq),
                             "bytes": nbytes, "hash_s": round(time.time() - t0, 1),
                             "knobs": {k: os.environ.get(k) for k in KNOBS},
                             **(extra or {})}) + "\n")
        for name, kind, t in items:
            k = _key(t)
            fh.write(json.dumps({"name": name, "kind": kind, "dtype": str(t.dtype),
                                 "shape": list(t.shape), "stride": list(t.stride()),
                                 "offset": t.storage_offset(), "device": t.device.type,
                                 "hash": hashes[k]}) + "\n")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"boot-fast hash: {role} {len(items)} tensors ({len(uniq)} unique, "
          f"{nbytes / 2**30:.2f} GiB) in {time.time() - t0:.1f}s -> {path}", flush=True)


def dump_runner(target, drafter=None):
    items = collect(target, "target.")
    if drafter is not None:
        items += collect(drafter, "drafter.")
    _dump("gpu", items)


def dump_offload(layers, packed_tables):
    items = []
    for name, layer in layers.items():
        items += collect(layer, f"offload.{name}.")
    # The packed table is a 27 GB file-backed tensor; no fix changes it, and a
    # hash would read all of it. Record its shape here and its file below.
    skipped = [(n, list(t.shape)) for n, _, t in items if n.endswith("._packed_table")]
    items = [x for x in items if not x[0].endswith("._packed_table")]
    packed = {"skipped_tensors": skipped}
    for name, path in (packed_tables or {}).items():
        st = os.stat(path)
        packed[name] = {"path": path, "size": st.st_size, "mtime": st.st_mtime}
    _dump("offload", items, {"packed_tables": packed})
