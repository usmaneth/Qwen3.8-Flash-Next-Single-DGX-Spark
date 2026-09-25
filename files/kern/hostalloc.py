# SPDX-License-Identifier: Apache-2.0
"""L7 H1: move the weight tensors of a loaded model into another memory kind.

The bw-ceiling study read cudaHostAlloc memory 3-4.5% faster than cudaMalloc
memory on GB10 (same stream kernel). The weights are about 90% of the bytes
of a decode step, so if the real kernels see the same gain the step gets
shorter with the same bits (class A: no kernel and no math change).

move_weights(model, to="host") copies every CUDA tensor of the model of at
least MIN_BYTES (parameters, buffers and plain tensor attributes such as
the R4/R5/R8 weight copies) into a torch.cuda.MemPool backed by
kern_hostalloc.so (the mode comes from KERN_HOSTALLOC_MODE), one storage at
a time. Views that share a storage keep sharing it: each tensor gets a view
of the new storage with its old offset, size and stride (t.data = view). to="device"
moves them back into the default caching allocator. Every CUDA graph holds
the old addresses, so the caller recaptures after a move (kd_ext knob
"call"). The old storages must be freed; move_weights reports the bytes
that the default allocator gave back (torch.cuda.memory_allocated() minus the
allocator's live bytes), and the caller stops the arm when the
freed share is below 95% (a reference that this function did not see keeps
the old copy alive and doubles the memory).
"""
import gc
import os

import torch

MIN_BYTES = 1 << 20
SO_ENV = "KERN_HOSTALLOC_SO"
_POOL = {"pool": None, "alloc": None, "so": None}
_MOVED = {}  # storage data_ptr (new) -> "host"


def _so_path() -> str:
    return os.environ.get(SO_ENV) or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "kern_hostalloc.so")


def pool():
    if _POOL["pool"] is None:
        from torch.cuda.memory import CUDAPluggableAllocator

        so = _so_path()
        alloc = CUDAPluggableAllocator(so, "kern_host_malloc", "kern_host_free")
        _POOL.update(alloc=alloc, so=so, pool=torch.cuda.MemPool(alloc.allocator()))
    return _POOL["pool"]


def stats() -> dict:
    import ctypes

    if _POOL["so"] is None:
        return {}
    lib = ctypes.CDLL(_POOL["so"])
    buf = (ctypes.c_size_t * 4)()
    lib.kern_host_stats(buf)
    return {"live": buf[0], "peak": buf[1], "count": buf[2], "mode": buf[3]}


def collect(model: torch.nn.Module, min_bytes: int = MIN_BYTES, device_type: str = "cuda"):
    """Group the model's large tensors by storage.

    Returns {storage key: [(owner, attr name, tensor), ...]} and the byte size
    of each storage. Parameters, buffers and tensor attributes of every module
    count; the same tensor object is listed once.
    """
    groups, sizes, seen = {}, {}, set()
    for mod in model.modules():
        items = list(mod._parameters.items()) + list(mod._buffers.items())
        items += [(k, v) for k, v in vars(mod).items()
                  if isinstance(v, torch.Tensor) and not k.startswith("_") and k not in mod._parameters
                  and k not in mod._buffers]
        for name, t in items:
            if t is None or not isinstance(t, torch.Tensor) or t.device.type != device_type:
                continue
            if id(t) in seen:
                continue
            seen.add(id(t))
            st = t.untyped_storage()
            nbytes = st.nbytes()
            if nbytes < min_bytes:
                continue
            key = st.data_ptr()
            groups.setdefault(key, []).append((mod, name, t))
            sizes[key] = nbytes
    return groups, sizes


def _rebind(entries, new_storage):
    # t.data = view swaps the tensor's storage in place, for Parameters and for
    # plain tensors, so every holder of the same Python object sees the move.
    with torch.no_grad():
        for _mod, _name, t in entries:
            view = torch.empty(0, dtype=t.dtype, device=new_storage.device)
            view.set_(new_storage, t.storage_offset(), t.size(), t.stride())
            t.data = view


def move_weights(model: torch.nn.Module, to: str = "host", min_bytes: int = MIN_BYTES,
                 empty_every: int = 2 << 30, _pool_ctx=None) -> dict:
    """Move the large tensors of ``model`` to the pool (to="host") or back (to="device")."""
    assert to in ("host", "device")
    dev_type = "cuda" if _pool_ctx is None else "cpu"
    groups, sizes = collect(model, min_bytes, dev_type)
    if to == "host":
        todo = [k for k in groups if k not in _MOVED]
    else:
        todo = [k for k in groups if k in _MOVED]
    # torch counts the pool blocks in memory_allocated(); the allocator's own
    # live count separates the pool from the default allocator.
    before = torch.cuda.memory_allocated() if dev_type == "cuda" else 0
    live0 = stats().get("live", 0)
    moved = since = 0
    for key in todo:
        entries = groups[key]
        old = entries[0][2].untyped_storage()
        if _pool_ctx is not None:
            ctx = _pool_ctx()
        elif to == "host":
            ctx = torch.cuda.use_mem_pool(pool())
        else:
            import contextlib

            ctx = contextlib.nullcontext()
        with ctx:
            new = torch.empty(old.nbytes(), dtype=torch.uint8, device=old.device).untyped_storage()
        new.copy_(old)
        _rebind(entries, new)
        if to == "host":
            _MOVED[new.data_ptr()] = "host"
        else:
            _MOVED.pop(key, None)
        moved += sizes[key]
        since += sizes[key]
        del old, new, entries
        if dev_type == "cuda" and since >= empty_every:
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            since = 0
    if dev_type == "cuda":
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated() if dev_type == "cuda" else 0
    live1 = stats().get("live", 0)
    freed = (before - live0) - (after - live1)  # bytes the default allocator gave back
    res = {"to": to, "storages": len(todo), "bytes": moved,
           "default_pool_freed": freed if to == "host" else -freed,
           "freed_share": (freed / moved if moved and to == "host" else None), "alloc": stats()}
    return res


def kd_call(model: torch.nn.Module, arg) -> dict:
    """kd_ext "call" entry: arg "host" or "device"."""
    return move_weights(model, to=str(arg))
