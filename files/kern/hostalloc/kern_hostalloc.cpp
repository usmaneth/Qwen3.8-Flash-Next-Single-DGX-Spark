// SPDX-License-Identifier: Apache-2.0
// L7 H1: a torch CUDAPluggableAllocator for the weight tensors.
//
// The bw-ceiling study (a1-spark2-20260925T005729, alloc mode) read a
// cudaHostAlloc buffer at 238.6 GB/s and a cudaMalloc buffer at 231.2 GB/s
// with the same stream kernel (+3.2%; +4.5% in the verifier run). Pageable,
// managed and registered memory read at 163-167 GB/s. This allocator lets a
// torch.cuda.MemPool hold the weights in one of these kinds:
//   KERN_HOSTALLOC_MODE=host      cudaHostAlloc(cudaHostAllocMapped), device
//                                 pointer from cudaHostGetDevicePointer
//                                 (the same address on GB10, UVA).
//   KERN_HOSTALLOC_MODE=vmm_dev   cuMemCreate on the device location, 2 MiB
//                                 granularity, cuMemMap + cuMemSetAccess.
//   KERN_HOSTALLOC_MODE=vmm_host  cuMemCreate on CU_MEM_LOCATION_TYPE_HOST_NUMA 0.
//   KERN_HOSTALLOC_MODE=device    cudaMalloc (the control arm in the pool).
// The mode is read once, at the first allocation. Errors print to stderr and
// return nullptr (torch raises an OOM error).
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace {

enum Mode { kHost = 0, kVmmDev = 1, kVmmHost = 2, kDevice = 3 };

struct VmmRec {
  CUmemGenericAllocationHandle h;
  size_t size;
};

std::mutex g_mu;
std::unordered_map<void*, VmmRec> g_vmm;
int g_mode = -1;
size_t g_live = 0, g_peak = 0, g_count = 0;

int mode() {
  if (g_mode >= 0) return g_mode;
  const char* m = std::getenv("KERN_HOSTALLOC_MODE");
  g_mode = kHost;
  if (m && !std::strcmp(m, "vmm_dev")) g_mode = kVmmDev;
  if (m && !std::strcmp(m, "vmm_host")) g_mode = kVmmHost;
  if (m && !std::strcmp(m, "device")) g_mode = kDevice;
  return g_mode;
}

void* vmm_alloc(size_t size, int device, bool host) {
  CUmemAllocationProp prop;
  std::memset(&prop, 0, sizeof prop);
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = host ? CU_MEM_LOCATION_TYPE_HOST_NUMA : CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = host ? 0 : device;
  size_t gran = 0;
  if (cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS || !gran)
    return nullptr;
  size_t sz = (size + gran - 1) / gran * gran;
  CUmemGenericAllocationHandle h;
  if (cuMemCreate(&h, sz, &prop, 0) != CUDA_SUCCESS) return nullptr;
  CUdeviceptr p = 0;
  if (cuMemAddressReserve(&p, sz, gran, 0, 0) != CUDA_SUCCESS) {
    cuMemRelease(h);
    return nullptr;
  }
  if (cuMemMap(p, sz, 0, h, 0) != CUDA_SUCCESS) {
    cuMemAddressFree(p, sz);
    cuMemRelease(h);
    return nullptr;
  }
  CUmemAccessDesc acc;
  std::memset(&acc, 0, sizeof acc);
  acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  acc.location.id = device;
  acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  if (cuMemSetAccess(p, sz, &acc, 1) != CUDA_SUCCESS) {
    cuMemUnmap(p, sz);
    cuMemAddressFree(p, sz);
    cuMemRelease(h);
    return nullptr;
  }
  g_vmm[(void*)p] = VmmRec{h, sz};
  return (void*)p;
}

}  // namespace

extern "C" {

void* kern_host_malloc(ssize_t size, int device, cudaStream_t stream) {
  (void)stream;
  if (size <= 0) return nullptr;
  std::lock_guard<std::mutex> lk(g_mu);
  cudaSetDevice(device);
  void* p = nullptr;
  switch (mode()) {
    case kHost: {
      void* hp = nullptr;
      cudaError_t e = cudaHostAlloc(&hp, (size_t)size, cudaHostAllocMapped);
      if (e != cudaSuccess) {
        std::fprintf(stderr, "kern_hostalloc: cudaHostAlloc(%zd) failed: %s\n", size, cudaGetErrorString(e));
        return nullptr;
      }
      if (cudaHostGetDevicePointer(&p, hp, 0) != cudaSuccess || p != hp) {
        // The pool frees by the device pointer; demand the UVA identity.
        std::fprintf(stderr, "kern_hostalloc: device pointer differs from the host pointer\n");
        cudaFreeHost(hp);
        return nullptr;
      }
      break;
    }
    case kVmmDev:
    case kVmmHost:
      p = vmm_alloc((size_t)size, device, mode() == kVmmHost);
      if (!p) std::fprintf(stderr, "kern_hostalloc: VMM allocation of %zd bytes failed\n", size);
      break;
    default:
      if (cudaMalloc(&p, (size_t)size) != cudaSuccess) p = nullptr;
  }
  if (p) {
    g_live += (size_t)size;
    g_count += 1;
    if (g_live > g_peak) g_peak = g_live;
  }
  return p;
}

void kern_host_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
  (void)stream;
  if (!ptr) return;
  std::lock_guard<std::mutex> lk(g_mu);
  cudaSetDevice(device);
  switch (mode()) {
    case kHost:
      cudaFreeHost(ptr);
      break;
    case kVmmDev:
    case kVmmHost: {
      auto it = g_vmm.find(ptr);
      if (it != g_vmm.end()) {
        cuMemUnmap((CUdeviceptr)ptr, it->second.size);
        cuMemAddressFree((CUdeviceptr)ptr, it->second.size);
        cuMemRelease(it->second.h);
        g_vmm.erase(it);
      }
      break;
    }
    default:
      cudaFree(ptr);
  }
  g_live -= (size_t)size;
}

// Statistics for the tests and the micro: live bytes, peak bytes, count, mode.
void kern_host_stats(size_t* out4) {
  std::lock_guard<std::mutex> lk(g_mu);
  out4[0] = g_live;
  out4[1] = g_peak;
  out4[2] = g_count;
  out4[3] = (size_t)mode();
}

}  // extern "C"
