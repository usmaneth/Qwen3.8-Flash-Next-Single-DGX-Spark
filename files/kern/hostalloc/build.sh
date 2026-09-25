#!/usr/bin/env bash
# Build kern_hostalloc.so in the pinned image (CPU only; links the libcuda stub).
#   bash build.sh [out_dir]
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$HERE}
docker run --rm --user "$(id -u):$(id -g)" --entrypoint bash -v "$HERE":/src:ro -v "$OUT":/out vllm/vllm-openai:qwen38-flash-next -c \
  'g++ -O2 -shared -fPIC -std=c++17 -I/usr/local/cuda/include /src/kern_hostalloc.cpp \
     -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lcudart -lcuda -o /out/kern_hostalloc.so \
   && nm -D --defined-only /out/kern_hostalloc.so | grep -c "kern_host_" '
