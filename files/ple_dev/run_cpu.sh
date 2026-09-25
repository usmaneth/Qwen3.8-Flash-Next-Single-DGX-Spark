#!/usr/bin/env bash
# Run one CPU-only PLE study on spark1 under the house rules:
# nice 19, idle I/O class, cores 0-4,10-14, no GPU, a memory cap, and a
# start only while MemAvailable >= MIN_AVAIL_GIB (default 20).
#   run_cpu.sh <mem-cap, e.g. 6G> <log> -- <command ...>
set -euo pipefail
cap=$1; log=$2; shift 3
min=${MIN_AVAIL_GIB:-20}
while :; do
  avail=$(awk '/^MemAvailable:/ {print int($2/1048576)}' /proc/meminfo)
  [ "$avail" -ge "$min" ] && break
  echo "$(date -u +%FT%TZ) wait: MemAvailable ${avail} GiB < ${min} GiB" >>"$log"
  sleep 60
done
echo "$(date -u +%FT%TZ) start on $(hostname): $*" >>"$log"
exec nice -n 19 ionice -c3 taskset -c 0-4,10-14 env CUDA_VISIBLE_DEVICES= \
  systemd-run --user --scope -q -p MemoryMax="$cap" "$@" >>"$log" 2>&1
