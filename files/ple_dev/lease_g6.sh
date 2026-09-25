#!/usr/bin/env bash
# One G6 micro lease for TECHNIQUES.md #7 (PLAN: /models/usman/frontier/next/ple/PLAN.md).
# No server. The pinned image runs g6_bench.py. The lease floats:
#   /home/usman/bin/gpu-lease-any --nodes spark1,spark2 --timeout 43200 fx-ple-g6-<n> -- \
#     bash /models/usman/qwen38-flash-wt/fx-ple/files/ple_dev/lease_g6.sh <n>
# LEASE_NODE (from gpu-lease-any) picks the node. spark2 gets the tools and the
# golden data with rsync first, and the run goes over ssh. Budget: 45 min.
set -u
N=${1:-1}
NODE=${LEASE_NODE:-$(hostname -s)}
WT=/models/usman/qwen38-flash-wt/fx-ple
DEV=$WT/files/ple_dev
F=/models/usman/frontier/next/ple
RUN=$F/g6-$N-$NODE-$(date -u +%Y%m%dT%H%M%SZ)
TABLE_DIR=/models/usman/vllm-ple-cache
IMG=vllm/vllm-openai:qwen38-flash-next
mkdir -p "$RUN"; L=$RUN/run.log
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$L"; }
on() { if [[ "$NODE" == "$(hostname -s)" ]]; then bash -c "$1"; else ssh "$NODE" "$1"; fi; }

if [[ "$NODE" != "$(hostname -s)" ]]; then
  ssh "$NODE" "mkdir -p $DEV $F/parity-20260925 $F/coldmiss-20260925 $RUN"
  rsync -a "$DEV/" "$NODE:$DEV/"
  rsync -a "$F/parity-20260925/" "$NODE:$F/parity-20260925/"
  rsync -a "$F/coldmiss-20260925/coldmiss.json" "$NODE:$F/coldmiss-20260925/"
fi
( cd "$WT" && git rev-parse HEAD ) > "$RUN/head.txt"
log "G6 lease $N on $NODE, head $(cat "$RUN/head.txt")"
# Stop rules before the GPU work: no vLLM server may run. The memory rule for
# a GPU job is the memwatch floor (brief correction 2026-09-25 05:30): this
# job has no server and maps at most a few MB of the table, so it only needs
# MemAvailable above the floor with a margin (10 GiB) on either node.
if on "docker ps --format '{{.Names}}'" | grep -q vllm; then log "a vllm container is up: stop"; exit 3; fi
avail=$(on "awk '/^MemAvailable:/ {print int(\$2/1048576)}' /proc/meminfo")
log "MemAvailable ${avail} GiB"
if [[ "$avail" -lt 10 ]]; then log "MemAvailable < 10 GiB: stop"; exit 4; fi
log "probe before: $(on 'bash /models/usman/qwen38-tune/probe.sh 2>&1 | tail -1')"
on "timeout 2700 docker run --rm --gpus all --name fx-ple-g6 --entrypoint python3 \
  -e TRITON_CACHE_DIR=/triton-cache -v /models/usman/triton-cache:/triton-cache \
  -v $TABLE_DIR:$TABLE_DIR:ro -v $DEV:$DEV:ro -v $F/parity-20260925:$F/parity-20260925:ro \
  -v $F/coldmiss-20260925:$F/coldmiss-20260925:ro -v $RUN:$RUN \
  $IMG $DEV/g6_bench.py --golden $F/parity-20260925 --parity $F/parity-20260925/parity.json \
  --coldmiss $F/coldmiss-20260925/coldmiss.json --out $RUN" 2>&1 \
  | grep -v '^INFO\|^WARNING\|^W0' | tee -a "$L"
rc=${PIPESTATUS[0]}
on "docker rm -f fx-ple-g6 >/dev/null 2>&1" || true
if [[ "$NODE" != "$(hostname -s)" ]]; then rsync -a "$NODE:$RUN/" "$RUN/"; fi
log "probe after: $(on 'bash /models/usman/qwen38-tune/probe.sh 2>&1 | tail -1')"
log "DONE rc=$rc $RUN"
exit "$rc"
