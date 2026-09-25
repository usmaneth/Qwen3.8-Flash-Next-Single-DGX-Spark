#!/usr/bin/env bash
# One kern-l7 lease (copy of kern-decode ab/lease_inlaunch.sh with WT=kern-l7 and
# the L7 wrapper inlaunch_l7.py): probe, one launch, in-launch arms, stop.
#   gpu-lease run --nodes spark2 --timeout 7200 kern-<label> -- \
#       bash lease_inlaunch.sh <label> <plan.json> [VAR=value ...]
# The VAR=value pairs go to start.sh (for example VLLM_MOE_DET_FINALIZE=1).
# PROFILE=1 adds the torch profiler config (no stacks, 40 iterations).
# The lease ends with ./stop.sh and a check that no vllm-fn-tp1 container is
# up. A watchdog stops the server after LEASE_MAX_S (default 3300 s).
set -u
LABEL=$1; PLAN=$2; shift 2
PY=/home/usman/.pyenv/versions/3.12.14/bin/python3
WT=/models/usman/qwen38-flash-wt/kern-l7
K=/models/usman/kern-decode
R=/models/usman/qwen38-tune
OUT=$K/runs/$LABEL-$(hostname -s)-$(date +%Y%m%dT%H%M%S); mkdir -p "$OUT"; L=$OUT/run.log
T0=$(date +%s); el() { echo $(( $(date +%s) - T0 )); }
log() { echo "[$(date +%T) +$(el)s] $*" | tee -a "$L"; }
LEASE_MAX_S=${LEASE_MAX_S:-3300}
cp "$PLAN" "$OUT/plan.json"
( cd $WT && { git rev-parse HEAD 2>/dev/null && git status --short; } || cat .kern_head ) > "$OUT/head.txt" 2>&1
log "lease $LABEL on $(hostname -s), worktree $(head -1 "$OUT/head.txt"), env: $*"
( while :; do echo "$(date +%s) $(awk '/MemAvailable|MemFree/{printf "%s %d ", $1, $2/1024}' /proc/meminfo)"; sleep 5; done ) > "$OUT/mem.log" 2>&1 &
MEMPID=$!
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv -l 5 > "$OUT/gpu.csv" 2>&1 &
GPUPID=$!
DONE=0
finish() {
  [[ $DONE == 1 ]] && return; DONE=1
  log "final stop (no restore)"
  docker logs vllm-fn-tp1 > "$OUT/container.log" 2>&1
  ( cd $WT && ./stop.sh >> "$OUT/stop.log" 2>&1 )
  pkill -P $WDPID 2>/dev/null; kill $MEMPID $GPUPID $WDPID 2>/dev/null
  if docker ps --format '{{.Names}}' | grep -q vllm-fn-tp1; then
    log "WARN: container still up; docker rm -f"; docker rm -f vllm-fn-tp1 >> "$OUT/stop.log" 2>&1
  fi
  docker ps --format '{{.Names}}' | grep -q vllm-fn-tp1 && log "ERROR: container still up" || log "server stopped"
  log "probe after stop: $(bash $R/probe.sh 2>&1 | tail -1)"
}
trap 'finish; exit 130' INT TERM HUP
# The watchdog does not inherit stdout: over ssh (lease_any.sh) an orphaned
# sleep would keep the session, and so the GPU lease, open after the lease.
( sleep "$LEASE_MAX_S"; echo "[watchdog] $LEASE_MAX_S s: stop" >> "$L"; kill -TERM $$ ) </dev/null >/dev/null 2>&1 &
WDPID=$!

cd $WT || exit 1
./stop.sh >> "$OUT/stop.log" 2>&1
for _ in $(seq 1 60); do [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 95 ] && break; sleep 3; done
log "MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GiB, probe $(bash $R/probe.sh 2>&1 | tail -1)"
# The kern-gate lease was queued before R3a v2 and R6 existed: it runs the
# PLE wait micro test first and adds the R6 build (logged in run.log).
# Optional GPU microbenches before the launch (PRE_MICRO="tool.py tool2.py").
for tool in ${PRE_MICRO:-}; do
  cp "$K/tools/"*.py "$OUT/" 2>/dev/null
  log "pre-micro: $tool"
  timeout 900 docker run --rm --gpus all --name kern-micro --entrypoint python3 \
    -e TRITON_CACHE_DIR=/triton-cache -v /models/usman/triton-cache:/triton-cache \
    -v $WT/files/kern:/k:ro -v "$OUT":/o vllm/vllm-openai:qwen38-flash-next "/o/$tool" 2>&1 \
    | grep -v "^INFO\|^WARNING\|^W0" | tee -a "$L" | tail -3
  docker rm -f kern-micro >/dev/null 2>&1
done
if [[ " ${PRE_MICRO:-} " == *" ple_wait_test.py "* ]] && ! grep -q '"all_ok": true' "$OUT/ple_wait.json" 2>/dev/null; then
  log "ple_wait_test failed: launch without PLE_GPU_WAIT (the R3a arms fail their kd_set and are skipped)"
  set -- "${@/PLE_GPU_WAIT=1/PLE_GPU_WAIT=0}"
fi
BASE_VLLM="$(env "$@" bash -c 'source ./.env >/dev/null 2>&1; printf %s "$EXTRA_VLLM_ARGS"')"
EXTRA=()
if [[ "${PROFILE:-0}" == 1 ]]; then
  mkdir -p "$OUT/prof"; chmod 777 "$OUT/prof"
  BASE_DOCKER="$(env "$@" bash -c 'source ./.env >/dev/null 2>&1; printf %s "$EXTRA_DOCKER_ARGS"')"
  cfg="'{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/prof\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":false,\"ignore_frontend\":true,\"torch_profiler_dump_cuda_time_total\":false,\"max_iterations\":40}'"
  EXTRA=(EXTRA_VLLM_ARGS="$BASE_VLLM --profiler-config $cfg" EXTRA_DOCKER_ARGS="$BASE_DOCKER -v $OUT/prof:/prof")
fi
log "start.sh"
env "$@" "${EXTRA[@]}" ./start.sh > "$OUT/start.log" 2>&1 || { log "start failed: $(tail -3 "$OUT/start.log")"; finish; exit 1; }
cp .last_launch.sh "$OUT/launch.sh" 2>/dev/null
up=0
for _ in $(seq 1 300); do
  [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8888/v1/models)" = 200 ] && { up=1; break; }
  docker ps --format '{{.Names}}' | grep -q vllm-fn-tp1 || break
  sleep 5
done
[[ $up == 1 ]] || { log "not ready"; finish; exit 1; }
log "ready: $(grep -oE "cudagraph_capture_sizes': \[[^]]*\]|GPU KV cache size: [0-9,]+ tokens" "$OUT/start.log" | sort -u | tr '\n' ' ')"
docker logs vllm-fn-tp1 2>&1 | grep -E "FP8 lm_head|Injected|MTP draft head|Capturing|capturing finished|DET|determin" | cut -c1-240 >> "$L"
DEADLINE=$(( T0 + LEASE_MAX_S - 240 ))
$PY $WT/tools/l7/inlaunch_l7.py --plan "$OUT/plan.json" --out "$OUT" --deadline $DEADLINE 2>&1 | tee -a "$L"
if [[ "${PROFILE:-0}" == 1 ]]; then
  sleep 20
  ls -la "$OUT/prof" >> "$L" 2>&1
fi
docker logs vllm-fn-tp1 2>&1 | grep -E "FP8 lm_head risk|Traceback|Error" | tail -20 | cut -c1-240 >> "$L"
finish
log "DONE $OUT"
