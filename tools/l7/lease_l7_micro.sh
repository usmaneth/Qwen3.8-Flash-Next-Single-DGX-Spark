#!/usr/bin/env bash
# L7 micro lease (no server): the GPU gates of plans/l7-skinny.json, stage M1.
#   /home/usman/bin/gpu-lease-any --nodes spark1,spark2 --timeout 43200 kern3-micro-1 -- \
#       bash /models/usman/qwen38-flash-wt/kern-l7/tools/l7/lease_l7_micro.sh
# gpu-lease-any sets LEASE_NODE. On spark2 the script copies the worktree
# parts, the kern-decode tools and the fx-hc12 tools with rsync -a to the same
# paths and runs the same commands over ssh. The script is safe to run again
# from the start: each run writes a new output directory, and a preempted run
# leaves only its own directory and no container (trap).
set -u
NODE=${LEASE_NODE:?run under gpu-lease-any}
WT=/models/usman/qwen38-flash-wt/kern-l7
WT6=/models/usman/qwen38-flash-wt/kern-decode
H=/models/usman/qwen38-flash-wt/fx-hc12/tools/hc12
KT=/models/usman/kern-decode/tools
MAXS=${LEASE_MAX_S:-3300}
if [[ "$NODE" == spark2 ]]; then HUB=/home/usman/.cache/huggingface/hub; USE="SPARK2_USE=1"; else HUB=/models/usman/hf/hub; USE="SPARK_USE=1"; fi
OUT=/models/usman/kern-decode/runs/kern3-micro-$NODE-$(date +%Y%m%dT%H%M%S)
mkdir -p "$OUT"; L=$OUT/run.log
log() { echo "[$(date +%T)] $*" | tee -a "$L"; }
NAME=kern3-micro-$$
if [[ "$NODE" == "$(hostname -s)" ]]; then RUN=(bash -c); else RUN=(ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=8 "$NODE"); fi
cleanup() { "${RUN[@]}" "docker rm -f $NAME >/dev/null 2>&1" ; }
trap cleanup EXIT INT TERM
T0=$(date +%s)
if [[ "$NODE" == spark1 && ! -e /models/usman/qwen38-flash/.serve-hold ]]; then
  log "spark1: .serve-hold is missing (Codex can serve): no GPU work"; exit 3
fi
if [[ "$NODE" != "$(hostname -s)" ]]; then
  for d in "$WT/files/kern" "$WT/tools/l7" "$WT6/files/kern" "$H" "$KT"; do
    ssh "$NODE" mkdir -p "$d" && rsync -a --exclude orig --exclude out "$d/" "$NODE:$d/" || { log "rsync $d to $NODE failed"; exit 4; }
  done
  ssh "$NODE" mkdir -p "$OUT"
  ssh "$NODE" test -d "$HUB/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4" || log "no checkpoint on $NODE: hc12 section will fail"
fi
"${RUN[@]}" "docker ps --format '{{.Names}}' | grep -q vllm-fn-tp1" && { log "a vllm server is up on $NODE: stop"; exit 1; }
MA=$("${RUN[@]}" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo")
(( MA >= 40 )) || { log "MemAvailable ${MA} GiB < 40 on $NODE: stop"; exit 5; }
"${RUN[@]}" "test -f $WT/files/kern/hostalloc/kern_hostalloc.so || bash $WT/files/kern/hostalloc/build.sh" >>"$L" 2>&1
log "kern3 micro on $NODE ($USE), kern-l7 $(git -C $WT rev-parse --short HEAD), kern-decode $(git -C $WT6 rev-parse --short HEAD), fx-hc12 $(git -C $H rev-parse --short HEAD), MemAvailable ${MA} GiB"
log "probe before: $("${RUN[@]}" "bash /models/usman/qwen38-tune/probe.sh 2>&1 | tail -1")"
cp "$WT/tools/l7/l7_micro.py" "$OUT/"
[[ "$NODE" != "$(hostname -s)" ]] && rsync -a "$OUT/" "$NODE:$OUT/"
section() {  # section <name> <timeout_s> [ENV=VAL ...]
  local sec=$1 tmo=$2; shift 2
  local now=$(( $(date +%s) - T0 ))
  if (( now + tmo > MAXS )); then log "skip $sec: $now s used, $tmo s needed, budget $MAXS s"; return; fi
  local envs="" tag=$sec
  for e in "$@"; do envs+=" -e $e"; tag+="-${e#*=}"; done
  log "section $tag (timeout $tmo s)"
  "${RUN[@]}" "timeout $tmo docker run --rm --gpus all --name $NAME --entrypoint python3 $envs \
    -e $USE -e TRITON_CACHE_DIR=/triton-cache -v /models/usman/triton-cache:/triton-cache \
    -e KERN_HOSTALLOC_SO=/k/hostalloc/kern_hostalloc.so \
    -v $WT/files/kern:/k:ro -v $WT6/files/kern:/k6:ro -v $KT:/t:ro -v $H:/h:ro -v $HUB:/hub:ro -e HC12_HUB=/hub \
    -v $OUT:/o vllm/vllm-openai:qwen38-flash-next /o/l7_micro.py --only $sec --out /o/$tag.json 2>&1 \
    | grep -v '^INFO\|^WARNING\|^W0'" | tee -a "$L"
  "${RUN[@]}" "docker rm -f $NAME >/dev/null 2>&1"
}
# Order: highest expected value first, so a short lease still decides the top rungs.
section h1 420 KERN_HOSTALLOC_MODE=host
section s1 900
section hc12 700
section r6t 500
section s3 150
section s2 600
section h1 360 KERN_HOSTALLOC_MODE=vmm_dev
section h1 360 KERN_HOSTALLOC_MODE=vmm_host
log "probe after: $("${RUN[@]}" "bash /models/usman/qwen38-tune/probe.sh 2>&1 | tail -1")"
[[ "$NODE" != "$(hostname -s)" ]] && rsync -a "$NODE:$OUT/" "$OUT/"
python3 "$WT/tools/l7/make_tiles.py" "$OUT" >>"$L" 2>&1 && log "tiles: $OUT/l7_tiles.json"
log "DONE $OUT"
