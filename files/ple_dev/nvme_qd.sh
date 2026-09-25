#!/usr/bin/env bash
# Run nvme_qd in REPS repetitions. Each repetition starts only when the
# NVMe device reads less than QUIET_MBS MB/s for 2 s. The log records the
# other read traffic during each repetition.
#   nvme_qd.sh <out-dir> [reps]
set -euo pipefail
out=$1; reps=${2:-3}
here=$(cd "$(dirname "$0")" && pwd)
T=/models/usman/vllm-ple-cache/Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/language_model.model.layers.1.ple.ple_embedding.ngram_embedding.packed_u8
dev=${DEV:-nvme0n1}; quiet=${QUIET_MBS:-20}; trials=${TRIALS:-400}
mkdir -p "$out"
gcc -O2 -o "$out/nvme_qd" "$here/nvme_qd.c"
sect() { awk -v d="$dev" '$3==d {print $6}' /proc/diskstats; }
for r in $(seq 1 "$reps"); do
  while :; do
    a=$(sect); sleep 2; b=$(sect)
    mbs=$(( (b - a) * 512 / 2 / 1000000 ))
    [ "$mbs" -lt "$quiet" ] && break
    echo "$(date -u +%FT%TZ) rep $r wait: device reads ${mbs} MB/s" >>"$out/run.log"
    sleep 20
  done
  s0=$(sect); t0=$(date -u +%FT%TZ)
  "$out/nvme_qd" "$T" $((1000 + r)) "$trials" 14 30 87 117 448 >"$out/rep$r.jsonl"
  s1=$(sect)
  echo "$(date -u +%FT%TZ) rep $r done (start $t0, quiet before ${mbs} MB/s, device read $(( (s1 - s0) * 512 / 1000000 )) MB during the rep)" >>"$out/run.log"
done
