#!/usr/bin/env bash
# Build cold_fault, write a private 2 GiB data file, run the cold-page modes
# for N in 14 56 87 117, and delete the data file.
#   cold_fault.sh <out-dir> [reps]
set -euo pipefail
out=$1; reps=${2:-2}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
gcc -O2 -o "$out/cold_fault" "$here/cold_fault.c"
data="$out/private-2g.bin"
trap 'rm -f "$data"' EXIT
head -c 2147483648 /dev/urandom >"$data"
sync "$data"
python3 -c "import os; fd=os.open('$data', os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)"
for r in $(seq 1 "$reps"); do
  for n in 14 56 87 117; do
    echo "$(date -u +%FT%TZ) rep $r n $n" >>"$out/run.log"
    "$out/cold_fault" "$data" $((r * 100 + n)) "${TRIALS:-300}" "$n" 250,500,1100,pop250,pop1100 >>"$out/rep$r.jsonl"
  done
done
