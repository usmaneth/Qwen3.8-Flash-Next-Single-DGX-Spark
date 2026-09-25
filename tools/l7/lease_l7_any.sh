#!/usr/bin/env bash
# Run lease_l7_inlaunch.sh on the node that gpu-lease-any gave (LEASE_NODE).
#   /home/usman/bin/gpu-lease-any --nodes spark1,spark2 --timeout 43200 kern3-<name> -- \
#       bash /models/usman/qwen38-flash-wt/kern-l7/tools/l7/lease_l7_any.sh kern3-<name> <plan.json> [VAR=value ...]
# spark1: stop if .serve-hold is missing (Codex can serve), else run here
#         with SPARK_USE=1.
# spark2: sync the worktree, the plans and the tools, then run over ssh
#         with SPARK2_USE=1.
# The script is safe to run again from the start: every run writes a new run
# directory, and lease_l7_inlaunch.sh stops the server at its end.
# LEASE_MAX_S (default 3300) and PROFILE pass through.
set -u
WT=/models/usman/qwen38-flash-wt/kern-l7
node="${LEASE_NODE:-$(hostname -s)}"
LABEL=$1; PLAN=$2; shift 2
MAXS=${LEASE_MAX_S:-3300}
echo "[$(date +%T)] $LABEL on $node (plan $PLAN, env $*)"
case "$node" in
  spark1)
    if [[ ! -e /models/usman/qwen38-flash/.serve-hold ]]; then
      echo "spark1: .serve-hold is missing (Codex can serve): no boot"; exit 3
    fi
    if [[ "$node" == "$(hostname -s)" ]]; then
      exec env PROFILE="${PROFILE:-0}" LEASE_MAX_S=$MAXS bash $WT/tools/l7/lease_l7_inlaunch.sh "$LABEL" "$PLAN" SPARK_USE=1 "$@"
    fi
    exec ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=8 spark1 \
      env PROFILE="${PROFILE:-0}" LEASE_MAX_S=$MAXS bash $WT/tools/l7/lease_l7_inlaunch.sh "$LABEL" "$PLAN" SPARK_USE=1 "$@"
    ;;
  spark2)
    bash $WT/tools/l7/sync_l7_spark2.sh || { echo "sync to spark2 failed"; exit 4; }
    exec ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=8 spark2 \
      env PROFILE="${PROFILE:-0}" LEASE_MAX_S=$MAXS bash $WT/tools/l7/lease_l7_inlaunch.sh "$LABEL" "$PLAN" SPARK2_USE=1 "$@"
    ;;
  *) echo "unknown node $node"; exit 2 ;;
esac
