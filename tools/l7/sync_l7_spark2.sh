#!/usr/bin/env bash
# Copy the kern-l7 worktree, the kern-decode lease tools and the plans to spark2.
# The extracted and generated image files stay out, so spark2 generates them
# from its own image. kern_hostalloc.so goes with the worktree (same image).
set -eu
WT=/models/usman/qwen38-flash-wt/kern-l7
( cd $WT && { git rev-parse HEAD; git status --short; } > .kern_head )
rsync -a --delete --exclude 'files/*.orig' --exclude 'files/*_patched.py' --exclude 'files/determinism/' \
  --exclude 'files/ple_offload/' --exclude 'files/kern/orig/' --exclude 'files/kern/out/' \
  --exclude '.last_launch.sh' --exclude '.git' --exclude '__pycache__' --exclude 'logs/' \
  $WT/ spark2:$WT/
rsync -a /models/usman/kern-decode/ab /models/usman/kern-decode/plans /models/usman/kern-decode/tools spark2:/models/usman/kern-decode/
rsync -a /models/usman/kern-decode/profile/*.py spark2:/models/usman/kern-decode/profile/
echo "synced $(head -1 $WT/.kern_head)"
