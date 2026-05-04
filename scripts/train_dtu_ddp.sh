#!/usr/bin/env bash
set -euo pipefail

# Environment / defaults (portable: Docker uses WORKSPACE=/workspace; local uses repo root if unset)
SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
WORKSPACE_DIR="${WORKSPACE:-$(cd -P "$SCRIPT_DIR/.." && pwd)}"
export WORKSPACE="${WORKSPACE:-$WORKSPACE_DIR}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PYTHONPATH:-$WORKSPACE_DIR}"
# Do not override CUDA_VISIBLE_DEVICES so all GPUs are used when unset
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1

# Container-safe trainpath: in Docker use DATA_PATH or /workspace/data; override with TRAINPATH
TRAINPATH="${TRAINPATH:-${DATA_PATH:-/workspace/data}/mvs_training/dtu}"

cd "$WORKSPACE_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$WORKSPACE_DIR/data/output/train_dual/${TIMESTAMP}"
mkdir -p "$LOGDIR"

# Persist stdout/stderr (survives SSH disconnects; keeps tracebacks)
exec > >(tee -a "$LOGDIR/stdout.log") 2> >(tee -a "$LOGDIR/stderr.log" >&2)

echo "[train_dtu_ddp] start $(date -Is) on $(hostname)"
echo "[train_dtu_ddp] LOGDIR=$LOGDIR"
nvidia-smi || true

# Use all visible GPUs (or detect from nvidia-smi if CUDA_VISIBLE_DEVICES unset)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
  NGPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NGPUS=${NGPUS:-1}
echo "[train_dtu_ddp] using $NGPUS GPU(s)"
# For numdepth=128 use lower LR (e.g. --lr=0.0005) to avoid divergence after epoch 1.

python -m torch.distributed.run \
  --nproc_per_node="$NGPUS" \
  --master_port=29500 \
  train.py \
  --dataset=dtu_yao \
  --batch_size=1 \
  --trainpath="$TRAINPATH" \
  --lr=0.001 \
  --epochs=10 \
  --view_num=7 \
  --inverse_depth=False \
  --image_scale=0.25 \
  --trainlist=lists/dtu/train.txt \
  --vallist=lists/dtu/val.txt \
  --testlist=lists/dtu/test.txt \
  --numdepth=64 \
  --interval_scale=3.18 \
  --logdir="$LOGDIR" \
  --summary_freq=20 \
  --save_freq_checkpoint=1 \
  --evidential_method=sder \
  --weight_reg=0.1 \
  --early_stopping \
  --early_stopping_patience=4 \
  --early_stopping_min_delta=0.1 \
  --optimizer=adam \

echo "[train_dtu_ddp] done $(date -Is)"
