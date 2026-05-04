#!/usr/bin/env bash
# Run eval_uncertainty_thresholds.py: find best uncertainty thresholds on val/test (using training data path for ground truth).

set -euo pipefail

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
WORKDIR="${WORKSPACE:-$(cd -P "$SCRIPT_DIR/.." && pwd)}"
TRAINPATH="${TRAINPATH:-${MVS_TRAINING:-${DATA_PATH:-/workspace/data}/mvs_training/dtu}}"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKDIR"
cd "$WORKDIR" || exit

python eval_uncertainty_thresholds.py \
    --dataset=dtu_yao \
    --trainpath="$TRAINPATH" \
    --vallist=lists/dtu/val.txt \
    --testlist=lists/dtu/test.txt \
    --batch_size=1 \
    --inverse_depth=False \
    --origin_size=False \
    --max_h=600 \
    --max_w=800 \
    --image_scale=0.25 \
    --light_idx=3 \
    --view_num=7 \
    --numdepth=64 \
    --interval_scale=3.18 \
    --loadckpt=./data/trained/best_model.ckpt \
    --outdir=./uncertainty_eval \
    --evidential_method=der \
    --n_thresh=100
