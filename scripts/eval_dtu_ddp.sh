#!/usr/bin/env bash
# Script to run eval.py DTU with DDP (2 GPUs)
# Based on VS Code launch configuration: "Python: eval.py DTU (DDP - 2 GPUs)"

# Portable paths: WORKSPACE from env or repo root; TESTPATH for DTU test set (container-safe default)
SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
WORKSPACE_DIR="${WORKSPACE:-$(cd -P "$SCRIPT_DIR/.." && pwd)}"
TESTPATH="${TESTPATH:-${DATA_PATH:-/workspace/data}/dtu_test}"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PYTHONPATH:-$WORKSPACE_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

cd "$WORKSPACE_DIR" || exit

# Run evaluation with DDP
python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29501 \
    eval.py \
    --dataset=data_eval_transform \
    --batch_size=1 \
    --inverse_depth=False \
    --numdepth=64 \
    --interval_scale=3.18 \
    --max_h=600 \
    --max_w=800 \
    --image_scale=0.25 \
    --view_num=7 \
    --testpath="$TESTPATH" \
    --testlist=lists/dtu/test.txt \
    --loadckpt=./checkpoints/model_dtu_v2.ckpt \
    --outdir=./outputs_dtu \
    --evidential_method=der
