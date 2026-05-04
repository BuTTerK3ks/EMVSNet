#!/usr/bin/env bash
# Local dataset path defaults.
# Copy and adjust values to your machine if needed.

# Root where DTU is expected at: $MVS_TRAINING/dtu
export MVS_TRAINING="${MVS_TRAINING:-/path/to/mvs_training}"

# Optional blended training root
export BLEND_TRAINING="${BLEND_TRAINING:-/path/to/blendedmvs}"

# DTU evaluation root
export DTU_TESTING="${DTU_TESTING:-$MVS_TRAINING/dtu}"

# Tanks and Temples root
export TP_TESTING="${TP_TESTING:-/path/to/tanksandtemples}"