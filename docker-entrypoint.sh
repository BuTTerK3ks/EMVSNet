#!/bin/bash
set -eo pipefail

# Set default paths if not provided (data under /workspace/data so TRAINPATH defaults work)
export DATA_PATH="${DATA_PATH:-/workspace/data}"
export OUTPUT_PATH="${OUTPUT_PATH:-/workspace/output}"
export WORKSPACE="${WORKSPACE:-/workspace}"
export TRAINPATH="${TRAINPATH:-$DATA_PATH/mvs_training/dtu}"

# Change to workspace directory
cd "$WORKSPACE"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"

# Print environment information
echo "=========================================="
echo "Docker Container Environment"
echo "=========================================="
echo "Python: $(python --version 2>&1)"
if python -c "import torch" 2>/dev/null; then
    echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
    echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "CUDA Version: $(python -c 'import torch; print(torch.version.cuda)')"
        echo "GPU Count: $(python -c 'import torch; print(torch.cuda.device_count())')"
    else
        echo "CUDA Version: N/A"
        echo "GPU Count: 0"
    fi
else
    echo "PyTorch: Not available"
    echo "CUDA Available: False"
fi
echo "Data Path: $DATA_PATH"
echo "Output Path: $OUTPUT_PATH"
echo "Workspace: $WORKSPACE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all}"
echo "=========================================="

# Check if data directory exists
if [ ! -d "$DATA_PATH" ]; then
    echo "WARNING: Data directory $DATA_PATH does not exist!"
    echo "Please ensure data is mounted at $DATA_PATH"
fi

# Export paths for Python scripts
export PYTHONPATH="${PYTHONPATH:-$WORKSPACE}"
if [ -n "${PYTHONPATH:-}" ] && [ "$PYTHONPATH" != "$WORKSPACE" ]; then
    export PYTHONPATH="$WORKSPACE:$PYTHONPATH"
else
    export PYTHONPATH="$WORKSPACE"
fi

# Handle signal forwarding for graceful shutdown
trap 'echo "Received signal, shutting down..."; exit 0' SIGTERM SIGINT

# Execute the command passed to the container
echo "Executing: $@"
exec "$@"
