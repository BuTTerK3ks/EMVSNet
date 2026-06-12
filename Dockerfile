# Dockerfile for EMVSNet Training on Vast.ai
# Base image with CUDA 11.3 support (compatible with PyTorch 1.10.1)
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PYTHONNOUSERSITE=1

# Install Python 3.9 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3-pip \
    python3.9-distutils \
    wget \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Upgrade pip for Python 3.9
RUN python3.9 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Create pip symlink to use python3.9's pip
RUN ln -sf /usr/bin/python3.9 /usr/local/bin/pip3.9 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/python3.9 1 || true

# Install numpy first (required by PyTorch and other packages)
RUN python3.9 -m pip install --no-cache-dir numpy==1.23.5

# Install PyTorch 1.10.1 with CUDA 11.3 support (using python3.9 -m pip to ensure correct Python)
RUN python3.9 -m pip install --no-cache-dir \
    torch==1.10.1+cu113 \
    torchvision==0.11.2+cu113 \
    torchaudio==0.10.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Install core data processing libraries
RUN python3.9 -m pip install --no-cache-dir \
    opencv-python==4.10.0.84 \
    plyfile \
    Pillow

# Install visualization and logging libraries
RUN python3.9 -m pip install --no-cache-dir \
    matplotlib \
    tensorboardx==2.6.2.2 \
    tensorboard==2.14.0 \
    tqdm \
    seaborn \
    torchviz \
    graphviz

# Install statistics and analysis libraries
RUN python3.9 -m pip install --no-cache-dir \
    scikit-learn \
    scipy \
    pandas

# Set working directory
WORKDIR /workspace

# Copy entrypoint script first (before general copy to ensure it's included)
COPY docker-entrypoint.sh /workspace/docker-entrypoint.sh
RUN chmod +x /workspace/docker-entrypoint.sh

# Copy codebase
COPY . /workspace/

# Set default environment variables
ENV DATA_PATH=/workspace/data
ENV OUTPUT_PATH=/workspace/output
ENV WORKSPACE=/workspace

# Create output directory
RUN mkdir -p /workspace/output

# Set entrypoint
ENTRYPOINT ["/workspace/docker-entrypoint.sh"]

# Default command: run training script with container paths (override for eval or custom args)
CMD ["/bin/bash", "/workspace/scripts/train_dtu_ddp.sh"]
