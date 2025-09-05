FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# get rid of them gosh darn INTERACTIVE PROMPTS
ENV DEBIAN_FRONTEND=noninteractive

# Mujoco + JAX env variables for GPU acceleration
ENV MUJOCO_GL=egl
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_FLAGS="--xla_gpu_triton_gemm_any=True"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libegl1-mesa \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglfw3 \
    libglfw3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    xvfb \
    libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/*

# Add NVIDIA EGL driver configuration for MuJoCo rendering
RUN mkdir -p /usr/share/glvnd/egl_vendor.d/ && \
    echo '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Upgrade pip and install Python packages
RUN python3 -m pip install --upgrade pip

# Install JAX with CUDA 12 support
RUN pip install --upgrade "jax[cuda12-pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install OpenGL dependencies for MuJoCo
RUN pip install PyOpenGL PyOpenGL_accelerate

# Install core dependencies
RUN pip install \
    PyOpenGL \
    PyOpenGL_accelerate \
    numpy \
    matplotlib \
    mediapy \
    mujoco \
    mujoco_mjx \
    brax \
    flax \
    ml_collections \
    etils \
    orbax-checkpoint
    
# Install orbax-checkpoint LAST To be extra safe w/ dependency conflicts
RUN pip install orbax-checkpoint

# Set working directory
WORKDIR /lab02-playground-intro

# Copy your project files
COPY . /lab02-playground-intro/