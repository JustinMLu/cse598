# Official NVIDIA JAX container (CUDA 12.8, Ubuntu 24.04.1 LTS)
FROM nvcr.io/nvidia/jax:25.04-py3

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Mujoco + JAX env variables
ENV MUJOCO_GL=egl
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_FLAGS="--xla_gpu_triton_gemm_any=True"


# Update apt inside container
RUN apt-get update

# Install dependencies & OpenGL libraries

# REPLACEMENTS FOR BLACKWELL:
# libgl1-mesa-glx   ->   libgl1 & libglx-mesa0
# libel1-mesa       ->   libegl1

RUN apt-get install -y \
    ffmpeg \
    git \
    wget \
    libgl1 \
    libglx-mesa0 \
    libgl1-mesa-dev \
    libegl1 \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglfw3 \
    libglfw3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip before we proceed
RUN python3 -m pip install --upgrade pip

# Install Python packages
RUN pip install \
    numpy \
    matplotlib \
    mediapy \
    mujoco==3.3.4 \
    mujoco_mjx==3.3.4 \
    orbax-checkpoint \
    etils \
    ml_collections

# Install brax very, very carefully
RUN pip install --no-dependencies \
    brax \
    jaxopt \
    jinja2 \
    markupsafe

# Install Jupyter & expose port
RUN pip install jupyterlab
EXPOSE 8888

# Copy into container
WORKDIR /lab02-playground-intro
COPY . /lab02-playground-intro/