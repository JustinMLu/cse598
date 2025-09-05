# Lab 02 - Playground Intro

My implementation of Lab 02 that can be run locally on a GPU-equipped computer with CUDA 12+, without having to struggle with Colab and its many, many flaws

## Requirements
To run locally, the host computer must have the following installed:
- Docker
- CUDA 12.x
- NVIDIA Container Toolkit

## Installation & Usage (Docker)
### 1. Initialize Git LFS
Robot mesh assets (`.obj`) are tracked & stored using **Git Large File Storage (LFS)**, so pulling them is required before starting.

```bash
# Install Git LFS
sudo apt-get install git-lfs

# Pull assets
git lfs install
git lfs pull
```

### 2. Build the Docker image

```bash
# Will take a while to complete - don't worry!
docker compose -f docker-compose-jupyter.yml build lab02
```

### 3. Spin up the Docker Container

```bash
# Spin up container
docker compose -f docker-compose-jupyter.yml up lab02
```

### 4. Open Jupyter Notebook session in your browser
```bash
# You can also paste the URL directly into your browser
start http://localhost:8888
```

## Alternate Usages

### Directly run main.py instead of dealing with the Jupyter Notebook (WIP)
I plan to rewrite the Jupyter Notebook as a Python script in `main.py` that directly skips to using the Go2Env.
If you want to use `main.py` instead of `02_lab_student.ipynb`, replace Step 3-4 with the following:

```bash
# Step 3 - Build image
docker compose -f docker-compose-python.yml build lab02

# Step 4 - Spin up container (runs main.py)
docker compose -f docker-compose-python.yml up lab02
```

### Open a shell inside the Container (won't start Jupyter Notebook)
```bash
# Run bash inside container
docker compose -f docker-compose-jupyter.yml run --rm lab02 bash

# (Inside container): now you can manually run main.py
python3 main.py
```