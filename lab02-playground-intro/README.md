# Lab 02 - Playground Intro

My implementation of Lab 02 that can be run locally on a GPU-equipped computer with CUDA 12+, without having to struggle with Colab and its many, many flaws

## Requirements
To run locally, the host computer must have the following installed:
- Docker
- CUDA 12.x
- NVIDIA Container Toolkit

## Installation & Usage
### Initialize Git LFS
Robot mesh assets (`.obj`) are tracked & stored using **Git LFS**, so pulling them is required before starting.

```bash
# Install Git LFS (if you don't have it)
sudo apt-get install git-lfs

# Pull assets
git lfs install
git lfs pull
```

### Build the Docker image
```bash
# Will take awhile
docker compose -f docker-compose-deploy.yml build lab02
```

### Run main.py (instead of the Jupyter Notebook)

```bash
# Default command is just 'python3 main.py'
docker compose -f docker-compose-deploy.yml up lab02
```

### Open a shell inside the Container (without running main.py)
```bash
# Run bash inside container
docker compose -f docker-compose-deploy.yml run --rm lab02 bash

# (Inside container): now you can manually run main.py
python3 main.py
```