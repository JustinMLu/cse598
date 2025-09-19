# CSE 598 - Action and Perception Laboratory

#### Author: ```Justin Lu | lujust@umich.edu```

My implementation of the CSE 598 labs that can be run locally on a GPU-equipped computer with CUDA 12+, without having to struggle with Colab and its many, many flaws

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

### 2. Build the Docker image for the lab yo want
Docker service names are defined in the compose .yml files - we'll use `lab02` as an example for the remainder of the README.

```bash
# First go into the correct directory
cd lab02-playground-intro

#  This will take a while to complete
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

### Open a shell inside the Container (won't start Jupyter Notebook server)
```bash
# Run bash inside container
docker compose -f docker-compose-jupyter.yml run --rm lab02 bash

# (Inside container): now you can manually run main.py
python3 main.py
```