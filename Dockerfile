FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# Confirm CUDA compiler exists in the devel image
RUN nvcc --version

# Basic system + Python setup
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3 \
    python3-pip \
    python3-venv \
  && rm -rf /var/lib/apt/lists/*

# Use virtualenv instead of system-wide pip
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python3 -m venv $VIRTUAL_ENV \
  && python -m pip install --upgrade pip setuptools wheel

# Install minimal RunPod worker deps
RUN pip install --no-cache-dir \
    runpod==1.7.9 \
    requests==2.32.5

WORKDIR /app

COPY rp_handler.py /app/rp_handler.py

CMD ["python", "/app/rp_handler.py"]