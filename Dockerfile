FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

ARG WHISPER_CPP_COMMIT=df7638d8229a243af8a4b5a8ae557e0d74e0a0ae

# Confirm CUDA compiler exists
RUN nvcc --version

# System deps for building whisper.cpp + Python worker
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
  && rm -rf /var/lib/apt/lists/*

# Python virtualenv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python3 -m venv $VIRTUAL_ENV \
  && python -m pip install --upgrade pip setuptools wheel

# Minimal RunPod deps only for this test
RUN pip install --no-cache-dir \
    runpod==1.7.9 \
    requests==2.32.5

# Build whisper.cpp with CUDA, but do NOT download model yet
WORKDIR /opt

RUN git clone https://github.com/ggml-org/whisper.cpp.git whisper.cpp \
  && cd whisper.cpp \
  && git checkout ${WHISPER_CPP_COMMIT} \
  && cmake -B build -DGGML_CUDA=1 -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build --config Release -j"$(nproc)" \
  && test -x /opt/whisper.cpp/build/bin/whisper-cli

WORKDIR /app

COPY rp_handler.py /app/rp_handler.py

ENV WHISPER_CPP_BIN=/opt/whisper.cpp/build/bin/whisper-cli
ENV WHISPER_CPP_MODEL_DIR=/opt/whisper.cpp/models
ENV WHISPER_CPP_MODEL=large-v3
ENV WHISPER_CPP_THREADS=4
ENV WHISPER_CPP_TIMEOUT_SECONDS=3600

CMD ["python", "/app/rp_handler.py"]