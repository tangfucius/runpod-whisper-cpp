FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG WHISPER_CPP_COMMIT=df7638d8229a243af8a4b5a8ae557e0d74e0a0ae
ARG WHISPER_MODEL=large-v3
ARG BUILD_JOBS=4

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /opt

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
  && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ggml-org/whisper.cpp.git whisper.cpp \
  && cd whisper.cpp \
  && git checkout ${WHISPER_CPP_COMMIT} \
  && cmake -B build -DGGML_CUDA=1 -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build --config Release -j"${BUILD_JOBS}" \
  && bash models/download-ggml-model.sh ${WHISPER_MODEL}

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    python3 \
    python3-pip \
    python3-requests \
  && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/whisper.cpp /opt/whisper.cpp
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt
COPY handler.py /app/handler.py
COPY rp_handler.py /app/rp_handler.py

ENV WHISPER_CPP_BIN=/opt/whisper.cpp/build/bin/whisper-cli
ENV WHISPER_CPP_MODEL_DIR=/opt/whisper.cpp/models
ENV WHISPER_CPP_MODEL=large-v3
ENV WHISPER_CPP_THREADS=4
ENV WHISPER_CPP_TIMEOUT_SECONDS=3600

CMD ["python3", "-u", "/app/rp_handler.py"]
