FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG WHISPER_CPP_COMMIT=df7638d8229a243af8a4b5a8ae557e0d74e0a0ae
ARG WHISPER_MODEL=large-v3

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    python3 \
    python3-pip \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/ggml-org/whisper.cpp.git whisper.cpp \
  && cd whisper.cpp \
  && git checkout ${WHISPER_CPP_COMMIT} \
  && cmake -B build -DGGML_CUDA=1 -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build --config Release -j"$(nproc)" \
  && bash models/download-ggml-model.sh ${WHISPER_MODEL}

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt
COPY handler.py /app/handler.py

ENV WHISPER_CPP_BIN=/opt/whisper.cpp/build/bin/whisper-cli
ENV WHISPER_CPP_MODEL_DIR=/opt/whisper.cpp/models
ENV WHISPER_CPP_MODEL=large-v3
ENV WHISPER_CPP_THREADS=4
ENV WHISPER_CPP_TIMEOUT_SECONDS=3600

CMD ["python3", "/app/handler.py"]
