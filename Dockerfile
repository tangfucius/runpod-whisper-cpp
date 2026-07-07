FROM ghcr.io/ggml-org/whisper.cpp:main-cuda-df7638d8229a243af8a4b5a8ae557e0d74e0a0ae

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# The upstream image uses ENTRYPOINT ["bash", "-c"].
# Reset it so RunPod can run our Python worker normally.
ENTRYPOINT []

# Install Python for the RunPod handler.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3 \
    python3-pip \
    python3-venv \
  && rm -rf /var/lib/apt/lists/*

# Use a venv instead of installing into system Python.
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python3 -m venv $VIRTUAL_ENV \
  && python -m pip install --upgrade pip setuptools wheel

WORKDIR /worker

COPY requirements.txt /worker/requirements.txt
RUN pip install --no-cache-dir -r /worker/requirements.txt

# Download the model into a stable location.
# large-v3 is about 2.9 GiB, so this layer may still take a few minutes.
RUN mkdir -p /models \
  && cd /app \
  && ./models/download-ggml-model.sh large-v3 /models \
  && test -f /models/ggml-large-v3.bin

COPY handler.py /worker/handler.py

# Your handler checks this exact env var and requires the path to exist.
ENV WHISPER_CPP_BIN=/app/build/bin/whisper-cli
ENV WHISPER_CPP_MODEL_DIR=/models
ENV WHISPER_CPP_MODEL=large-v3
ENV WHISPER_CPP_THREADS=4
ENV WHISPER_CPP_TIMEOUT_SECONDS=3600
ENV WHISPER_CPP_STALL_TIMEOUT_SECONDS=60

RUN test -x /app/build/bin/whisper-cli \
  && test -f /models/ggml-large-v3.bin

CMD ["python", "/worker/handler.py"]
