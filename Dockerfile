FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN nvcc --version

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages runpod==1.7.9 requests==2.32.5

WORKDIR /app
COPY rp_handler.py /app/rp_handler.py

CMD ["python3", "/app/rp_handler.py"]