FROM python:3.11-slim

RUN pip install runpod

COPY rp_handler.py /app/rp_handler.py

CMD ["python", "/app/rp_handler.py"]