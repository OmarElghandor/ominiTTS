# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv && \
    pip install --upgrade pip setuptools wheel

RUN pip install \
    torch==2.8.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && \
    pip install omnivoice==0.1.5

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PORT=8080 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    libsndfile1 \
    curl \
    gosu \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

COPY requirements-bootstrap.txt /tmp/requirements-bootstrap.txt
RUN pip install -r /tmp/requirements-bootstrap.txt

WORKDIR /app
COPY app/ /app/app/
COPY api/ /app/api/
COPY scripts/ /app/scripts/
COPY handler.py /app/handler.py
COPY entrypoint.sh /entrypoint.sh

RUN useradd -m -u 1000 appuser && \
    mkdir -p /data/omnivoice-model && \
    chown -R appuser:appuser /app /data/omnivoice-model && \
    chmod +x /entrypoint.sh

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f "http://localhost:${PORT}/healthz" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]

# ---------------------------------------------------------------------------
# Serverless target — queue worker for RunPod on-demand endpoints.
# Build: docker build --platform linux/amd64 --target serverless \
#          -t <registry>/omnivoice-serverless:latest .
# Runs as non-root (appuser via entrypoint). Model weights stay on Network Volume.
# ---------------------------------------------------------------------------
FROM runtime AS serverless

ENV MODEL_STORE_DIR=/runpod-volume/omnivoice-model \
    DEVICE=cuda:0 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SKIP_RECURSIVE_CHOWN=1 \
    MAX_CONCURRENT_REQUESTS=1 \
    MAX_QUEUE_SIZE=8 \
    REQUEST_TIMEOUT=120 \
    LOG_LEVEL=INFO \
    OUTPUT_FORMAT=wav \
    SPEECH_PROVIDER=omnivoice \
    WARMUP_TEXT=Hello \
    READY_MARKER_PATH=/tmp/omnivoice-ready

# Ready after SpeechEngine warmup writes READY_MARKER_PATH (no FastAPI port).
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD test -f /tmp/omnivoice-ready || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-u", "/app/handler.py"]

# Default target for `docker build` / compose — FastAPI Pod image (must stay last).
FROM runtime
