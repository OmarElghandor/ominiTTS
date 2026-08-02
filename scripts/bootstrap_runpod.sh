#!/usr/bin/env bash
# One-shot OmniVoice weight bootstrap for RunPod Network Volumes.
#
# Run this INSIDE a Pod (or any machine) that has the Network Volume mounted,
# in the SAME datacenter as your Serverless endpoint.
#
# Usage:
#   bash scripts/bootstrap_runpod.sh
#   bash scripts/bootstrap_runpod.sh /custom/path/omnivoice-model
#   IMAGE=youruser/omnivoice:latest bash scripts/bootstrap_runpod.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

detect_store_dir() {
  if [[ -n "${1:-}" ]]; then
    echo "$1"
    return
  fi
  if [[ -n "${MODEL_STORE_DIR:-}" ]]; then
    echo "$MODEL_STORE_DIR"
    return
  fi
  # Serverless workers mount the Network Volume here.
  if [[ -d /runpod-volume ]]; then
    echo "/runpod-volume/omnivoice-model"
    return
  fi
  # GPU Pods typically attach the volume at /workspace.
  if [[ -d /workspace ]]; then
    echo "/workspace/omnivoice-model"
    return
  fi
  # Local / fallback
  echo "${ROOT}/model-store"
}

STORE_DIR="$(detect_store_dir "${1:-}")"
mkdir -p "$STORE_DIR"

echo "=============================================="
echo " OmniVoice RunPod bootstrap"
echo "=============================================="
echo "Detected / chosen MODEL_STORE_DIR: $STORE_DIR"
echo

if [[ -d /runpod-volume ]]; then
  echo "Volume mount detected: /runpod-volume  (Serverless-style)"
elif [[ -d /workspace ]]; then
  echo "Volume mount detected: /workspace  (Pod-style)"
  echo "Serverless will see the same files as /runpod-volume/... when this volume is attached."
else
  echo "WARNING: neither /runpod-volume nor /workspace found."
  echo "If this is not a RunPod host, weights will only land at: $STORE_DIR"
fi
echo

# Prefer Docker image bootstrap when Docker + image are available.
IMAGE="${IMAGE:-}"
if [[ -z "$IMAGE" ]] && command -v docker >/dev/null 2>&1; then
  # Best-effort: use a locally tagged image if present
  if docker image inspect omnivoice-serverless:latest >/dev/null 2>&1; then
    IMAGE="omnivoice-serverless:latest"
  elif docker image inspect omnivoice-service:latest >/dev/null 2>&1; then
    IMAGE="omnivoice-service:latest"
  fi
fi

run_via_docker() {
  local img="$1"
  echo "Bootstrapping via Docker image: $img"
  docker run --rm --gpus all \
    -e MODEL_STORE_DIR=/data/omnivoice-model \
    -e HF_HUB_OFFLINE=0 \
    -e TRANSFORMERS_OFFLINE=0 \
    -e BOOTSTRAP_ONLY=1 \
    ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
    -v "${STORE_DIR}:/data/omnivoice-model" \
    "$img"
}

run_via_python() {
  echo "Bootstrapping via local Python (scripts/bootstrap_model.py)"
  export MODEL_STORE_DIR="$STORE_DIR"
  export HF_HUB_OFFLINE=0
  export TRANSFORMERS_OFFLINE=0
  if [[ -f requirements-bootstrap.txt ]]; then
    python3 -m pip install -q -r requirements-bootstrap.txt
  fi
  python3 scripts/bootstrap_model.py
}

if [[ -n "$IMAGE" ]] && command -v docker >/dev/null 2>&1; then
  run_via_docker "$IMAGE"
else
  run_via_python
fi

echo
echo "=============================================="
echo " Verify flat layout"
echo "=============================================="
ls -la "$STORE_DIR" | head -40
echo
du -sh "$STORE_DIR"

MISSING=0
for f in config.json model.safetensors tokenizer.json .omnivoice-bootstrap-complete; do
  if [[ ! -e "$STORE_DIR/$f" ]]; then
    echo "MISSING: $STORE_DIR/$f"
    MISSING=1
  fi
done
if [[ ! -d "$STORE_DIR/audio_tokenizer" ]]; then
  echo "MISSING: $STORE_DIR/audio_tokenizer/"
  MISSING=1
fi

if [[ "$MISSING" -ne 0 ]]; then
  echo
  echo "Bootstrap finished but required files are missing. Re-run or check HF network access."
  exit 1
fi

# Map Pod path → Serverless env hint
SERVERLESS_DIR="$STORE_DIR"
if [[ "$STORE_DIR" == /workspace/omnivoice-model ]]; then
  SERVERLESS_DIR="/runpod-volume/omnivoice-model"
fi

echo
echo "=============================================="
echo " Bootstrap OK — set these on the Serverless endpoint, then restart"
echo "=============================================="
cat <<EOF
MODEL_STORE_DIR=${SERVERLESS_DIR}
DEVICE=cuda:0
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
EOF
echo
echo "Then restart / redeploy the worker. Logs should show load + warmup, not empty store."
