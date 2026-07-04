#!/bin/sh
set -e

# MODEL_STORE_DIR explicit > Railway auto-injected mount path > default
MODEL_STORE_DIR="${MODEL_STORE_DIR:-${RAILWAY_VOLUME_MOUNT_PATH:-/data/omnivoice-model}}"
export MODEL_STORE_DIR
mkdir -p "$MODEL_STORE_DIR"
chown -R appuser:appuser "$MODEL_STORE_DIR"

# One-time volume seeding on Railway: set BOOTSTRAP_ONLY=1, deploy, wait for success
# logs, remove the variable, then redeploy normally. Volumes are only mounted at runtime,
# so bootstrap cannot run via `railway run` from your laptop.
if [ "${BOOTSTRAP_ONLY:-0}" = "1" ]; then
  echo "BOOTSTRAP_ONLY=1 — downloading weights into ${MODEL_STORE_DIR} (API will not start)"
  exec gosu appuser python /app/scripts/bootstrap_model.py
fi

exec gosu appuser "$@"
