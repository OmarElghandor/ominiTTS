#!/bin/sh
set -e

# Railway volume mount path wins when attached (see app/model_store.resolve_model_store_dir)
MODEL_STORE_DIR="${RAILWAY_VOLUME_MOUNT_PATH:-${MODEL_STORE_DIR:-/data/omnivoice-model}}"
export MODEL_STORE_DIR
mkdir -p "$MODEL_STORE_DIR"
chown -R appuser:appuser "$MODEL_STORE_DIR"
echo "MODEL_STORE_DIR=${MODEL_STORE_DIR} RAILWAY_VOLUME_MOUNT_PATH=${RAILWAY_VOLUME_MOUNT_PATH:-<unset>}"

# One-time volume seeding on Railway: set BOOTSTRAP_ONLY=1, deploy, wait for success
# logs, remove the variable, then redeploy normally. Volumes are only mounted at runtime,
# so bootstrap cannot run via `railway run` from your laptop.
if [ "${BOOTSTRAP_ONLY:-0}" = "1" ]; then
  echo "BOOTSTRAP_ONLY=1 — downloading weights into ${MODEL_STORE_DIR} (API will not start)"
  exec gosu appuser python /app/scripts/bootstrap_model.py
fi

exec gosu appuser "$@"
