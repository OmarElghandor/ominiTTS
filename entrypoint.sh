#!/bin/sh
set -e

MODEL_STORE_DIR="${MODEL_STORE_DIR:-/data/omnivoice-model}"
export MODEL_STORE_DIR
mkdir -p "$MODEL_STORE_DIR"
chown -R appuser:appuser "$MODEL_STORE_DIR"
echo "MODEL_STORE_DIR=${MODEL_STORE_DIR}"

# One-time volume seeding: set BOOTSTRAP_ONLY=1, start the container once, wait for
# "Bootstrap complete." in logs, remove the variable, then restart normally.
if [ "${BOOTSTRAP_ONLY:-0}" = "1" ]; then
  echo "BOOTSTRAP_ONLY=1 — downloading weights into ${MODEL_STORE_DIR} (API will not start)"
  exec gosu appuser python /app/scripts/bootstrap_model.py
fi

exec gosu appuser "$@"
