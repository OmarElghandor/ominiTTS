#!/bin/sh
set -e

MODEL_STORE_DIR="${MODEL_STORE_DIR:-/data/omnivoice-model}"
export MODEL_STORE_DIR
mkdir -p "$MODEL_STORE_DIR"

# Directory ownership for appuser. Recursive chown of multi-GB weights is slow on
# serverless cold start — use SKIP_RECURSIVE_CHOWN=1 and world-readable files instead.
chown appuser:appuser "$MODEL_STORE_DIR" 2>/dev/null || true
if [ "${SKIP_RECURSIVE_CHOWN:-0}" = "1" ]; then
  chmod -R a+rX "$MODEL_STORE_DIR" 2>/dev/null || true
else
  chown -R appuser:appuser "$MODEL_STORE_DIR" 2>/dev/null || true
fi
echo "MODEL_STORE_DIR=${MODEL_STORE_DIR}"

# One-time volume seeding: set BOOTSTRAP_ONLY=1, start the container once, wait for
# "Bootstrap complete." in logs, then remove the variable, then restart normally.
if [ "${BOOTSTRAP_ONLY:-0}" = "1" ]; then
  echo "BOOTSTRAP_ONLY=1 — downloading weights into ${MODEL_STORE_DIR} (API will not start)"
  exec gosu appuser python /app/scripts/bootstrap_model.py
fi

exec gosu appuser "$@"
