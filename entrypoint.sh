#!/bin/sh
set -e

# MODEL_STORE_DIR explicit > Railway auto-injected mount path > default
MODEL_STORE_DIR="${MODEL_STORE_DIR:-${RAILWAY_VOLUME_MOUNT_PATH:-/data/omnivoice-model}}"
export MODEL_STORE_DIR
mkdir -p "$MODEL_STORE_DIR"
chown -R appuser:appuser "$MODEL_STORE_DIR"
exec gosu appuser "$@"
