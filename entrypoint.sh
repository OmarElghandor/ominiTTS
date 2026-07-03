#!/bin/sh
set -e

MODEL_STORE_DIR="${MODEL_STORE_DIR:-/data/omnivoice-model}"
mkdir -p "$MODEL_STORE_DIR"
chown -R appuser:appuser "$MODEL_STORE_DIR"
exec gosu appuser "$@"
