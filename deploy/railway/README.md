# Railway deployment (archived)

This directory preserves the Railway deployment config from the CPU-only integration attempt.

**Active deployment target is RunPod** — see [`../runpod/README.md`](../runpod/README.md).

The main [`README.md`](../../README.md) retains the full Railway troubleshooting history (volume seeding, `BOOTSTRAP_ONLY`, permission fixes, offline-mode fixes) for reference.

## Files

- `railway.json` — Dockerfile build config, 600s healthcheck timeout, restart policy

## Notes

- Railway required `RAILWAY_RUN_UID=0` so the entrypoint could `chown` the volume before dropping to `appuser`.
- Railway injected `RAILWAY_VOLUME_MOUNT_PATH` when a volume was attached; path resolution now uses `MODEL_STORE_DIR` only (see current `entrypoint.sh` and `app/model_store.py`).
