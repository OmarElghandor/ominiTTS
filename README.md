# OmniVoice TTS Service

Standalone HTTP microservice wrapping [OmniVoice](https://github.com/k2-fsa/OmniVoice) — a zero-shot, 600+ language voice cloning TTS model — for use in **Langify** (AI-first English learning app for Arabic speakers).

This service is the **self-hosted tier** in Langify's TTS architecture:

| Tier | Provider | Use case |
|------|----------|----------|
| Static / template audio | ElevenLabs | Pre-generated lesson content |
| Dynamic / LLM content | Azure TTS | Real-time lesson narration |
| Self-hosted (scale path) | **This service** | Cost control at scale (Groq API → RunPod → dedicated GPU) |

**This service is NOT wired into the Langify Node/Express backend yet.** A follow-up task will add it as a provider option in the backend TTS router.

---

## Architecture: offline runtime, one-time bootstrap

Production containers **never contact huggingface.co**. Weights are downloaded once via `scripts/bootstrap_model.py` into a persistent volume (`MODEL_STORE_DIR`), then the API loads exclusively from that local path with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

The bootstrap script uses `snapshot_download(local_dir=...)` so files land **flat** at the volume root (`config.json`, `model.safetensors`, `tokenizer.json`, `audio_tokenizer/`, etc.). Do **not** use `cache_dir` — that creates a nested `models--k2-fsa--OmniVoice/snapshots/...` layout that `OmniVoice.from_pretrained(local_path)` cannot read.

```
  [One-time]  bootstrap_model.py  ──►  MODEL_STORE_DIR (volume)
                                              │
  [Runtime]   FastAPI / OmniVoice   ◄─────────┘  (offline only)
```

Re-run bootstrap only when intentionally upgrading to a newer OmniVoice checkpoint — not on every deploy.

---

## Quick start (local)

### Prerequisites

- Docker and Docker Compose
- (Recommended) NVIDIA GPU + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for usable inference latency

### Setup

```bash
cd omnivoice-service
cp .env.example .env
```

### 1. Bootstrap model weights (one-time)

Create a host directory for weights (compose bind-mounts it into the container):

```bash
mkdir -p ./model-store
```

Set in `.env` (or export for one-off commands):

```
OMNIVOICE_VOLUME_HOST_PATH=./model-store
```

Seed weights before starting the API:

```bash
# Bootstrap into the compose bind mount via a one-off container
docker compose run --rm \
  -e HF_HUB_OFFLINE=0 \
  -e TRANSFORMERS_OFFLINE=0 \
  -e MODEL_STORE_DIR=/data/omnivoice-model \
  omnivoice-api \
  python scripts/bootstrap_model.py

# Or bootstrap directly to a local directory (no Docker)
pip install -r requirements-bootstrap.txt
MODEL_STORE_DIR=./model-store python scripts/bootstrap_model.py
```

If Hugging Face is slow or blocked from your network, set `HF_ENDPOINT` (e.g. `https://hf-mirror.com`) for the bootstrap step only.

On success, bootstrap prints total on-disk size (~3.27 GB) and a top-level file listing. It verifies `config.json`, `model.safetensors`, tokenizer files, and `audio_tokenizer/{config.json, model.safetensors}` before declaring success. Re-running against a valid store is safe — it skips the download and reprints the summary.

### 2. Run the API

```bash
docker compose up --build
```

With hot reload for app code changes:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

### GPU

`docker-compose.yml` enables GPU reservation by default (RunPod pods and local dev with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)). Set in `.env`:

```
DEVICE=cuda:0
```

For CPU-only testing, set `DEVICE=cpu`. If CUDA is requested but unavailable, the service logs a warning and falls back to CPU automatically.

### Health checks

| Endpoint | Meaning |
|----------|---------|
| `GET /healthz` | Process is up (returns 200 even while model is loading) |
| `GET /readyz` | Model finished loading (503 until ready) |
| `POST /v1/tts/*` | TTS synthesis (no auth — internal/trusted network only) |

```bash
curl http://localhost:8080/healthz
curl http://localhost:8080/readyz
```

Wait for `/readyz` to return 200 before sending TTS requests. First boot after bootstrap loads multi-GB weights from the volume into memory/VRAM.

Interactive API docs: `http://localhost:8080/docs`

---

## API contract

All endpoints are unauthenticated — intended for internal service-to-service use on a trusted network (e.g. Langify backend or RunPod proxy with access restricted upstream).

Successful TTS responses return **24 kHz mono WAV** (`Content-Type: audio/wav`).

All error responses (non-audio) return JSON with a `message` field.

### `POST /v1/tts/clone` — Voice cloning

Clone a voice from a short reference clip (3–10 s recommended).

**JSON body:**

```json
{
  "text": "Hello, this is a cloned voice.",
  "ref_audio": "<base64-encoded audio>",
  "ref_text": "Transcript of the reference audio.",
  "language": "en",
  "num_step": 32,
  "speed": 1.0,
  "duration": null
}
```

**Multipart form:** fields `text`, `ref_audio` (file), optional `ref_text`, `language`, `num_step`, `speed`, `duration`.

If `ref_text` is omitted, OmniVoice auto-transcribes via Whisper (slower — a warning is logged server-side).

### `POST /v1/tts/design` — Voice design

Synthesize with a described voice (no reference audio).

```json
{
  "text": "Hello, this is a designed voice.",
  "instruct": "female, young adult, high pitch, british accent",
  "language": "en",
  "num_step": 32,
  "speed": 1.0
}
```

`instruct` is a comma-separated attribute string per [OmniVoice voice design docs](https://github.com/k2-fsa/OmniVoice/blob/master/docs/voice-design.md).

### `POST /v1/tts/auto` — Auto voice

No reference audio or voice description — the model picks a voice.

```json
{
  "text": "This sentence uses an automatically chosen voice.",
  "language": "en",
  "num_step": 32,
  "speed": 1.0
}
```

### Generation parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_step` | `32` | Diffusion steps (`16` for faster inference) |
| `speed` | `1.0` | Speech rate multiplier |
| `duration` | `null` | Fixed output length in seconds (overrides `speed`) |
| `language` | `null` | Optional language hint (e.g. `"en"`) |

### Example (curl)

```bash
curl -X POST http://localhost:8080/v1/tts/auto \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from OmniVoice."}' \
  --output out.wav
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `k2-fsa/OmniVoice` | HF repo id — **bootstrap script only**, not used at API runtime |
| `MODEL_STORE_DIR` | `/data/omnivoice-model` | Container path to self-hosted weights (must match compose volume mount target) |
| `OMNIVOICE_VOLUME_HOST_PATH` | `/workspace/omnivoice-model` | Host path for compose bind mount (RunPod Network Volume; set `./model-store` locally) |
| `DEVICE` | `cuda:0` | Inference device (`cpu` for CPU-only testing) |
| `DTYPE` | auto | `float16` on GPU, `float32` on CPU (override optional) |
| `HF_HUB_OFFLINE` | `1` | Must stay `1` in production — blocks HF network access |
| `TRANSFORMERS_OFFLINE` | `1` | Must stay `1` in production — blocks transformers downloads |
| `MAX_TEXT_LENGTH` | `500` | Max input text length |
| `PORT` | `8080` | HTTP port |

See [`.env.example`](.env.example) for a template.

---

## RunPod deployment (production GPU)

**Active deployment target.** OmniVoice requires GPU for production latency; RunPod persistent GPU Pods with Network Volumes replace the archived Railway CPU attempt.

Full step-by-step guide: [`deploy/runpod/README.md`](deploy/runpod/README.md)

### Summary

1. Create a **Network Volume** (≥ 20 GB) and a **GPU Pod** in the same datacenter; attach the volume (default host mount `/workspace`).
2. Expose **HTTP port 8080** in Pod settings.
3. SSH into the Pod, clone the repo, `cp .env.example .env`, set `OMNIVOICE_VOLUME_HOST_PATH=/workspace/omnivoice-model`.
4. Verify CUDA: `nvidia-smi` on host; after `docker compose up`, `torch.cuda.is_available()` must be `True` inside the container.
5. Bootstrap weights once via `docker compose run --rm ... python scripts/bootstrap_model.py`.
6. Start the API: `docker compose up --build -d`.
7. Access via RunPod proxy: `https://[POD_ID]-8080.proxy.runpod.net`

### Verification checklist

See [`deploy/runpod/README.md`](deploy/runpod/README.md) §7 for the full checklist:

1. Host GPU visible via `nvidia-smi`; driver CUDA version compatible with Dockerfile (`nvidia/cuda:12.8.0`)
2. `torch.cuda.is_available()` → `True` inside container
3. Bootstrap complete (~3.27 GB, flat layout)
4. `GET /readyz` → 200; logs show `device=cuda:0`
5. `POST /v1/tts/auto` → playable WAV with GPU latency
6. External reachability via RunPod proxy URL

---

## Railway deployment (archived — CPU-only reference)

> **Not the active deployment target.** Preserved for troubleshooting history from the CPU integration attempt. Config archived at [`deploy/railway/`](deploy/railway/). Production GPU → RunPod (above).

Railway built from the [`Dockerfile`](Dockerfile) via [`deploy/railway/railway.json`](deploy/railway/railway.json) — not Nixpacks/Railpack auto-detection.

### GPU requirement (critical)

OmniVoice is **not practically usable at production latency on CPU-only compute**. Per Langify's scale path, production GPU belongs on RunPod serverless or a dedicated GPU host when Railway cannot meet latency needs.

**As of Railway's current documentation, GPU instances are not available** on the platform ([Railway guides](https://docs.railway.com/guides/ai-agent-workers) state CPU-only). Deploy here for **API validation and integration testing on CPU only**, or migrate to a GPU provider for production inference.

<!-- TODO: verify against Railway docs if/when GPU plans become generally available -->

### Steps

1. **Create a Railway project** and connect this repository.
2. Set the **root directory** to `omnivoice-service/` if the repo contains other projects.
3. **Attach a persistent volume** ([Railway volumes guide](https://docs.railway.com/guides/volumes)):
   - Command Palette (`⌘K` / `Ctrl+K`) → **Create Volume**
   - Attach to the OmniVoice service
   - Mount path: `/data/omnivoice-model` (recommended) or any absolute path — if you omit `MODEL_STORE_DIR`, the service auto-uses Railway's `RAILWAY_VOLUME_MOUNT_PATH`
4. **Seed the volume once** (before the API can start — empty or invalid `MODEL_STORE_DIR` fails fast).

   Railway volumes are only mounted **inside the running container**, so bootstrap must run there — not via `railway run` on your laptop.

   **Option A — `BOOTSTRAP_ONLY` deploy (required for first-time seeding):**

   1. In the Railway dashboard → **ominiTTS** → **Variables**, add:
      ```
      BOOTSTRAP_ONLY=1
      ```
   2. **Deploy** (or redeploy). Watch **Deploy logs** until you see **`Bootstrap complete.`** with on-disk size **~3 GB** and top-level files (`config.json`, `model.safetensors`, `audio_tokenizer/`, etc.).
      - `Fetching 13 files: 100%` is **not** the finish line — wait for **`Bootstrap complete.`**
      - The deploy may show **Crashed/Exited** after bootstrap — that is normal; the volume data persists
   3. **Remove** `BOOTSTRAP_ONLY` from variables.
   4. **Redeploy** again — the API starts normally and loads from the volume.

   **Option B — `railway ssh` (only when the API container stays running):**

   ```bash
   railway ssh --service ominiTTS -- python /app/scripts/bootstrap_model.py
   ```

   If SSH says *"container is not running"*, use Option A instead.

   Bootstrap requires network access. The script clears offline flags internally.

   Re-run bootstrap only when upgrading checkpoints or repairing a broken/partial download, not on normal deploys.

5. **Set environment variables** in the Railway dashboard:

   | Variable | Value |
   |----------|-------|
   | `MODEL_STORE_DIR` | *(optional)* omit — auto-uses `RAILWAY_VOLUME_MOUNT_PATH` from the attached volume |
   | `DEVICE` | `cpu` (until GPU plans exist elsewhere use RunPod) |
   | `HF_HUB_OFFLINE` | `1` |
   | `TRANSFORMERS_OFFLINE` | `1` |
   | `RAILWAY_RUN_UID` | `0` (required — volumes mount as root; entrypoint chowns then drops to `appuser`) |
   | `BOOTSTRAP_ONLY` | *(one-time only)* `1` — run bootstrap on deploy instead of starting the API; remove after volume is seeded |

6. **Deploy.** `healthcheckTimeout` is 600 s in [`deploy/railway/railway.json`](deploy/railway/railway.json) for cold model load. Poll `/readyz` before routing traffic.

### Post-deploy verification

1. **Bootstrap** (if not already done, or to repair a broken volume):
   - Set `BOOTSTRAP_ONLY=1`, deploy, confirm **Bootstrap complete** (~3.27 GB), remove the variable, redeploy; **or**
   - If the service is running: `railway ssh --service ominiTTS -- python /app/scripts/bootstrap_model.py`
   Expect flat top-level files — no `models--*` nesting.
2. **Redeploy** `omnivoice-api`.
3. **Logs** — look for the **offline mode active** line, **no** `huggingface_hub` download activity, and successful model load (no `OSError` about missing `model.safetensors`).
4. **`GET /readyz`** — should return 200 once load completes.
5. **End-to-end TTS**:
   ```bash
   curl -X POST https://<your-railway-host>/v1/tts/auto \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello from OmniVoice."}' \
     --output out.wav
   ```
   Confirm `out.wav` is playable before declaring the deployment done.

### Volume mount path

**Existing volume at `/data/huggingface`:** you do not need to remount immediately. Railway injects `RAILWAY_VOLUME_MOUNT_PATH=/data/huggingface` at runtime; the entrypoint, bootstrap script, and API config use that path automatically when `MODEL_STORE_DIR` is unset.

**Recommended (new deployments):** mount at `/data/omnivoice-model` and optionally set `MODEL_STORE_DIR=/data/omnivoice-model` explicitly.

**To change an existing volume mount path** (e.g. migrate `/data/huggingface` → `/data/omnivoice-model`):
1. Service → **Settings** → **Volumes** → edit mount path (or `railway volume update --volume <name> --mount-path /data/omnivoice-model`)
2. Re-run bootstrap (`BOOTSTRAP_ONLY=1` deploy, or `railway ssh` if the service is up) — data at the old path is not visible after remounting

**Deploy error:** *"requires a volume to be mounted at /data/omnivoice-model"* — caused by a stale `requiredMountPath` in an older `railway.json`. Current archived config no longer enforces a fixed path; redeploy after pulling latest, or update your volume mount / `MODEL_STORE_DIR` to match.

### Troubleshooting: common error loop on Railway

| Symptom | Cause | Fix |
|---------|-------|-----|
| `OSError: no file named model.safetensors` | Volume never seeded or wrong layout (`models--*` cache nesting) | `BOOTSTRAP_ONLY=1` deploy; wait for **`Bootstrap complete.`** ~3 GB |
| `missing: config.json; missing: model.safetensors; ...` | Bootstrap interrupted before finish (removed `BOOTSTRAP_ONLY` at `100%`) | Re-add `BOOTSTRAP_ONLY=1`, redeploy, wait for **`Bootstrap complete.`** |
| `railway run python ...` → `No such file or directory` | `railway run` executes on your Mac, not in the container | Use `BOOTSTRAP_ONLY=1` or `railway ssh` |
| `railway ssh` → *container is not running* | API was crash-looping on empty volume (older builds) or bootstrap deploy exited | Use `BOOTSTRAP_ONLY=1`; after latest code, API stays up with `/readyz` 503 instead of exiting |
| Deploy stuck at *Deploying* during bootstrap | Health check targets `/healthz` but API is not started in bootstrap mode | Normal — watch **Deploy logs** for `Bootstrap complete.`, not deployment badge |

**Do not remove `BOOTSTRAP_ONLY` until deploy logs show `Bootstrap complete.` with ~3 GB on disk.**

```
  Empty volume
       │
       ▼
  API can't load model ──► /readyz 503 (load_error tells you to bootstrap)
       │
       ▼
  BOOTSTRAP_ONLY=1 deploy ──► download + verify ──► Bootstrap complete. (~3 GB)
       │
       ▼
  Remove BOOTSTRAP_ONLY ──► redeploy ──► model loads ──► /readyz 200
```

### Cold start behavior

- `/healthz` — returns 200 once the process starts (Railway health check target)
- `/readyz` — returns 503 until model load completes

Restarting/redeploying the API container makes **zero network calls to huggingface.co** — weights load only from the mounted volume.

---

## Known limitations

- **Single uvicorn worker** — the GPU model is loaded once into VRAM/RAM; multiple workers would each load a separate copy. Concurrency requires a request queue (not implemented).
- **No built-in request queueing** — concurrent requests share one model instance.
- **No authentication** — do not expose the public domain to the open internet without a gateway in front.
- **24 kHz output only** — no resampling unless the caller handles it downstream.
- **CPU inference is slow** — acceptable for testing; production needs GPU hardware.
- **Bootstrap is manual** — the API never downloads weights; an empty volume prevents startup.

---

## Project structure

```
omnivoice-service/
├── app/
│   ├── main.py              # FastAPI routes, lifespan, exception handlers
│   ├── tts.py               # OmniVoice wrapper (singleton model)
│   ├── model_store.py       # Shared MODEL_STORE_DIR layout verification
│   ├── schemas.py           # Pydantic request/response models
│   ├── config.py            # Environment-driven settings
├── scripts/
│   └── bootstrap_model.py   # One-time HF weight download (not used at API runtime)
├── deploy/
│   ├── runpod/
│   │   └── README.md        # RunPod GPU deployment guide (active)
│   └── railway/
│       ├── railway.json     # Archived Railway config
│       └── README.md
├── Dockerfile
├── docker-compose.yml
├── docker-compose.dev.yml
├── requirements.txt
├── requirements-bootstrap.txt
├── .dockerignore
├── .env.example
└── README.md
```

---

## License

OmniVoice model and library: see [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice). This wrapper service is part of the Langify project infrastructure.
