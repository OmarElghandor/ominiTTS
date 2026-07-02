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
# Edit .env — set API_KEY to a random secret
```

### 1. Bootstrap model weights (one-time)

Seed the Docker volume before starting the API:

```bash
# Install bootstrap deps locally (or use the one-off compose command below)
pip install -r requirements-bootstrap.txt

# Option A — bootstrap into the compose volume via a one-off container (runs as root for pip)
docker compose run --rm --user root \
  -e HF_HUB_OFFLINE=0 \
  -e TRANSFORMERS_OFFLINE=0 \
  -e MODEL_STORE_DIR=/data/omnivoice-model \
  omnivoice-api \
  sh -c "pip install -r /app/requirements-bootstrap.txt && python /app/scripts/bootstrap_model.py"

# Option B — bootstrap to a local directory, then bind-mount it in compose for dev
MODEL_STORE_DIR=./model-store python scripts/bootstrap_model.py
```

If Hugging Face is slow or blocked from your network, set `HF_ENDPOINT` (e.g. `https://hf-mirror.com`) for the bootstrap step only.

### 2. Run the API

```bash
docker compose up --build
```

With hot reload for app code changes:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

### GPU (local only)

Uncomment the `deploy.resources.reservations.devices` block in `docker-compose.yml`, then set in `.env`:

```
DEVICE=cuda:0
```

If no GPU is detected at startup, the service logs a warning and falls back to CPU automatically.

### Health checks

| Endpoint | Auth | Meaning |
|----------|------|---------|
| `GET /healthz` | None | Process is up (returns 200 even while model is loading) |
| `GET /readyz` | None | Model finished loading (503 until ready) |

```bash
curl http://localhost:8080/healthz
curl http://localhost:8080/readyz
```

Wait for `/readyz` to return 200 before sending TTS requests. First boot after bootstrap loads multi-GB weights from the volume into memory/VRAM.

Interactive API docs: `http://localhost:8080/docs`

---

## API contract

All `/v1/tts/*` routes require the `X-API-Key` header matching the `API_KEY` environment variable. Missing or wrong keys return **401**:

```json
{"message": "Invalid or missing API key"}
```

Successful responses return **24 kHz mono WAV** (`Content-Type: audio/wav`).

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
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from OmniVoice."}' \
  --output out.wav
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | *(required)* | Shared secret for `X-API-Key` header — generate fresh, never reuse |
| `MODEL_NAME` | `k2-fsa/OmniVoice` | HF repo id — **bootstrap script only**, not used at API runtime |
| `MODEL_STORE_DIR` | `/data/omnivoice-model` | Local path to self-hosted weights (must match volume mount) |
| `DEVICE` | `cuda:0` | Inference device (`cpu` on CPU-only hosts) |
| `DTYPE` | auto | `float16` on GPU, `float32` on CPU (override optional) |
| `HF_HUB_OFFLINE` | `1` | Must stay `1` in production — blocks HF network access |
| `TRANSFORMERS_OFFLINE` | `1` | Must stay `1` in production — blocks transformers downloads |
| `MAX_TEXT_LENGTH` | `500` | Max input text length |
| `PORT` | `8080` | HTTP port |

See [`.env.example`](.env.example) for a template.

---

## Railway deployment

Railway builds from the [`Dockerfile`](Dockerfile) via [`railway.json`](railway.json) — not Nixpacks/Railpack auto-detection.

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
   - Mount path: `/data/omnivoice-model` (must match `MODEL_STORE_DIR`)
4. **Seed the volume once** (before the API can start — empty `MODEL_STORE_DIR` fails fast):

   ```bash
   # Bootstrap requires network access to Hugging Face — override offline flags for this one-off run
   railway run -- \
     sh -c "HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 pip install huggingface_hub && python /app/scripts/bootstrap_model.py"
   ```

   Re-run bootstrap only when upgrading checkpoints, not on normal deploys.

5. **Set environment variables** in the Railway dashboard:

   | Variable | Value |
   |----------|-------|
   | `API_KEY` | Generate a new random secret |
   | `MODEL_STORE_DIR` | `/data/omnivoice-model` |
   | `DEVICE` | `cpu` (until GPU plans exist elsewhere use RunPod) |
   | `HF_HUB_OFFLINE` | `1` |
   | `TRANSFORMERS_OFFLINE` | `1` |

6. **Volume permissions:** volumes mount as root. The container runs as non-root (`appuser`, UID 1000). Set `RAILWAY_RUN_UID=0` on the service if the app cannot read the volume ([Railway volumes docs](https://docs.railway.com/guides/volumes)).
7. **Deploy.** `healthcheckTimeout` is 600 s in `railway.json` for cold model load. Poll `/readyz` before routing traffic.

### Cold start behavior

- `/healthz` — returns 200 once the process starts (Railway health check target)
- `/readyz` — returns 503 until model load completes

Restarting/redeploying the API container makes **zero network calls to huggingface.co** — weights load only from the mounted volume.

---

## Known limitations

- **Single uvicorn worker** — the GPU model is loaded once into VRAM/RAM; multiple workers would each load a separate copy. Concurrency requires a request queue (not implemented).
- **No built-in request queueing** — concurrent requests share one model instance.
- **Shared API key only** — suitable for internal service-to-service calls, not direct client exposure.
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
│   ├── schemas.py           # Pydantic request/response models
│   ├── config.py            # Environment-driven settings
│   └── auth.py              # X-API-Key verification
├── scripts/
│   └── bootstrap_model.py   # One-time HF weight download (not used at API runtime)
├── Dockerfile
├── docker-compose.yml
├── docker-compose.dev.yml
├── railway.json
├── requirements.txt
├── requirements-bootstrap.txt
├── .dockerignore
├── .env.example
└── README.md
```

---

## License

OmniVoice model and library: see [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice). This wrapper service is part of the Langify project infrastructure.
# ominiTTS
