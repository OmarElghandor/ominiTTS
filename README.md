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

## Quick start (local)

### Prerequisites

- Docker and Docker Compose
- (Optional) NVIDIA GPU + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU inference

### Setup

```bash
cd omnivoice-service
cp .env.example .env
# Edit .env — set API_KEY to a random secret
```

### Run

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

Wait for `/readyz` to return 200 before sending TTS requests. First boot downloads multi-GB model weights into the named Docker volume (`omnivoice-model-cache`). Subsequent restarts reuse the cache.

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
| `MODEL_NAME` | `k2-fsa/OmniVoice` | HuggingFace model ID |
| `DEVICE` | `cuda:0` | Inference device (`cpu` for Railway testing) |
| `DTYPE` | auto | `float16` on GPU, `float32` on CPU (override optional) |
| `HF_HOME` | `/data/huggingface` | Model cache directory (must match volume mount) |
| `MAX_TEXT_LENGTH` | `500` | Max input text length |
| `PORT` | `8080` | HTTP port |

See [`.env.example`](.env.example) for a template.

---

## Railway deployment (CPU testing)

Railway builds from the [`Dockerfile`](Dockerfile) via [`railway.json`](railway.json) — not Nixpacks.

### Important: Railway has no GPU instances

As of Railway's current documentation, **GPU instances are not available**. This service can be deployed on Railway for **CPU-based testing and API validation only**. CPU inference is far slower than GPU and is not suitable for production latency. When production GPU is needed, migrate to RunPod or a dedicated GPU host (per Langify's scale path).

### Steps

1. **Create a Railway project** and connect this repository.
2. Set the **root directory** to `omnivoice-service/` (if the repo contains other projects).
3. **Attach a persistent volume:**
   - Open the Command Palette (`⌘K` / `Ctrl+K`) → **Create Volume**
   - Attach it to the OmniVoice service
   - Set mount path to `/data/huggingface`
4. **Set environment variables** in the Railway dashboard:

   | Variable | Value |
   |----------|-------|
   | `API_KEY` | Generate a new random secret |
   | `DEVICE` | `cpu` |
   | `HF_HOME` | `/data/huggingface` |
   | `MODEL_NAME` | `k2-fsa/OmniVoice` |

5. **Scale resources** under Settings → Resources (more CPU/RAM helps CPU inference, but remains slow).
6. **Deploy.** First deploy downloads model weights (several minutes). `healthcheckTimeout` is set to 600 s in `railway.json`. Poll `/readyz` to confirm the model is callable.

### Volume permissions

If the non-root container user cannot write to the mounted volume, set `RAILWAY_RUN_UID=0` in Railway service variables (Railway docs recommend this for volume write access with custom UIDs).

### Cold start behavior

- `/healthz` — returns 200 once the process starts (Railway health check target)
- `/readyz` — returns 503 until model load completes; use this before routing traffic in custom setups

---

## Known limitations

- **Single uvicorn worker** — the GPU model is loaded once into VRAM/RAM; multiple workers would each load a separate copy. Concurrency requires a request queue (not implemented).
- **No built-in request queueing** — concurrent requests share one model instance.
- **Shared API key only** — suitable for internal service-to-service calls, not direct client exposure.
- **24 kHz output only** — no resampling unless the caller handles it downstream.
- **CPU inference is slow** — acceptable for testing; production needs GPU hardware.

---

## Project structure

```
omnivoice-service/
├── app/
│   ├── main.py       # FastAPI routes, lifespan, exception handlers
│   ├── tts.py        # OmniVoice wrapper (singleton model)
│   ├── schemas.py    # Pydantic request/response models
│   ├── config.py     # Environment-driven settings
│   └── auth.py       # X-API-Key verification
├── Dockerfile
├── docker-compose.yml
├── docker-compose.dev.yml
├── railway.json
├── requirements.txt
├── .dockerignore
├── .env.example
└── README.md
```

---

## License

OmniVoice model and library: see [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice). This wrapper service is part of the Langify project infrastructure.
# ominiTTS
