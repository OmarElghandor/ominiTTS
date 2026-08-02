# RunPod Serverless deployment (on-demand)

Queue-based Serverless endpoint built on **SpeechEngine** → **ModelManager** → **OmniVoiceProvider**. Workers scale to **zero** when idle and bill only while running.

This is separate from the [persistent GPU Pod guide](README.md) (FastAPI `/v1/tts/*`).

| | Pod (FastAPI) | Serverless (this guide) |
|--|---------------|-------------------------|
| Entrypoint | `uvicorn app.main:app` | `python handler.py` → `api/handler.py` |
| API | `/v1/tts/*` → raw WAV | `/run` + `/runsync` → JSON + base64 WAV |
| Contract | Clone / design / auto JSON | OpenAI-compatible (`input`, `voice`, …) |
| Billing | Always-on GPU | On-demand (active workers = 0) |
| Weights | Volume → `/data/omnivoice-model` | Network Volume → `/runpod-volume/omnivoice-model` |
| Startup | Soft load; `/readyz` 503 until ready | **Fail-fast** load + warmup; process exits if missing |

**Langify:** Wire the TTS provider to this Serverless contract (`audio_base64`), not the Pod proxy URL.

Architecture (shared with FastAPI):

```
api/handler.py  or  app/main.py
        ↓
  SpeechEngine.generate(...)
        ↓
  ModelManager (singleton)
        ↓
  SpeechProvider (omnivoice today)
```

Swap providers later via `SPEECH_PROVIDER` without changing the handler or FastAPI routes.

---

## 1. Network Volume + bootstrap

1. RunPod → **Storage** → Network Volume ≥ **20 GB** in a datacenter with your target GPUs.
2. Attach the volume to the Serverless endpoint (mounts at **`/runpod-volume`**).
3. Seed once into `omnivoice-model` on that volume (never at runtime):

```bash
# On a temporary Pod in the SAME datacenter with the volume attached:
mkdir -p /workspace/omnivoice-model
docker run --rm --gpus all \
  -e MODEL_STORE_DIR=/data/omnivoice-model \
  -e HF_HUB_OFFLINE=0 \
  -e TRANSFORMERS_OFFLINE=0 \
  -e BOOTSTRAP_ONLY=1 \
  -v /workspace/omnivoice-model:/data/omnivoice-model \
  <registry>/omnivoice-service:latest
```

Wait for **`Bootstrap complete.`** (~3.27 GB flat layout). Serverless uses:

```
MODEL_STORE_DIR=/runpod-volume/omnivoice-model
```

Workers never contact Hugging Face (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).

---

## 2. Build and push

```bash
cd omnivoice-service
docker build --platform linux/amd64 --target serverless \
  -t <registry>/omnivoice-serverless:latest .
docker push <registry>/omnivoice-serverless:latest
```

Default `docker build` (no `--target`) still produces the FastAPI/Pod image.

---

## 3. Endpoint settings

| Setting | Value |
|---------|--------|
| Image | `<registry>/omnivoice-serverless:latest` |
| GPU | **16 GB+ VRAM** |
| Active workers | **0** (on-demand) |
| Max workers | 1–2 |
| Idle timeout | **60–300 s** |
| Execution timeout | **120–300 s** |
| FlashBoot | **On** |
| Network Volume | Attached |

### Environment

| Variable | Value |
|----------|--------|
| `MODEL_STORE_DIR` | `/runpod-volume/omnivoice-model` |
| `DEVICE` | `cuda:0` |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | `1` |
| `MAX_CONCURRENT_REQUESTS` | `1` |
| `MAX_QUEUE_SIZE` | `8` |
| `REQUEST_TIMEOUT` | `120` |
| `LOG_LEVEL` | `INFO` |
| `OUTPUT_FORMAT` | `wav` |
| `SPEECH_PROVIDER` | `omnivoice` |
| `WARMUP_TEXT` | `Hello` |
| `SKIP_RECURSIVE_CHOWN` | `1` (baked into image) |

Startup sequence: CUDA init → load from volume → tokenizer → warmup `"Hello"` → write `/tmp/omnivoice-ready` → accept jobs. Missing model → **process exits**.

---

## 4. Request / response contract

### OpenAI-compatible input

```json
{
  "model": "omnivoice",
  "voice": "default",
  "input": "Hello world",
  "language": "en",
  "speed": 1.0,
  "num_step": 32
}
```

| Field | Role |
|-------|------|
| `input` | Text to speak (required). Legacy alias: `text` |
| `voice` | `default` / `auto` → auto mode; other values → design instruct |
| `instruct` | Explicit design mode |
| `ref_audio` + `ref_text` | Clone mode (base64 audio) |
| `mode` | Optional explicit `auto` / `design` / `clone` |
| `language`, `speed`, `num_step`, `duration` | Generation params |

### Success output

```json
{
  "audio_base64": "<base64 WAV>",
  "sample_rate": 24000,
  "content_type": "audio/wav"
}
```

### Errors

```json
{"error": "human-readable message"}
```

Includes queue-full and timeout cases.

---

## 5. Call the endpoint

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"model":"omnivoice","voice":"default","input":"Hello from OmniVoice."}}'

# Decode:
jq -r '.output.audio_base64' response.json | base64 -d > out.wav
```

Async: `POST /run` then poll `/status/<job_id>`.

---

## 6. Local test

```bash
cd omnivoice-service
export MODEL_STORE_DIR=./model-store
export DEVICE=cuda:0   # or cpu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python handler.py --test_input '{"input":{"input":"Hello"}}'
```

Fail-fast: missing store exits before accepting jobs.

---

## 7. Metrics

Each request emits one structured JSON log line with:

`request_id`, `queue_wait_ms`, `inference_ms`, `total_latency_ms`, `characters`, `audio_duration`, `gpu_memory_mb`, `peak_gpu_memory_mb`, `cold_start`, `provider`, `mode`.

---

## 8. Cold starts

With active workers = 0, the first request pays for container start + volume load + warmup. Mitigations: FlashBoot, Network Volume (no HF download), longer idle timeout, singleton model. Near-zero latency requires Active workers ≥ 1.

---

## References

- [RunPod Serverless overview](https://docs.runpod.io/serverless/overview)
- [Endpoint configuration](https://docs.runpod.io/serverless/endpoints/endpoint-configurations)
- [Network volumes](https://docs.runpod.io/storage/network-volumes)
- Pod guide: [README.md](README.md)
