# RunPod deployment (production GPU)

Deploy the OmniVoice TTS API on a **RunPod persistent GPU Pod** with a **Network Volume** for model weights. This replaces the archived Railway CPU-only setup.

**Prerequisites:** RunPod account, SSH key added in RunPod settings, basic familiarity with Docker Compose.

---

## 1. Provision GPU Pod + Network Volume

### Create a Network Volume

1. RunPod console → **Storage** → **Network Volumes** → **Create Network Volume**
2. Size: **≥ 20 GB** (model store is ~3.27 GB; leave headroom)
3. Pick a **datacenter** — the GPU Pod must deploy in the same region

Network volumes are persistent and independent of the Pod. They must be attached **at Pod creation**; you cannot attach or detach later without recreating the Pod.

### Deploy a GPU Pod

1. RunPod console → **Pods** → **Deploy**
2. Select a **GPU** in the same datacenter as your Network Volume
3. Choose a template with Docker and GPU support (RunPod PyTorch templates work)
4. Under **Network Volume**, select the volume you created
5. Default host mount path: **`/workspace`** (configurable via `volumeMountPath` at creation — default is fine; use a subdirectory for model weights)
6. Under **Expose HTTP Ports**, add: **`8080`**
7. Optionally expose **TCP port 22** for SSH if not already enabled by the template
8. Deploy

> **Secure Cloud** is recommended for stable public IPs. Community Cloud IPs may change on restart.

---

## 2. CUDA compatibility check

SSH into the Pod (Connect → SSH in the RunPod console), then:

```bash
nvidia-smi
```

Note the **CUDA Version** reported by the driver (top-right of `nvidia-smi` output). The Dockerfile uses:

- Base image: `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04`
- PyTorch: `2.8.0` with `cu128` wheels

The host driver must support CUDA **12.8** (or be new enough for the 12.8 runtime). If `nvidia-smi` shows a lower max CUDA (e.g. 12.4), adjust the Dockerfile base image tag and PyTorch wheel index to match before building — a mismatch causes confusing GPU failures.

---

## 3. Deploy the service

Two paths — pick based on your workflow:

| Path | When to use |
|------|-------------|
| **A: Build on Pod** | Fast iteration — clone repo on the Pod and `docker compose up --build` |
| **B: Registry image** | Stable prod — push image to Docker Hub/GHCR, pull on Pod, use `image:` in compose |

### Path A — Build on Pod (recommended for first deploy)

```bash
# SSH into the Pod
git clone <your-repo-url>
cd omni-tts/omnivoice-service   # adjust path to match your clone layout
cp .env.example .env

# Create host directory for model weights on the Network Volume
mkdir -p /workspace/omnivoice-model

# Edit .env if needed — defaults are correct for RunPod:
#   OMNIVOICE_VOLUME_HOST_PATH=/workspace/omnivoice-model
#   MODEL_STORE_DIR=/data/omnivoice-model
#   DEVICE=cuda:0

docker compose up --build -d
```

### Path B — Registry image

Build and push from your machine or CI:

```bash
docker build -t ghcr.io/<org>/omnivoice-service:latest .
docker push ghcr.io/<org>/omnivoice-service:latest
```

On the Pod, use a compose override or edit `docker-compose.yml` to replace `build: .` with `image: ghcr.io/<org>/omnivoice-service:latest`, then `docker compose up -d`.

---

## 4. Environment variables

Set in `.env` on the Pod (see [`.env.example`](../../.env.example)):

| Variable | Value | Notes |
|----------|-------|-------|
| `MODEL_STORE_DIR` | `/data/omnivoice-model` | Container path; must match compose volume mount target |
| `OMNIVOICE_VOLUME_HOST_PATH` | `/workspace/omnivoice-model` | Host path on Network Volume (compose bind mount) |
| `DEVICE` | `cuda:0` | CPU fallback still works if GPU passthrough fails |
| `HF_HUB_OFFLINE` | `1` | Required in production |
| `TRANSFORMERS_OFFLINE` | `1` | Required in production |
| `PORT` | `8080` | Must match exposed HTTP port |

Do **not** set `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` to `0` for normal API operation — only during bootstrap.

---

## 5. Bootstrap model weights (one-time)

SSH into the Pod. The Pod stays running — no Railway-style chicken-and-egg with crashed containers.

### Option A — `docker compose run` (recommended)

```bash
cd omnivoice-service
mkdir -p /workspace/omnivoice-model

docker compose run --rm \
  -e HF_HUB_OFFLINE=0 \
  -e TRANSFORMERS_OFFLINE=0 \
  omnivoice-api \
  python scripts/bootstrap_model.py
```

Wait for **`Bootstrap complete.`** with on-disk size **~3.27 GB** and flat top-level files (`config.json`, `model.safetensors`, `audio_tokenizer/`, etc.).

### Option B — `BOOTSTRAP_ONLY=1`

Add `BOOTSTRAP_ONLY=1` to `.env` or compose `environment`, then:

```bash
docker compose up --build
```

Watch logs for **`Bootstrap complete.`**, then remove `BOOTSTRAP_ONLY`, restart:

```bash
docker compose down
docker compose up -d
```

Re-run bootstrap only when upgrading checkpoints or repairing a broken download.

---

## 6. Expose the API externally

RunPod HTTP proxy URL format ([docs](https://docs.runpod.io/pods/configuration/expose-ports)):

```
https://[POD_ID]-8080.proxy.runpod.net
```

Replace `[POD_ID]` with your Pod's ID from the RunPod console.

Example health check:

```bash
curl https://<POD_ID>-8080.proxy.runpod.net/healthz
curl https://<POD_ID>-8080.proxy.runpod.net/readyz
```

Example TTS request:

```bash
curl -X POST https://<POD_ID>-8080.proxy.runpod.net/v1/tts/auto \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from OmniVoice on RunPod."}' \
  --output out.wav
```

### Proxy constraints

- **HTTPS only** — proxy terminates TLS; internal service uses HTTP on port 8080
- **100-second timeout** — Cloudflare enforces a max connection time. Long TTS requests must complete within 100s or use TCP exposure / async job patterns
- **Public accessibility** — endpoint is reachable on the internet. Restrict access via Langify backend-only routing (service has no built-in auth)

The API binds `0.0.0.0:8080` via uvicorn — required for proxy access.

---

## 7. Verification checklist

Run these on the Pod after deploy:

### 7.1 Host GPU

```bash
nvidia-smi
```

Confirm GPU is visible and driver CUDA version matches Dockerfile expectations (see §2).

### 7.2 Container GPU passthrough

```bash
docker compose up --build -d
docker compose exec omnivoice-api python -c "import torch; print(torch.cuda.is_available())"
```

**Must print `True`.** If `False`, fix GPU passthrough (`nvidia-container-toolkit`, compose GPU block) before proceeding.

### 7.3 Bootstrap verification

After bootstrap (§5), confirm:

- Total size ~3.27 GB
- Flat layout: `config.json`, `model.safetensors`, `tokenizer.json`, `audio_tokenizer/`
- Marker file `.omnivoice-bootstrap-complete` present
- No `models--*` cache nesting

### 7.4 Readiness

```bash
docker compose restart
curl -s http://localhost:8080/readyz | jq .
```

- HTTP **200** when `model_loaded: true`
- Container logs show `device=cuda:0` — **not** "CUDA not available — falling back to CPU"

### 7.5 End-to-end synthesis

```bash
curl -X POST http://localhost:8080/v1/tts/auto \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from OmniVoice."}' \
  --output out.wav
```

Confirm `out.wav` is playable. Latency should be noticeably lower than the Railway CPU deployment.

### 7.6 External reachability

From outside the Pod (your laptop or Langify backend):

```bash
curl https://<POD_ID>-8080.proxy.runpod.net/readyz
```

---

## 8. Local dev on the Pod

Hot reload without rebuilding the image:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

Mounts `./app` into the container and runs uvicorn with `--reload`.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `torch.cuda.is_available()` → `False` | GPU not passed into container | Confirm `deploy.resources.reservations.devices` in compose; install `nvidia-container-toolkit` on host |
| `OSError: no file named model.safetensors` | Volume not bootstrapped | Run bootstrap (§5) |
| `/readyz` 503, load_error mentions empty store | Same | Run bootstrap |
| `524` from proxy URL | Request exceeded 100s | Shorten text; use fewer `num_step`; or expose TCP and call directly |
| Permission errors on volume | Host mount owned by root | Entrypoint `chown`s `MODEL_STORE_DIR` — ensure container starts as root (default); check logs for chown errors |
| CUDA version mismatch errors | Driver vs image mismatch | Adjust Dockerfile CUDA tag per §2 |

---

## References

- [RunPod Network Volumes](https://docs.runpod.io/storage/network-volumes)
- [RunPod Expose Ports](https://docs.runpod.io/pods/configuration/expose-ports)
- [RunPod Create Pod API](https://docs.runpod.io/api-reference/pods/POST/pods)
