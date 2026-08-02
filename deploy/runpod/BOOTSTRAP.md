# Fix empty MODEL_STORE_DIR on RunPod

Your worker log:

```
CUDA initialized on cuda:0
MODEL_STORE_DIR is empty — ...
```

means the GPU is fine, but the Network Volume has **no OmniVoice weights** yet. Workers never download from Hugging Face.

---

## 1. Confirm the volume

| Mode | Volume mount | Target store dir |
|------|--------------|------------------|
| **Serverless** | `/runpod-volume` | `/runpod-volume/omnivoice-model` |
| **GPU Pod** | `/workspace` | `/workspace/omnivoice-model` |

On a Pod in the **same datacenter** as the volume:

```bash
ls -la /workspace
# and/or
ls -la /runpod-volume
```

If neither exists, attach a Network Volume (≥ 20 GB) before bootstrapping.

---

## 2. Bootstrap (one-time)

SSH into that Pod, clone/pull the repo, then:

```bash
cd /path/to/ominiTTS   # repo root
bash scripts/bootstrap_runpod.sh
```

Or with your Docker image:

```bash
IMAGE=<your-registry>/omnivoice-serverless:latest bash scripts/bootstrap_runpod.sh
```

Wait for **`Bootstrap complete.`** (~3.27 GB). Confirm:

```bash
ls /workspace/omnivoice-model/
# config.json  model.safetensors  tokenizer.json  audio_tokenizer/  .omnivoice-bootstrap-complete
du -sh /workspace/omnivoice-model
```

The script auto-detects `/runpod-volume` vs `/workspace` and prints the exact env vars to set next.

---

## 3. Align endpoint env and restart

On the **Serverless** endpoint:

```
MODEL_STORE_DIR=/runpod-volume/omnivoice-model
DEVICE=cuda:0
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

Redeploy / restart the worker. Success looks like:

1. `CUDA initialized on cuda:0`
2. `Loading speech provider from MODEL_STORE_DIR=...`
3. `Warmup complete` / `ModelManager ready`
4. **No** empty-store `RuntimeError`

Then test `/runsync`.

---

## Do not

- Set `HF_HUB_OFFLINE=0` on the production worker to auto-download
- Bake the 3 GB model into the Docker image
- Use a volume in a **different** datacenter than the endpoint

See also: [serverless.md](serverless.md) §1, [README.md](README.md) §5 (Pod bootstrap).
