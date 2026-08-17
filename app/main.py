import asyncio
import base64
import binascii
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.schemas import (
    AutoRequest,
    CloneRequest,
    DesignRequest,
    HealthResponse,
    SpeakerInfo,
    SpeakersResponse,
)
from app.speech import (
    QueueFullError,
    RequestTimeoutError,
    SpeechRequest,
    get_speech_engine,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _health_payload() -> HealthResponse:
    engine = get_speech_engine()
    load_error = engine.get_load_error()
    status = "ok" if not load_error else "degraded"
    return HealthResponse(
        status=status,
        device=engine.get_device(),
        model_loaded=engine.is_ready(),
        model_loading=engine.is_loading(),
        load_error=load_error,
    )


def _validate_text_length(text: str) -> None:
    settings = get_settings()
    if len(text) > settings.MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail={
                "message": f"text exceeds maximum length of {settings.MAX_TEXT_LENGTH} characters"
            },
        )


def _require_model_ready() -> None:
    engine = get_speech_engine()
    if engine.is_ready():
        return

    load_error = engine.get_load_error()
    if load_error:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Model failed to load. Check container logs and /readyz for details.",
                "load_error": load_error,
            },
        )

    if engine.is_loading():
        raise HTTPException(
            status_code=503,
            detail={
                "message": (
                    "Model is still loading from MODEL_STORE_DIR. Poll GET /readyz until "
                    "model_loaded is true."
                ),
            },
        )

    raise HTTPException(
        status_code=503,
        detail={"message": "Model is not loaded. Check /readyz and service logs."},
    )


def _audio_response(wav_bytes: bytes) -> Response:
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Length": str(len(wav_bytes))},
    )


def _decode_ref_audio_b64(ref_audio_b64: str) -> bytes:
    try:
        return base64.b64decode(ref_audio_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"message": "ref_audio must be valid base64-encoded audio data"},
        ) from exc


def _suffix_from_upload(filename: str | None) -> str:
    if not filename:
        return ".wav"
    suffix = Path(filename).suffix.lower()
    return suffix if suffix in {".wav", ".mp3", ".flac", ".ogg", ".m4a"} else ".wav"


async def _synthesize(request: SpeechRequest) -> bytes:
    engine = get_speech_engine()
    try:
        result = await engine.generate(request)
    except QueueFullError as exc:
        raise HTTPException(
            status_code=503,
            detail={"message": str(exc)},
        ) from exc
    except RequestTimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail={"message": str(exc)},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc)}) from exc
    return result.audio


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    engine = get_speech_engine()

    # Soft-fail for Pod/local: empty volume surfaces via /readyz instead of crashing.
    async def _load():
        await run_in_threadpool(
            lambda: engine.startup(settings, fail_fast=False)
        )

    load_task = asyncio.create_task(_load())
    app.state.model_load_task = load_task
    yield
    if not load_task.done():
        load_task.cancel()
        try:
            await load_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="VoiceTut TTS Service",
    description=(
        "Self-hosted Egyptian-Arabic TTS microservice wrapping VoiceTut-TTS "
        "(fine-tuned OmniVoice) with built-in speakers, normalization, and cloning"
    ),
    version="1.1.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    if errors:
        first = errors[0]
        loc = " -> ".join(str(part) for part in first.get("loc", []))
        msg = first.get("msg", "Validation error")
        message = f"{loc}: {msg}" if loc else msg
    else:
        message = "Validation error"
    return JSONResponse(status_code=422, content={"message": message})


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "message" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"message": str(exc.detail)})


@app.exception_handler(torch.cuda.OutOfMemoryError)
async def cuda_oom_handler(request: Request, exc: torch.cuda.OutOfMemoryError):
    logger.error("CUDA out of memory during request: %s", exc)
    get_speech_engine().clear_cuda_cache()
    return JSONResponse(
        status_code=503,
        content={"message": "Service temporarily overloaded. Try again shortly."},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error during request")
    return JSONResponse(
        status_code=500,
        content={"message": "An internal error occurred while processing the request."},
    )


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return _health_payload()


@app.get("/readyz", response_model=HealthResponse)
async def readyz() -> HealthResponse | JSONResponse:
    payload = _health_payload()
    if not payload.model_loaded:
        return JSONResponse(
            status_code=503,
            content=payload.model_dump(),
        )
    return payload


@app.get("/v1/speakers", response_model=SpeakersResponse)
async def list_speakers() -> SpeakersResponse:
    _require_model_ready()
    settings = get_settings()
    engine = get_speech_engine()
    speakers = [
        SpeakerInfo(
            speaker_id=s.get("speaker_id", ""),
            speaker_name=s.get("speaker_name", ""),
            gender=s.get("gender", "") or "",
            tags=list(s.get("tags") or []),
        )
        for s in engine.list_speakers()
    ]
    return SpeakersResponse(speakers=speakers, default_speaker=settings.DEFAULT_SPEAKER)


@app.post("/v1/tts/clone")
async def tts_clone(request: Request):
    _require_model_ready()
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        text = form.get("text")
        if not text or not str(text).strip():
            raise HTTPException(status_code=422, detail={"message": "text is required"})
        text = str(text).strip()
        _validate_text_length(text)

        ref_file = form.get("ref_audio")
        if ref_file is None or not hasattr(ref_file, "read"):
            raise HTTPException(status_code=422, detail={"message": "ref_audio file is required"})
        ref_audio_bytes = await ref_file.read()
        if not ref_audio_bytes:
            raise HTTPException(status_code=422, detail={"message": "ref_audio file is empty"})

        ref_text = form.get("ref_text")
        ref_text = str(ref_text).strip() if ref_text else None
        language = form.get("language")
        language = str(language).strip() if language else None
        num_step = int(form.get("num_step", 32))
        speed = float(form.get("speed", 1.0))
        duration_raw = form.get("duration")
        duration = float(duration_raw) if duration_raw not in (None, "") else None
        suffix = _suffix_from_upload(getattr(ref_file, "filename", None))
    else:
        body = await request.json()
        parsed = CloneRequest.model_validate(body)
        text = parsed.text
        _validate_text_length(text)
        ref_audio_bytes = _decode_ref_audio_b64(parsed.ref_audio)
        ref_text = parsed.ref_text
        language = parsed.language
        num_step = parsed.num_step
        speed = parsed.speed
        duration = parsed.duration
        suffix = ".wav"

    wav_bytes = await _synthesize(
        SpeechRequest(
            text=text,
            mode="clone",
            language=language,
            num_step=num_step,
            speed=speed,
            duration=duration,
            ref_audio=ref_audio_bytes,
            ref_text=ref_text,
            ref_audio_suffix=suffix,
        )
    )
    return _audio_response(wav_bytes)


@app.post("/v1/tts/design")
async def tts_design(body: DesignRequest):
    _require_model_ready()
    _validate_text_length(body.text)

    # Built-in speaker selection uses auto mode; free-text instruct uses design.
    mode = "auto" if body.speaker and not body.instruct else "design"
    wav_bytes = await _synthesize(
        SpeechRequest(
            text=body.text,
            mode=mode,
            instruct=body.instruct,
            speaker=body.speaker,
            language=body.language,
            num_step=body.num_step,
            speed=body.speed,
            duration=body.duration,
        )
    )
    return _audio_response(wav_bytes)


@app.post("/v1/tts/auto")
async def tts_auto(body: AutoRequest):
    _require_model_ready()
    _validate_text_length(body.text)

    wav_bytes = await _synthesize(
        SpeechRequest(
            text=body.text,
            mode="auto",
            speaker=body.speaker,
            language=body.language,
            num_step=body.num_step,
            speed=body.speed,
            duration=body.duration,
        )
    )
    return _audio_response(wav_bytes)
