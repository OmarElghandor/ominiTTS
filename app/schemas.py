from pydantic import BaseModel, Field, field_validator


class GenerationParams(BaseModel):
    num_step: int = Field(default=32, ge=1, le=128)
    speed: float = Field(default=1.0, gt=0.0, le=5.0)
    duration: float | None = Field(default=None, gt=0.0)
    language: str | None = None


class CloneRequest(BaseModel):
    text: str = Field(..., min_length=1)
    ref_audio: str = Field(..., description="Base64-encoded reference audio (WAV/MP3/FLAC)")
    ref_text: str | None = Field(
        default=None,
        description="Transcript of reference audio; omitting triggers Whisper auto-transcription",
    )
    num_step: int = Field(default=32, ge=1, le=128)
    speed: float = Field(default=1.0, gt=0.0, le=5.0)
    duration: float | None = Field(default=None, gt=0.0)
    language: str | None = None

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("text must not be empty")
        return stripped


class DesignRequest(BaseModel):
    text: str = Field(..., min_length=1)
    instruct: str = Field(
        ...,
        min_length=1,
        description="Comma-separated voice attributes (e.g. 'female, young adult, british accent')",
    )
    num_step: int = Field(default=32, ge=1, le=128)
    speed: float = Field(default=1.0, gt=0.0, le=5.0)
    duration: float | None = Field(default=None, gt=0.0)
    language: str | None = None

    @field_validator("text", "instruct")
    @classmethod
    def not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("field must not be empty")
        return stripped


class AutoRequest(BaseModel):
    text: str = Field(..., min_length=1)
    num_step: int = Field(default=32, ge=1, le=128)
    speed: float = Field(default=1.0, gt=0.0, le=5.0)
    duration: float | None = Field(default=None, gt=0.0)
    language: str | None = None

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("text must not be empty")
        return stripped


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    model_loading: bool = False
    load_error: str | None = None
