from pydantic import BaseModel, Field, field_validator, model_validator


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
    language: str | None = Field(
        default=None,
        description="Language code: 'arz'/'ar' (Egyptian Arabic) or 'en'",
    )

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("text must not be empty")
        return stripped


class DesignRequest(BaseModel):
    text: str = Field(..., min_length=1)
    instruct: str | None = Field(
        default=None,
        description="Comma-separated voice attributes (e.g. 'female, young adult'). "
        "Mutually exclusive with speaker.",
    )
    speaker: str | None = Field(
        default=None,
        description="Built-in VoiceTut speaker name (e.g. 'Mohamed', 'Asmaa'). "
        "Mutually exclusive with instruct.",
    )
    num_step: int = Field(default=32, ge=1, le=128)
    speed: float = Field(default=1.0, gt=0.0, le=5.0)
    duration: float | None = Field(default=None, gt=0.0)
    language: str | None = Field(
        default=None,
        description="Language code: 'arz'/'ar' (Egyptian Arabic) or 'en'",
    )

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("text must not be empty")
        return stripped

    @field_validator("instruct", "speaker")
    @classmethod
    def optional_not_blank(cls, v: str | None) -> str | None:
        if v is None:
            return None
        stripped = v.strip()
        return stripped or None

    @model_validator(mode="after")
    def require_instruct_or_speaker(self) -> "DesignRequest":
        if self.instruct and self.speaker:
            raise ValueError("Choose ONE of: instruct or speaker")
        if not self.instruct and not self.speaker:
            raise ValueError("instruct or speaker is required")
        return self


class AutoRequest(BaseModel):
    text: str = Field(..., min_length=1)
    speaker: str | None = Field(
        default=None,
        description="Built-in VoiceTut speaker name (e.g. 'Mohamed'). "
        "Defaults to DEFAULT_SPEAKER when omitted.",
    )
    num_step: int = Field(default=32, ge=1, le=128)
    speed: float = Field(default=1.0, gt=0.0, le=5.0)
    duration: float | None = Field(default=None, gt=0.0)
    language: str | None = Field(
        default=None,
        description="Language code: 'arz'/'ar' (Egyptian Arabic) or 'en'",
    )

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("text must not be empty")
        return stripped

    @field_validator("speaker")
    @classmethod
    def speaker_not_blank(cls, v: str | None) -> str | None:
        if v is None:
            return None
        stripped = v.strip()
        return stripped or None


class SpeakerInfo(BaseModel):
    speaker_id: str
    speaker_name: str
    gender: str = ""
    tags: list[str] = Field(default_factory=list)


class SpeakersResponse(BaseModel):
    speakers: list[SpeakerInfo]
    default_speaker: str


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    model_loading: bool = False
    load_error: str | None = None
