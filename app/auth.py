from fastapi import Header, HTTPException

from app.config import get_settings


async def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    settings = get_settings()
    if not settings.API_KEY:
        raise HTTPException(
            status_code=503,
            detail={"message": "Service is misconfigured: API_KEY is not set."},
        )
    if not x_api_key or x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=401,
            detail={"message": "Invalid or missing API key"},
        )
