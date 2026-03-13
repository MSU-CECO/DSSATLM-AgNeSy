from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

from .config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str = Security(_api_key_header)) -> str:
    if api_key != settings.weather_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Supply it in the X-API-Key header.",
        )
    return api_key
