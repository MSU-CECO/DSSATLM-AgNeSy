"""
Convenience launcher — run with:
    python run.py
or via uvicorn directly:
    uvicorn weather_api.main:app --host 0.0.0.0 --port 8001 --reload
"""
import uvicorn
from src.weather_api.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "src.weather_api.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )