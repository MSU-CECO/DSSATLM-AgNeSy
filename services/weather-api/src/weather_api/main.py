from contextlib import asynccontextmanager

from fastapi import FastAPI

from .data import load_parquet
from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the parquet file once at startup, release on shutdown."""
    load_parquet()
    yield


app = FastAPI(
    title="Weather API",
    description=(
        "Daily weather data from NASA/ORNL DAYMET_V4 at 1km resolution. "
        "Returns observations for the nearest grid point to any queried coordinate. "
        "Part of the DSSATLM-AgNeSy project."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
