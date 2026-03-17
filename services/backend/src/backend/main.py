from __future__ import annotations

import logging
import os

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import eval as eval_router
from .routers import pro as pro_router
from .schemas import HealthResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_raw_origins = os.environ.get("CORS_ORIGINS", "*")
CORS_ORIGINS: list[str] = (
    ["*"] if _raw_origins.strip() == "*" else [o.strip() for o in _raw_origins.split(",")]
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    cors_display = CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["* (all origins — dev mode)"]
    logger.info("DSSATLM backend starting up. CORS origins: %s", cors_display)
    eval_key_set = bool(os.environ.get("EVAL_KEY"))
    logger.info("EVAL_KEY configured: %s", eval_key_set)
    yield


app = FastAPI(
    title="DSSATLM-AgNeSy Backend",
    description="FastAPI backend for LLM-powered DSSAT agricultural simulations.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pro_router.router)
app.include_router(eval_router.router)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """Liveness check — returns 200 OK when the server is running."""
    return HealthResponse()
