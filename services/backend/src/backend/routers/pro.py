from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..schemas import ProQueryRequest
from ..streaming import stream_pro

router = APIRouter(prefix="/api", tags=["pro"])

SUPPORTED_MODELS: list[str] = [
    "gpt-4o",
    "gpt-4o-mini",
    "claude-sonnet",
    "llama-3.3-70b",
    "dsr1-llama-70b",
]


@router.get("/models")
async def list_models() -> dict:
    """Return the list of supported model slugs."""
    return {"models": SUPPORTED_MODELS}


@router.post("/query")
async def pro_query(body: ProQueryRequest) -> StreamingResponse:
    """
    Run a single-model DSSAT simulation and stream SSE events back.

    Event sequence:
        stage(parsing/start)      -> stage(parsing/done)        -> parsed(...)
        stage(simulating/start)   -> stage(simulating/done)
        stage(interpreting/start) -> stage(interpreting/done)
        result(...) | error(...)
    """
    return StreamingResponse(
        stream_pro(
            farmer_query=body.farmer_query,
            openrouter_api_key=body.openrouter_api_key,
            wandb_api_key=body.wandb_api_key,
            model=body.model,
            wandb_project=body.wandb_project,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   
        },
    )
