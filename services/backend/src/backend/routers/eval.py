from __future__ import annotations

import os
import json

from fastapi import APIRouter, Header, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from ..schemas import EvalQueryRequest
from ..streaming import stream_eval

router = APIRouter(prefix="/api/eval", tags=["eval"])


def _check_eval_key(x_eval_key: str | None) -> None:
    expected = os.environ.get("EVAL_KEY")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Eval mode is not configured on this server (EVAL_KEY not set).",
        )
    if x_eval_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid eval key.",
        )


def _assemble_farmer_query(
    query: str,
    crop: str,
    variety: str,
    planting_date: str,
    latitude: float,
    longitude: float,
    irrigation_events: list,
    nitrogen_events: list,
    phosphorus_events: list,
    potassium_events: list,
) -> str:
    def fmt_events(evs, unit):
        if not evs:
            return "none"
        return ", ".join(f"{d}: {a}{unit}" for d, a in evs)

    return (
        f"{query}\n\n"
        f"Simulation context:\n"
        f"  crop={crop}, variety={variety}\n"
        f"  planting_date={planting_date}\n"
        f"  latitude={latitude}, longitude={longitude}\n"
        f"  irrigation applications: {fmt_events(irrigation_events, 'mm')}\n"
        f"  nitrogen applications: {fmt_events(nitrogen_events, 'kg/ha')}\n"
        f"  phosphorus applications: {fmt_events(phosphorus_events, 'kg/ha')}\n"
        f"  potassium applications: {fmt_events(potassium_events, 'kg/ha')}\n"
    )


@router.get("/query")
async def eval_query_sse(
    query: str = Query(..., min_length=3),
    crop: str = Query(...),
    variety: str = Query(...),
    planting_date: str = Query(...),
    latitude: float = Query(...),
    longitude: float = Query(...),
    irrigation_events: str = Query("[]"),   # JSON-encoded [[date, amount], ...]
    nitrogen_events: str = Query("[]"),
    phosphorus_events: str = Query("[]"),
    potassium_events: str = Query("[]"),
    model_a: str = Query("claude-sonnet"),
    model_b: str = Query("gpt-4o"),
    model_c: str | None = Query(None),
    model_d: str | None = Query(None),
    model_e: str | None = Query(None),
    openrouter_api_key: str = Query(..., min_length=10),
    wandb_api_key: str | None = Query(None),
    wandb_project: str | None = Query(None),
    x_eval_key: str | None = Query(None, alias="eval_key"),
) -> StreamingResponse:
    """
    GET version of /api/eval/query for EventSource clients.
    eval_key is passed as a query param since EventSource can't set headers.
    Now accepts full JSON-encoded event lists matching /api/query.
    """
    _check_eval_key(x_eval_key)

    farmer_query = _assemble_farmer_query(
        query=query,
        crop=crop,
        variety=variety,
        planting_date=planting_date,
        latitude=latitude,
        longitude=longitude,
        irrigation_events=json.loads(irrigation_events),
        nitrogen_events=json.loads(nitrogen_events),
        phosphorus_events=json.loads(phosphorus_events),
        potassium_events=json.loads(potassium_events),
    )
    models = [m for m in [model_a, model_b, model_c, model_d, model_e] if m is not None]
    return StreamingResponse(
        stream_eval(
            farmer_query=farmer_query,
            openrouter_api_key=openrouter_api_key,
            wandb_api_key=wandb_api_key,
            models=models,
            wandb_project=wandb_project,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/query")
async def eval_query(
    body: EvalQueryRequest,
    x_eval_key: str | None = Header(None, alias="X-Eval-Key"),
) -> StreamingResponse:
    """POST version — kept for programmatic clients."""
    _check_eval_key(x_eval_key)

    if len(body.models) < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Eval mode requires at least 2 models.",
        )

    return StreamingResponse(
        stream_eval(
            farmer_query=body.farmer_query,
            openrouter_api_key=body.openrouter_api_key,
            wandb_api_key=body.wandb_api_key,
            models=body.models,
            wandb_project=body.wandb_project,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

