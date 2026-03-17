from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from ..schemas import ProQueryRequest
from ..streaming import stream_pro, stream_pro_reinterpret

import json

router = APIRouter(prefix="/api", tags=["pro"])

SUPPORTED_MODELS: list[str] = [
    "gpt-4o",
    "gpt-4o-mini",
    "claude-sonnet",
    "llama-3.3-70b",
    "dsr1-llama-70b",
]


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


@router.get("/models")
async def list_models() -> dict:
    """Return the list of supported model slugs."""
    return {"models": SUPPORTED_MODELS}


@router.get("/query")
async def pro_query_sse(
    query: str = Query(..., min_length=3),
    crop: str = Query(...),
    variety: str = Query(...),
    planting_date: str = Query(...),
    latitude: float = Query(...),
    longitude: float = Query(...),
    irrigation_events: str = Query("[]"),
    nitrogen_events: str = Query("[]"),
    phosphorus_events: str = Query("[]"),
    potassium_events: str = Query("[]"),
    model: str = Query("claude-sonnet"),
    openrouter_api_key: str = Query(..., min_length=10),
    wandb_api_key: str | None = Query(None),
    wandb_project: str | None = Query(None),
) -> StreamingResponse:
    irr = json.loads(irrigation_events)
    nit = json.loads(nitrogen_events)
    pho = json.loads(phosphorus_events)
    pot = json.loads(potassium_events)

    farmer_query = _assemble_farmer_query(
        query=query,
        crop=crop,
        variety=variety,
        planting_date=planting_date,
        latitude=latitude,
        longitude=longitude,
        irrigation_events=irr,
        nitrogen_events=nit,
        phosphorus_events=pho,
        potassium_events=pot,
    )
    return StreamingResponse(
        stream_pro(
            farmer_query=farmer_query,
            openrouter_api_key=openrouter_api_key,
            wandb_api_key=wandb_api_key,
            model=model,
            wandb_project=wandb_project,
            # Pass raw sim inputs so stream_pro can compute the hash
            latitude=latitude,
            longitude=longitude,
            crop=crop,
            variety=variety,
            planting_date=planting_date,
            irrigation_events=irr,
            nitrogen_events=nit,
            phosphorus_events=pho,
            potassium_events=pot,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.api_route("/query/reinterpret", methods=["GET", "HEAD"])
async def pro_reinterpret_sse(
    query: str = Query(..., min_length=3),
    sim_hash: str = Query(..., min_length=10, description="SHA-256 sim_hash from a previous /query result event"),
    model: str = Query("claude-sonnet"),
    openrouter_api_key: str = Query(..., min_length=10),
    wandb_api_key: str | None = Query(None),
    wandb_project: str | None = Query(None),
) -> StreamingResponse:
    """
    Re-run only the interpreter step using cached simulation outputs.

    The frontend calls this endpoint when the user changes the question but
    not the simulation inputs (crop, location, dates, applications).  
    Basically, it kinda passes back the sim_hash from the previous result event; 
    that way the backend looks up the cached outputs and skips parsing + simulation entirely.

    Returns the same SSE event stream shape as /query, but only emits:
      - stage(interpreting, start)
      - stage(interpreting, done)
      - result  (with the same sim_hash echoed back)

    Returns 404 if the sim_hash is not in the cache (e.g. server restarted).
    The frontend should fall back to a full /query in that case.
    """
    return StreamingResponse(
        stream_pro_reinterpret(
            farmer_query=query,
            sim_hash=sim_hash,
            openrouter_api_key=openrouter_api_key,
            wandb_api_key=wandb_api_key,
            model=model,
            wandb_project=wandb_project,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/query")
async def pro_query(body: ProQueryRequest) -> StreamingResponse:
    """POST version — accepts a pre-assembled farmer_query JSON body."""
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
