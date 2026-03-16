from __future__ import annotations

import os

from fastapi import APIRouter, Header, HTTPException, status
from fastapi.responses import StreamingResponse

from ..schemas import EvalQueryRequest
from ..streaming import stream_eval

router = APIRouter(prefix="/api/eval", tags=["eval"])


def _check_eval_key(x_eval_key: str | None) -> None:
    """Raise 401 if the provided key doesn't match the server-side secret."""
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


@router.post("/query")
async def eval_query(
    body: EvalQueryRequest,
    x_eval_key: str | None = Header(None, alias="X-Eval-Key"),
) -> StreamingResponse:
    """
    Run multiple models in parallel and stream interleaved SSE events.

    Requires X-Eval-Key header matching the EVAL_KEY environment variable.

    Each SSE event includes a 'model' field so the frontend can route events
    to the correct column.  Result events additionally include raw_dssat_output
    for the evaluator survey.
    """
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
