from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared input types
# ---------------------------------------------------------------------------

class IrrigationEvent(BaseModel):
    date: str = Field(..., description="ISO date string, e.g. '2023-06-15'")
    amount_mm: float = Field(..., gt=0, description="Irrigation amount in millimetres")


# ---------------------------------------------------------------------------
# "Pro mode": single model query
# ---------------------------------------------------------------------------

class ProQueryRequest(BaseModel):
    farmer_query: str = Field(..., min_length=10, description="Natural-language query assembled by the frontend")
    openrouter_api_key: str = Field(..., min_length=10)
    wandb_api_key: str | None = Field(None, description="Optional — omit to skip WandB logging")
    model: str = Field("gpt-4o-mini", description="OpenRouter model slug")
    wandb_project: str | None = Field(None, description="WandB project name; ignored if wandb_api_key is None")


# ---------------------------------------------------------------------------
# "Eval mode": multi-model query (like in the paper, to help evalutors assess each models' outputs)
# ---------------------------------------------------------------------------

class EvalQueryRequest(BaseModel):
    farmer_query: str = Field(..., min_length=10)
    openrouter_api_key: str = Field(..., min_length=10)
    wandb_api_key: str | None = Field(None)
    models: list[str] = Field(
        default=["gpt-4o", "llama-3.3-70b"],
        min_length=2,
        description="Two or more model slugs to run in parallel",
    )
    wandb_project: str | None = Field(None)


# ---------------------------------------------------------------------------
# SSE event payloads  (serialised to JSON and sent as SSE data lines)
# ---------------------------------------------------------------------------

class StageEvent(BaseModel):
    event: str = "stage"
    model: str
    stage: str      # "parsing" | "simulating" | "interpreting"
    status: str     # "start" | "done"


class ParsedEvent(BaseModel):
    event: str = "parsed"
    model: str
    lat: float | None = None
    lon: float | None = None
    crop: str | None = None
    variety: str | None = None
    planting_date: str | None = None
    irrigation_events: list[IrrigationEvent] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict, description="Full parser output for debugging")


class QuestionAnswer(BaseModel):
    key: str
    question: str
    matched_question: str   # quite note: this is coerced to str in streaming.py. so pipeline'll return bool when no match
    answer_for_farmer: str
    expert_like_answer: str


class ResultEvent(BaseModel):
    event: str = "result"
    model: str
    answer: str
    answers: list[QuestionAnswer] = Field(default_factory=list)
    yield_kg_ha: float | None = None
    harvest_date: str | None = None
    wandb_run_id: str | None = None
    wandb_run_url: str | None = None
    raw_dssat_output: str | None = Field(None, description="Eval mode only; expert_like_answer")


class ErrorEvent(BaseModel):
    event: str = "error"
    model: str | None = None
    friendly: str
    detail: str | None = None   # raw traceback; always sent, frontend should put it in collapsible


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    