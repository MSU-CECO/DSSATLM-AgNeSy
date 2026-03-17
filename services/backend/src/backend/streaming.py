"""
SSE streaming utilities.

"""
from __future__ import annotations

import asyncio
import json
import logging
import traceback
from collections.abc import AsyncGenerator
from typing import Any

from .pipeline_cache import get_pipeline
from . import sim_cache
from .schemas import (
    ErrorEvent,
    ParsedEvent,
    ResultEvent,
    StageEvent,
)

logger = logging.getLogger(__name__)

FRIENDLY_ERRORS: list[tuple[str, str]] = [
    ("ValueError", "The query could not be parsed — please rephrase your question."),
    ("TimeoutError", "The simulation timed out. Try a simpler query or try again."),
    ("TimeoutException", "The simulation timed out. Try a simpler query or try again."),
    ("ConnectTimeout", "The simulation timed out. Try a simpler query or try again."),
    ("ReadTimeout", "The simulation timed out. Try a simpler query or try again."),
    ("AuthenticationError", "Invalid API key — please check your OpenRouter or WandB key."),
    ("UnauthorizedError", "Invalid API key — please check your OpenRouter or WandB key."),
]


def _friendly_message(exc: Exception) -> str:
    exc_name = type(exc).__name__
    for name, msg in FRIENDLY_ERRORS:
        if exc_name == name:
            return msg
    return "An unexpected error occurred. Please try again."


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _stage(model: str, stage: str, status: str) -> str:
    return _sse(StageEvent(model=model, stage=stage, status=status).model_dump())


import os as _os

_DEBUG = _os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")


def _error_sse(exc: Exception, model: str | None = None) -> str:
    return _sse(
        ErrorEvent(
            model=model,
            friendly=_friendly_message(exc),
            detail=traceback.format_exc() if _DEBUG else None,
        ).model_dump()
    )


def _extract_parsed_fields(logs: dict) -> dict:
    """
    Pull structured input fields from dssatlm_parser_response in logs.
    """
    parser_resp = logs.get("dssatlm_parser_response", {})
    irrigation_raw = parser_resp.get("irrigation_events") or []
    irrigation = []
    for ev in irrigation_raw:
        if isinstance(ev, dict) and "date" in ev:
            irrigation.append({
                "date": ev.get("date", ""),
                "amount_mm": float(ev.get("amount_mm") or ev.get("amount") or 0),
            })
    return {
        "lat": parser_resp.get("latitude") or parser_resp.get("lat"),
        "lon": parser_resp.get("longitude") or parser_resp.get("lon"),
        "crop": parser_resp.get("crop_name") or parser_resp.get("crop"),
        "variety": parser_resp.get("crop_variety") or parser_resp.get("variety"),
        "planting_date": parser_resp.get("planting_date"),
        "irrigation_events": irrigation,
        "raw": parser_resp,
    }


def _extract_sim_fields(logs: dict) -> dict:
    """Pull yield and harvest date from dssatlm_simulator_response."""
    sim = logs.get("dssatlm_simulator_response", {})
    dry_weight = sim.get("Dry weight, yield and yield components", {})
    dates = sim.get("Dates", {})
    yield_kg_ha = (
        dry_weight.get("Harvested yield (kg [dm]/ha)")
        or dry_weight.get("Yield at harvest maturity (kg [dm]/ha)")
    )
    harvest_date = (
        dates.get("Harvest date")
        or dates.get("Physiological maturity date")
    )
    return {
        "yield_kg_ha": float(yield_kg_ha) if yield_kg_ha is not None else None,
        "harvest_date": str(harvest_date) if harvest_date is not None else None,
    }


def _build_answer_text(outputs: dict) -> str:
    """
    outputs is keyed 'question_1', 'question_2', ...
    Each value has 'answer_for_farmer'. Join them into one readable block.
    """
    if not outputs:
        return ""
    parts = [
        q_data.get("answer_for_farmer", "")
        for q_data in outputs.values()
        if q_data.get("answer_for_farmer")
    ]
    return "\n\n".join(parts)


def _build_answers_list(outputs: dict) -> list[dict]:
    """Return the full structured per-question output list for the frontend."""
    return [
        {
            "key": q_key,
            "question": q_data.get("question_statement", ""),
            "matched_question": str(q_data.get("matched_question_found") or ""),
            "answer_for_farmer": q_data.get("answer_for_farmer", ""),
            "expert_like_answer": q_data.get("expert_like_answer", ""),
        }
        for q_key, q_data in outputs.items()
    ]


def _get_wandb_run_id(pipeline) -> str | None:
    try:
        run = getattr(pipeline, "_wandb_run", None)
        return run.id if run else None
    except Exception:
        return None


def _get_wandb_run_url(pipeline) -> str | None:
    try:
        run = getattr(pipeline, "_wandb_run", None)
        return run.url if run else None
    except Exception:
        return None


async def stream_pro(
    farmer_query: str,
    openrouter_api_key: str,
    wandb_api_key: str | None,
    model: str,
    wandb_project: str | None,
    # Simulation input fields — used to compute sim_hash for caching
    latitude: float | None = None,
    longitude: float | None = None,
    crop: str | None = None,
    variety: str | None = None,
    planting_date: str | None = None,
    irrigation_events: list | None = None,
    nitrogen_events: list | None = None,
    phosphorus_events: list | None = None,
    potassium_events: list | None = None,
) -> AsyncGenerator[str, None]:
    """SSE generator for Pro mode — single model, full pipeline run."""

    loop = asyncio.get_event_loop()

    # Compute sim_hash upfront so we can attach it to the result event
    computed_hash: str | None = None
    if all(v is not None for v in [latitude, longitude, crop, variety, planting_date]):
        computed_hash = sim_cache.compute_sim_hash(
            latitude=latitude,
            longitude=longitude,
            crop=crop,
            variety=variety,
            planting_date=planting_date,
            irrigation_events=irrigation_events or [],
            nitrogen_events=nitrogen_events or [],
            phosphorus_events=phosphorus_events or [],
            potassium_events=potassium_events or [],
        )

    try:
        pipeline = get_pipeline(openrouter_api_key, wandb_api_key, model, wandb_project)
    except Exception as exc:
        logger.exception("Failed to instantiate pipeline")
        yield _error_sse(exc, model)
        return

    yield _stage(model, "parsing", "start")
    yield _stage(model, "simulating", "start")
    yield _stage(model, "interpreting", "start")

    try:
        outputs: dict = await loop.run_in_executor(
            None, pipeline.answer_query, farmer_query
        )
    except Exception as exc:
        logger.exception("Pipeline execution failed")
        yield _stage(model, "parsing", "done")
        yield _stage(model, "simulating", "done")
        yield _stage(model, "interpreting", "done")
        yield _error_sse(exc, model)
        return

    logs = pipeline.get_logs()

    if not logs.get("pipeline_ran_successfully", True):
        errors = logs.get("execution_errors", {})
        error_text = " | ".join(v for v in errors.values() if v)
        exc = RuntimeError(error_text or "Pipeline failed — check DSSAT inputs.")
        yield _stage(model, "parsing", "done")
        yield _stage(model, "simulating", "done")
        yield _stage(model, "interpreting", "done")
        yield _error_sse(exc, model)
        return

    # Store outputs in sim_cache so /reinterpret can retrieve them later
    if computed_hash is not None:
        sim_cache.store(computed_hash, outputs, logs)

    yield _stage(model, "parsing", "done")
    yield _sse(ParsedEvent(model=model, **_extract_parsed_fields(logs)).model_dump())
    yield _stage(model, "simulating", "done")
    yield _stage(model, "interpreting", "done")

    sim_fields = _extract_sim_fields(logs)

    yield _sse(
        ResultEvent(
            model=model,
            answer=_build_answer_text(outputs),
            answers=_build_answers_list(outputs),
            yield_kg_ha=sim_fields["yield_kg_ha"],
            harvest_date=sim_fields["harvest_date"],
            wandb_run_id=_get_wandb_run_id(pipeline),
            wandb_run_url=_get_wandb_run_url(pipeline),
            raw_dssat_output=None,
            sim_hash=computed_hash,
        ).model_dump()
    )


async def stream_pro_reinterpret(
    farmer_query: str,
    sim_hash: str,
    openrouter_api_key: str,
    wandb_api_key: str | None,
    model: str,
    wandb_project: str | None,
) -> AsyncGenerator[str, None]:
    """
    SSE generator for the /reinterpret endpoint.

    Skips parsing and simulation entirely — looks up cached outputs by
    sim_hash and re-runs only the interpreter step with the new question.
    Emits only: interpreting/start, interpreting/done, result (or error).
    """
    loop = asyncio.get_event_loop()

    # Look up cached simulation outputs
    cached = sim_cache.get(sim_hash)
    if cached is None:
        logger.warning("Reinterpret requested but hash not in cache: %s", sim_hash[:12])
        try:
            raise KeyError(f"Simulation cache miss for hash {sim_hash[:12]}…")
        except KeyError as exc:
            yield _sse(
                ErrorEvent(
                    model=model,
                    code="cache_miss",
                    friendly=_friendly_message(exc),
                    detail=traceback.format_exc() if _DEBUG else None,
                ).model_dump()
            )
        return

    cached_logs: dict = cached["logs"]
    
    # Pass the DSSAT simulator response (not the interpreter outputs) so that
    # answer_query_interpret_only feeds real simulation data into InterpreterModule.
    cached_sim_outputs: dict = cached_logs.get("dssatlm_simulator_response", {})

    try:
        pipeline = get_pipeline(openrouter_api_key, wandb_api_key, model, wandb_project)
    except Exception as exc:
        logger.exception("Failed to instantiate pipeline for reinterpret")
        yield _error_sse(exc, model)
        return

    yield _stage(model, "interpreting", "start")

    try:
        # answer_query_interpret_only takes the cached DSSAT sim outputs and
        # re-runs only the DSPy InterpreterModule with the new question text.
        outputs: dict = await loop.run_in_executor(
            None,
            pipeline.answer_query_interpret_only,
            farmer_query,
            cached_sim_outputs,
        )
    except Exception as exc:
        logger.exception("Reinterpret pipeline execution failed")
        yield _stage(model, "interpreting", "done")
        yield _error_sse(exc, model)
        return

    yield _stage(model, "interpreting", "done")

    sim_fields = _extract_sim_fields(cached_logs)

    yield _sse(
        ResultEvent(
            model=model,
            answer=_build_answer_text(outputs),
            answers=_build_answers_list(outputs),
            yield_kg_ha=sim_fields["yield_kg_ha"],
            harvest_date=sim_fields["harvest_date"],
            wandb_run_id=_get_wandb_run_id(pipeline),
            wandb_run_url=_get_wandb_run_url(pipeline),
            raw_dssat_output=None,
            sim_hash=sim_hash,  
        ).model_dump()
    )


async def stream_eval(
    farmer_query: str,
    openrouter_api_key: str,
    wandb_api_key: str | None,
    models: list[str],
    wandb_project: str | None,
) -> AsyncGenerator[str, None]:
    """
    SSE generator for Eval mode: multiple models run concurrently.

    Events from all models are interleaved on a single stream; each event
    carries a 'model' field so the frontend can route to the correct column.
    Result events include raw_dssat_output (expert_like_answer) for the survey.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()
    active = len(models)

    async def run_one(model: str) -> None:
        try:
            pipeline = get_pipeline(openrouter_api_key, wandb_api_key, model, wandb_project)

            await queue.put(_stage(model, "parsing", "start"))
            await queue.put(_stage(model, "simulating", "start"))
            await queue.put(_stage(model, "interpreting", "start"))

            outputs: dict = await loop.run_in_executor(
                None, pipeline.answer_query, farmer_query
            )
            logs = pipeline.get_logs()

            if not logs.get("pipeline_ran_successfully", True):
                errors = logs.get("execution_errors", {})
                error_text = " | ".join(v for v in errors.values() if v)
                raise RuntimeError(error_text or "Pipeline failed.")

            await queue.put(_stage(model, "parsing", "done"))
            await queue.put(_sse(
                ParsedEvent(model=model, **_extract_parsed_fields(logs)).model_dump()
            ))
            await queue.put(_stage(model, "simulating", "done"))
            await queue.put(_stage(model, "interpreting", "done"))

            sim_fields = _extract_sim_fields(logs)

            raw_dssat = None
            if outputs:
                first = next(iter(outputs.values()), {})
                raw_dssat = first.get("expert_like_answer")

            await queue.put(_sse(
                ResultEvent(
                    model=model,
                    answer=_build_answer_text(outputs),
                    answers=_build_answers_list(outputs),
                    yield_kg_ha=sim_fields["yield_kg_ha"],
                    harvest_date=sim_fields["harvest_date"],
                    wandb_run_id=_get_wandb_run_id(pipeline),
                    wandb_run_url=_get_wandb_run_url(pipeline),
                    raw_dssat_output=raw_dssat,
                    sim_hash=None,  # not relevant in eval mode
                ).model_dump()
            ))

        except Exception as exc:
            logger.exception("Eval pipeline failed for model=%s", model)
            await queue.put(_error_sse(exc, model))
        finally:
            await queue.put(None)  # sentinel

    tasks = [asyncio.create_task(run_one(m)) for m in models]

    while active > 0:
        item = await queue.get()
        if item is None:
            active -= 1
        else:
            yield item

    await asyncio.gather(*tasks, return_exceptions=True)
    