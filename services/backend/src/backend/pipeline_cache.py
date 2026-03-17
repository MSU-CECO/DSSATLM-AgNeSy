"""
Thread-safe LRU pipeline cache.

Key: (openrouter_api_key, wandb_api_key, model)
Value: DSSATLMPipeline instance

Eviction policy: oldest-inserted entry dropped when MAX_SIZE is reached.

"""
from __future__ import annotations

import logging
import os
import uuid
from collections import OrderedDict
from threading import Lock

try:
    from dssatlm.pipeline import DSSATLMPipeline
except ImportError:
    DSSATLMPipeline = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

MAX_SIZE: int = 10

_cache: OrderedDict[tuple[str, str | None, str], "DSSATLMPipeline"] = OrderedDict()
_lock: Lock = Lock()


def _make_key(openrouter_api_key: str, wandb_api_key: str | None, model: str) -> tuple:
    return (openrouter_api_key, wandb_api_key, model)


WANDB_DISABLED_SENTINEL = "disabled"


def _inject_env(openrouter_api_key: str, wandb_api_key: str | None) -> dict:
    """Set API keys in os.environ; return previous values for restore.

    DSSATLMPipeline._validate_api_keys() requires WANDB_API_KEY to be present
    unconditionally. When the user provides no WandB key we inject a sentinel
    value ('disabled'); the pipeline constructor accepts it and WandB logging
    is skipped because wandb_params=None is passed separately.
    """
    prev = {
        "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY"),
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY"),
    }
    os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
    os.environ["WANDB_API_KEY"] = wandb_api_key if wandb_api_key else WANDB_DISABLED_SENTINEL
    return prev


def _restore_env(prev: dict) -> None:
    """Restore os.environ to values captured by _inject_env."""
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def get_pipeline(
    openrouter_api_key: str,
    wandb_api_key: str | None,
    model: str,
    wandb_project: str | None = None,
) -> DSSATLMPipeline:
    """Return a cached pipeline, creating and caching one if absent."""

    key = _make_key(openrouter_api_key, wandb_api_key, model)

    with _lock:
        if key in _cache:
            logger.debug("Pipeline cache hit: model=%s", model)
            _cache.move_to_end(key)
            return _cache[key]

        # Evict oldest entry if at capacity
        if len(_cache) >= MAX_SIZE:
            evicted_key, _ = _cache.popitem(last=False)
            logger.info("Pipeline cache evicted: model=%s", evicted_key[2])

        logger.info("Pipeline cache miss — instantiating: model=%s", model)

        # Build wandb_params — DSSATLMPipeline passes this directly to wandb.init()
        wandb_params: dict | None = None
        if wandb_api_key and wandb_project:
            wandb_params = {
                "project": wandb_project,
                "name": f"run_{uuid.uuid4().hex[:8]}_{model}",
            }

        _inject_env(openrouter_api_key, wandb_api_key)
        pipeline = DSSATLMPipeline(
            parser_model_id=model,
            interpreter_model_id=model,
            wandb_params=wandb_params,
        )

        _cache[key] = pipeline
        return pipeline


def cache_size() -> int:
    with _lock:
        return len(_cache)


def clear_cache() -> None:
    """Flush the entire cache (useful in tests)."""
    with _lock:
        _cache.clear()
        logger.info("Pipeline cache cleared")

