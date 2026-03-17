"""
Thread-safe simulation output cache.

Key:   sim_hash (str) — SHA-256 of the simulation inputs that feed DSSAT
                        (latitude, longitude, crop, variety, planting_date,
                         irrigation_events, nitrogen_events, phosphorus_events,
                         potassium_events)
Value: dict with keys:
         outputs — return value of pipeline.answer_query()
         logs    — return value of pipeline.get_logs()

Eviction policy: oldest-inserted entry dropped when MAX_SIZE is reached.

The hash is computed by the backend (in streaming.py) so it is canonical and
cannot be spoofed by the frontend.  The frontend stores the opaque hash string
and sends it back on /api/query/reinterpret requests.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from threading import Lock

logger = logging.getLogger(__name__)

MAX_SIZE: int = 50

_cache: OrderedDict[str, dict] = OrderedDict()
_lock: Lock = Lock()


def compute_sim_hash(
    latitude: float,
    longitude: float,
    crop: str,
    variety: str,
    planting_date: str,
    irrigation_events: list,
    nitrogen_events: list,
    phosphorus_events: list,
    potassium_events: list,
) -> str:
    """Return a stable SHA-256 hex digest of the simulation inputs.

    Lists are sorted before serialisation so that event ordering does not
    produce different hashes for semantically identical inputs.
    """
    payload = {
        "latitude": round(float(latitude), 6),
        "longitude": round(float(longitude), 6),
        "crop": crop.strip().lower(),
        "variety": (variety or "").strip().lower(),
        "planting_date": planting_date,
        "irrigation_events": sorted(irrigation_events, key=lambda e: (e[0] if isinstance(e, list) else e.get("date", ""))),
        "nitrogen_events": sorted(nitrogen_events, key=lambda e: (e[0] if isinstance(e, list) else e.get("date", ""))),
        "phosphorus_events": sorted(phosphorus_events, key=lambda e: (e[0] if isinstance(e, list) else e.get("date", ""))),
        "potassium_events": sorted(potassium_events, key=lambda e: (e[0] if isinstance(e, list) else e.get("date", ""))),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def store(sim_hash: str, outputs: dict, logs: dict) -> None:
    """Store simulation outputs under sim_hash, evicting oldest if at capacity."""
    with _lock:
        if sim_hash in _cache:
            _cache.move_to_end(sim_hash)
            _cache[sim_hash] = {"outputs": outputs, "logs": logs}
            logger.debug("Sim cache updated: hash=%s", sim_hash[:12])
            return

        if len(_cache) >= MAX_SIZE:
            evicted, _ = _cache.popitem(last=False)
            logger.info("Sim cache evicted: hash=%s", evicted[:12])

        _cache[sim_hash] = {"outputs": outputs, "logs": logs}
        logger.info("Sim cache stored: hash=%s", sim_hash[:12])


def get(sim_hash: str) -> dict | None:
    """Return cached entry or None if not found."""
    with _lock:
        entry = _cache.get(sim_hash)
        if entry is not None:
            _cache.move_to_end(sim_hash)
            logger.debug("Sim cache hit: hash=%s", sim_hash[:12])
        else:
            logger.debug("Sim cache miss: hash=%s", sim_hash[:12])
        return entry


def cache_size() -> int:
    with _lock:
        return len(_cache)


def clear_cache() -> None:
    """Flush the entire cache (useful in tests)."""
    with _lock:
        _cache.clear()
        logger.info("Sim cache cleared")
