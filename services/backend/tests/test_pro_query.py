"""
Pro query endpoint tests.

All tests mock DSSATLMPipeline (i.e., no real LLM or DSSAT calls are made).
The mock reflects the actual pipeline.py API:
  - answer_query() returns dict keyed 'question_1', 'question_2', ...
  - get_logs() returns the full _logs dict
"""
import json
from unittest.mock import MagicMock, patch

from tests.conftest import DUMMY_OR_KEY, SAMPLE_QUERY

# Matches real DSSATLMPipeline._logs structure
MOCK_LOGS = {
    "pipeline_ran_successfully": True,
    "simulation_is_possible": True,
    "simulation_is_successful": True,
    "question_statements_parsed": ["What would be my crop yield?"],
    "dssatlm_parser_response": {
        "latitude": 42.29,
        "longitude": -85.59,
        "crop_name": "maize",
        "crop_variety": "MZ GREAT LAKES 582 KBS",
        "planting_date": "2023-05-01",
        "irrigation_events": [],
        "question_statements": ["What would be my crop yield?"],
    },
    "dssatlm_simulator_response": {
        "Dry weight, yield and yield components": {
            "Harvested yield (kg [dm]/ha)": 9063.0,
        },
        "Dates": {
            "Harvest date": "2023-09-11",
        },
    },
    "dssatlm_interpreter_response": {},
    "outputs": {},
    "execution_errors": {
        "step_1_parsing": "",
        "step_2_simulation": "",
        "step_3_interpreting": "",
        "unexpected": "",
    },
}

# Matches real answer_query() return value
MOCK_OUTPUTS = {
    "question_1": {
        "question_statement": "What would be my crop yield?",
        "matched_question_found": "What is the crop yield at maturity?",
        "retrieved_answer": "Yield: 9063 kg/ha",
        "answer_for_farmer": "Based on the simulation, your maize crop is expected to yield approximately 9,063 kg/ha.",
        "expert_like_answer": "The HWAM is 9063.0. Here is more definition: Harvested yield at maturity.",
    }
}


def _make_mock_pipeline():
    pipeline = MagicMock()
    pipeline.answer_query.return_value = MOCK_OUTPUTS
    pipeline.get_logs.return_value = MOCK_LOGS
    pipeline._wandb_run = None
    return pipeline


def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("data:"):
            events.append(json.loads(line[5:].strip()))
    return events


@patch("backend.pipeline_cache.DSSATLMPipeline", return_value=_make_mock_pipeline())
def test_pro_query_returns_event_stream(mock_cls, client):
    resp = client.post(
        "/api/query",
        json={"farmer_query": SAMPLE_QUERY, "openrouter_api_key": DUMMY_OR_KEY, "model": "gpt-4o-mini"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]


@patch("backend.pipeline_cache.DSSATLMPipeline", return_value=_make_mock_pipeline())
def test_pro_query_stage_sequence(mock_cls, client):
    resp = client.post(
        "/api/query",
        json={"farmer_query": SAMPLE_QUERY, "openrouter_api_key": DUMMY_OR_KEY, "model": "gpt-4o-mini"},
    )
    events = _parse_sse(resp.text)
    stage_events = [e for e in events if e["event"] == "stage"]
    # 3 start + 3 done = 6 stage events
    assert len(stage_events) == 6
    starts = [e for e in stage_events if e["status"] == "start"]
    dones = [e for e in stage_events if e["status"] == "done"]
    assert {e["stage"] for e in starts} == {"parsing", "simulating", "interpreting"}
    assert {e["stage"] for e in dones} == {"parsing", "simulating", "interpreting"}


@patch("backend.pipeline_cache.DSSATLMPipeline", return_value=_make_mock_pipeline())
def test_pro_query_has_parsed_event(mock_cls, client):
    resp = client.post(
        "/api/query",
        json={"farmer_query": SAMPLE_QUERY, "openrouter_api_key": DUMMY_OR_KEY, "model": "gpt-4o-mini"},
    )
    events = _parse_sse(resp.text)
    parsed = next((e for e in events if e["event"] == "parsed"), None)
    assert parsed is not None
    assert parsed["lat"] == 42.29
    assert parsed["crop"] == "maize"


@patch("backend.pipeline_cache.DSSATLMPipeline", return_value=_make_mock_pipeline())
def test_pro_query_result_fields(mock_cls, client):
    resp = client.post(
        "/api/query",
        json={"farmer_query": SAMPLE_QUERY, "openrouter_api_key": DUMMY_OR_KEY, "model": "gpt-4o-mini"},
    )
    events = _parse_sse(resp.text)
    result = next((e for e in events if e["event"] == "result"), None)
    assert result is not None
    assert result["yield_kg_ha"] == 9063.0
    assert result["harvest_date"] == "2023-09-11"
    assert "9,063" in result["answer"]
    assert len(result["answers"]) == 1
    assert result["raw_dssat_output"] is None   # not exposed in pro mode


@patch("backend.pipeline_cache.DSSATLMPipeline", return_value=_make_mock_pipeline())
def test_pro_query_result_ends_stream(mock_cls, client):
    resp = client.post(
        "/api/query",
        json={"farmer_query": SAMPLE_QUERY, "openrouter_api_key": DUMMY_OR_KEY, "model": "gpt-4o-mini"},
    )
    events = _parse_sse(resp.text)
    assert events[-1]["event"] == "result"


def test_pro_query_missing_openrouter_key_returns_422(client):
    resp = client.post("/api/query", json={"farmer_query": SAMPLE_QUERY})
    assert resp.status_code == 422


def test_pro_query_short_query_returns_422(client):
    resp = client.post(
        "/api/query",
        json={"farmer_query": "hi", "openrouter_api_key": DUMMY_OR_KEY},
    )
    assert resp.status_code == 422


@patch("backend.pipeline_cache.DSSATLMPipeline", return_value=_make_mock_pipeline())
def test_pro_query_pipeline_failure_emits_error_event(mock_cls, client):
    failed_pipeline = _make_mock_pipeline()
    failed_pipeline.get_logs.return_value = {
        **MOCK_LOGS,
        "pipeline_ran_successfully": False,
        "execution_errors": {"step_2_simulation": "Simulation not possible.", "step_1_parsing": "", "step_3_interpreting": "", "unexpected": ""},
    }
    failed_pipeline.answer_query.return_value = {}
    mock_cls.return_value = failed_pipeline

    resp = client.post(
        "/api/query",
        json={"farmer_query": SAMPLE_QUERY, "openrouter_api_key": DUMMY_OR_KEY, "model": "gpt-4o-mini"},
    )
    events = _parse_sse(resp.text)
    error = next((e for e in events if e["event"] == "error"), None)
    assert error is not None
    assert error["friendly"] != ""



# ---------------------------------------------------------------------------
# GET /api/query — sim_hash in result event
# ---------------------------------------------------------------------------

@patch("backend.pipeline_cache.DSSATLMPipeline", return_value=_make_mock_pipeline())
def test_get_pro_query_result_contains_sim_hash(mock_cls, client):
    """Full GET /api/query run must include sim_hash in the result event."""
    from tests.conftest import DUMMY_OR_KEY
    resp = client.get(
        "/api/query",
        params={
            "query": "What yield can I expect?",
            "crop": "maize",
            "variety": "MZ GREAT LAKES 582 KBS",
            "planting_date": "2023-05-01",
            "latitude": 42.263,
            "longitude": -85.648,
            "model": "gpt-4o-mini",
            "openrouter_api_key": DUMMY_OR_KEY,
        },
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    result = next((e for e in events if e["event"] == "result"), None)
    assert result is not None
    assert result["sim_hash"] is not None
    assert len(result["sim_hash"]) == 64


@patch("backend.pipeline_cache.DSSATLMPipeline", return_value=_make_mock_pipeline())
def test_get_pro_query_populates_sim_cache(mock_cls, client):
    """After a successful GET /api/query, the sim_cache must hold the entry."""
    from backend import sim_cache
    from backend.sim_cache import compute_sim_hash
    from tests.conftest import DUMMY_OR_KEY

    sim_cache.clear_cache()

    resp = client.get(
        "/api/query",
        params={
            "query": "What yield can I expect?",
            "crop": "maize",
            "variety": "MZ GREAT LAKES 582 KBS",
            "planting_date": "2023-05-01",
            "latitude": 42.263,
            "longitude": -85.648,
            "model": "gpt-4o-mini",
            "openrouter_api_key": DUMMY_OR_KEY,
        },
    )
    events = _parse_sse(resp.text)
    result = next(e for e in events if e["event"] == "result")
    sim_hash = result["sim_hash"]

    entry = sim_cache.get(sim_hash)
    assert entry is not None
    assert "outputs" in entry
    assert "logs" in entry


# ---------------------------------------------------------------------------
# GET /api/query/reinterpret
# ---------------------------------------------------------------------------

def test_reinterpret_returns_error_event_when_hash_not_in_cache(client):
    """Cold request with unknown hash must return 200 SSE stream with an error event."""
    from tests.conftest import DUMMY_OR_KEY
    resp = client.get(
        "/api/query/reinterpret",
        params={
            "query": "What yield can I expect?",
            "sim_hash": "a" * 64,
            "model": "gpt-4o-mini",
            "openrouter_api_key": DUMMY_OR_KEY,
        },
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    error = next((e for e in events if e["event"] == "error"), None)
    assert error is not None
    assert error["code"] == "cache_miss"


@patch("backend.pipeline_cache.DSSATLMPipeline")
def test_reinterpret_returns_event_stream_on_cache_hit(mock_cls, client):
    """With a warm cache, /reinterpret must return 200 text/event-stream."""
    from backend import sim_cache
    from backend.sim_cache import compute_sim_hash
    from tests.conftest import DUMMY_OR_KEY

    # Pre-populate sim_cache
    sim_hash = compute_sim_hash(
        latitude=42.263, longitude=-85.648, crop="maize",
        variety="MZ GREAT LAKES 582 KBS", planting_date="2023-05-01",
        irrigation_events=[], nitrogen_events=[],
        phosphorus_events=[], potassium_events=[],
    )
    sim_cache.store(sim_hash, MOCK_OUTPUTS, MOCK_LOGS)

    # Wire up mock pipeline with answer_query_interpret_only
    mock_pipeline = _make_mock_pipeline()
    mock_pipeline.answer_query_interpret_only = MagicMock(return_value=MOCK_OUTPUTS)
    mock_cls.return_value = mock_pipeline

    resp = client.get(
        "/api/query/reinterpret",
        params={
            "query": "What yield can I expect?",
            "sim_hash": sim_hash,
            "model": "gpt-4o-mini",
            "openrouter_api_key": DUMMY_OR_KEY,
        },
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]


@patch("backend.pipeline_cache.DSSATLMPipeline")
def test_reinterpret_only_emits_interpreting_stage(mock_cls, client):
    """Reinterpret stream must contain only interpreting stage events — no parsing or simulating."""
    from backend import sim_cache
    from backend.sim_cache import compute_sim_hash
    from tests.conftest import DUMMY_OR_KEY

    sim_hash = compute_sim_hash(
        latitude=42.263, longitude=-85.648, crop="maize",
        variety="MZ GREAT LAKES 582 KBS", planting_date="2023-05-01",
        irrigation_events=[], nitrogen_events=[],
        phosphorus_events=[], potassium_events=[],
    )
    sim_cache.store(sim_hash, MOCK_OUTPUTS, MOCK_LOGS)

    mock_pipeline = _make_mock_pipeline()
    mock_pipeline.answer_query_interpret_only = MagicMock(return_value=MOCK_OUTPUTS)
    mock_cls.return_value = mock_pipeline

    resp = client.get(
        "/api/query/reinterpret",
        params={
            "query": "What yield can I expect?",
            "sim_hash": sim_hash,
            "model": "gpt-4o-mini",
            "openrouter_api_key": DUMMY_OR_KEY,
        },
    )
    events = _parse_sse(resp.text)
    stage_events = [e for e in events if e["event"] == "stage"]
    stages_present = {e["stage"] for e in stage_events}

    assert "parsing" not in stages_present
    assert "simulating" not in stages_present
    assert "interpreting" in stages_present


@patch("backend.pipeline_cache.DSSATLMPipeline")
def test_reinterpret_result_echoes_sim_hash(mock_cls, client):
    """The result event from /reinterpret must echo back the same sim_hash."""
    from backend import sim_cache
    from backend.sim_cache import compute_sim_hash
    from tests.conftest import DUMMY_OR_KEY

    sim_hash = compute_sim_hash(
        latitude=42.263, longitude=-85.648, crop="maize",
        variety="MZ GREAT LAKES 582 KBS", planting_date="2023-05-01",
        irrigation_events=[], nitrogen_events=[],
        phosphorus_events=[], potassium_events=[],
    )
    sim_cache.store(sim_hash, MOCK_OUTPUTS, MOCK_LOGS)

    mock_pipeline = _make_mock_pipeline()
    mock_pipeline.answer_query_interpret_only = MagicMock(return_value=MOCK_OUTPUTS)
    mock_cls.return_value = mock_pipeline

    resp = client.get(
        "/api/query/reinterpret",
        params={
            "query": "When can I harvest?",
            "sim_hash": sim_hash,
            "model": "gpt-4o-mini",
            "openrouter_api_key": DUMMY_OR_KEY,
        },
    )
    events = _parse_sse(resp.text)
    result = next((e for e in events if e["event"] == "result"), None)
    assert result is not None
    assert result["sim_hash"] == sim_hash


@patch("backend.pipeline_cache.DSSATLMPipeline")
def test_reinterpret_calls_interpret_only_not_answer_query(mock_cls, client):
    """answer_query must never be called on the reinterpret path."""
    from backend import sim_cache
    from backend.sim_cache import compute_sim_hash
    from tests.conftest import DUMMY_OR_KEY

    sim_hash = compute_sim_hash(
        latitude=42.263, longitude=-85.648, crop="maize",
        variety="MZ GREAT LAKES 582 KBS", planting_date="2023-05-01",
        irrigation_events=[], nitrogen_events=[],
        phosphorus_events=[], potassium_events=[],
    )
    sim_cache.store(sim_hash, MOCK_OUTPUTS, MOCK_LOGS)

    mock_pipeline = _make_mock_pipeline()
    mock_pipeline.answer_query_interpret_only = MagicMock(return_value=MOCK_OUTPUTS)
    mock_cls.return_value = mock_pipeline

    client.get(
        "/api/query/reinterpret",
        params={
            "query": "What yield can I expect?",
            "sim_hash": sim_hash,
            "model": "gpt-4o-mini",
            "openrouter_api_key": DUMMY_OR_KEY,
        },
    )

    mock_pipeline.answer_query.assert_not_called()
    mock_pipeline.answer_query_interpret_only.assert_called_once()
    