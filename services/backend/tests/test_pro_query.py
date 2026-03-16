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

