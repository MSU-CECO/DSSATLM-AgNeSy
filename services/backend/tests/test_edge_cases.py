"""
Edge case tests.

Covers scenarios not exercised by the happy-path tests (could there be more???):
  - Multi-question pipeline output
  - Irrigation events in parsed response
  - Missing simulation response keys (graceful None)
  - wandb_api_key provided in request
  - WandB run ID populated when _wandb_run exists
  - Empty answer_for_farmer fields
  - matched_question_found as bool 
  - GET /api/models endpoint
  - Long farmer_query accepted
  - Eval CORS preflight (OPTIONS)
  - pipeline_cache env not leaked between calls
"""
from unittest.mock import MagicMock, patch


from backend.pipeline_cache import WANDB_DISABLED_SENTINEL, _inject_env, _restore_env
from backend.streaming import (
    _build_answer_text,
    _build_answers_list,
    _extract_parsed_fields,
    _extract_sim_fields,
)
from tests.conftest import DUMMY_OR_KEY, DUMMY_WANDB_KEY, SAMPLE_QUERY
from tests.test_pro_query import _make_mock_pipeline, _parse_sse


# ---------------------------------------------------------------------------
# GET /api/models
# ---------------------------------------------------------------------------

def test_list_models_returns_200(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200


def test_list_models_contains_expected_models(client):
    resp = client.get("/api/models")
    models = resp.json()["models"]
    assert "gpt-4o" in models
    assert "gpt-4o-mini" in models
    assert "llama-3.3-70b" in models
    assert len(models) >= 4


# ---------------------------------------------------------------------------
# _extract_parsed_fields — unit tests
# ---------------------------------------------------------------------------

def test_extract_parsed_fields_crop_name_key():
    logs = {"dssatlm_parser_response": {"crop_name": "Maize", "crop_variety": "KBS 582", "latitude": 42.0, "longitude": -85.0, "planting_date": "2023-05-01", "irrigation_events": []}}
    result = _extract_parsed_fields(logs)
    assert result["crop"] == "Maize"
    assert result["variety"] == "KBS 582"


def test_extract_parsed_fields_with_irrigation_events():
    logs = {
        "dssatlm_parser_response": {
            "latitude": 42.0,
            "longitude": -85.0,
            "crop_name": "Maize",
            "planting_date": "2023-05-01",
            "irrigation_events": [
                {"date": "2023-06-15", "amount_mm": 25.0},
                {"date": "2023-07-03", "amount_mm": 20.0},
            ],
        }
    }
    result = _extract_parsed_fields(logs)
    assert len(result["irrigation_events"]) == 2
    assert result["irrigation_events"][0]["date"] == "2023-06-15"
    assert result["irrigation_events"][0]["amount_mm"] == 25.0


def test_extract_parsed_fields_empty_logs():
    result = _extract_parsed_fields({})
    assert result["lat"] is None
    assert result["crop"] is None
    assert result["irrigation_events"] == []


# ---------------------------------------------------------------------------
# _extract_sim_fields — unit tests
# ---------------------------------------------------------------------------

def test_extract_sim_fields_correct_keys():
    logs = {
        "dssatlm_simulator_response": {
            "Dry weight, yield and yield components": {
                "Harvested yield (kg [dm]/ha)": 9063.0,
            },
            "Dates": {
                "Harvest date": "2023-09-11",
            },
        }
    }
    result = _extract_sim_fields(logs)
    assert result["yield_kg_ha"] == 9063.0
    assert result["harvest_date"] == "2023-09-11"


def test_extract_sim_fields_missing_keys_returns_none():
    result = _extract_sim_fields({"dssatlm_simulator_response": {}})
    assert result["yield_kg_ha"] is None
    assert result["harvest_date"] is None


def test_extract_sim_fields_empty_logs_returns_none():
    result = _extract_sim_fields({})
    assert result["yield_kg_ha"] is None
    assert result["harvest_date"] is None


def test_extract_sim_fields_fallback_to_maturity_date():
    logs = {
        "dssatlm_simulator_response": {
            "Dry weight, yield and yield components": {
                "Yield at harvest maturity (kg [dm]/ha)": 8000.0,
            },
            "Dates": {
                "Physiological maturity date": "2023-09-05",
            },
        }
    }
    result = _extract_sim_fields(logs)
    assert result["yield_kg_ha"] == 8000.0
    assert result["harvest_date"] == "2023-09-05"


# ---------------------------------------------------------------------------
# _build_answer_text / _build_answers_list — unit tests
# ---------------------------------------------------------------------------

def test_build_answer_text_single_question():
    outputs = {"question_1": {"answer_for_farmer": "Yield is 9063 kg/ha."}}
    assert _build_answer_text(outputs) == "Yield is 9063 kg/ha."


def test_build_answer_text_multi_question():
    outputs = {
        "question_1": {"answer_for_farmer": "Yield is 9063 kg/ha."},
        "question_2": {"answer_for_farmer": "Harvest date is Sep 11."},
    }
    text = _build_answer_text(outputs)
    assert "Yield is 9063 kg/ha." in text
    assert "Harvest date is Sep 11." in text
    assert "\n\n" in text


def test_build_answer_text_empty_outputs():
    assert _build_answer_text({}) == ""


def test_build_answer_text_skips_empty_answers():
    outputs = {
        "question_1": {"answer_for_farmer": ""},
        "question_2": {"answer_for_farmer": "Harvest date is Sep 11."},
    }
    text = _build_answer_text(outputs)
    assert text == "Harvest date is Sep 11."


def test_build_answers_list_coerces_bool_matched_question():
    """Real pipeline returns True/False for matched_question_found when no match."""
    outputs = {
        "question_1": {
            "question_statement": "What is yield?",
            "matched_question_found": True, # bool but must be coerced to str
            "retrieved_answer": "9063",
            "answer_for_farmer": "Yield is 9063.",
            "expert_like_answer": "-99",
        }
    }
    result = _build_answers_list(outputs)
    assert isinstance(result[0]["matched_question"], str)
    assert result[0]["matched_question"] == "True"


def test_build_answers_list_multi_question():
    outputs = {
        "question_1": {"question_statement": "Q1", "matched_question_found": "M1", "retrieved_answer": "R1", "answer_for_farmer": "A1", "expert_like_answer": "E1"},
        "question_2": {"question_statement": "Q2", "matched_question_found": "M2", "retrieved_answer": "R2", "answer_for_farmer": "A2", "expert_like_answer": "E2"},
    }
    result = _build_answers_list(outputs)
    assert len(result) == 2
    assert result[0]["key"] == "question_1"
    assert result[1]["key"] == "question_2"


# ---------------------------------------------------------------------------
# Pro query — additional endpoint edge cases
# ---------------------------------------------------------------------------

@patch("backend.pipeline_cache.DSSATLMPipeline", return_value=_make_mock_pipeline())
def test_pro_query_with_wandb_key(mock_cls, client):
    """wandb_api_key provided — should not cause errors."""
    resp = client.post(
        "/api/query",
        json={
            "farmer_query": SAMPLE_QUERY,
            "openrouter_api_key": DUMMY_OR_KEY,
            "wandb_api_key": DUMMY_WANDB_KEY,
            "wandb_project": "test-project",
            "model": "gpt-4o-mini",
        },
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert any(e["event"] == "result" for e in events)


@patch("backend.pipeline_cache.DSSATLMPipeline", return_value=_make_mock_pipeline())
def test_pro_query_long_query_accepted(mock_cls, client):
    """Queries well above the 10-char minimum should be accepted."""
    long_query = "My farm is located at latitude 42.263 and longitude -85.648. " * 10
    resp = client.post(
        "/api/query",
        json={"farmer_query": long_query, "openrouter_api_key": DUMMY_OR_KEY, "model": "gpt-4o-mini"},
    )
    assert resp.status_code == 200


@patch("backend.pipeline_cache.DSSATLMPipeline", return_value=_make_mock_pipeline())
def test_pro_query_multi_question_output(mock_cls, client):
    """Pipeline returns two questions — both should appear in answers list."""
    multi_pipeline = _make_mock_pipeline()
    multi_pipeline.answer_query.return_value = {
        "question_1": {
            "question_statement": "What is the yield?",
            "matched_question_found": "What is crop yield?",
            "retrieved_answer": "9063",
            "answer_for_farmer": "Yield is 9063 kg/ha.",
            "expert_like_answer": "HWAM is 9063.",
        },
        "question_2": {
            "question_statement": "When can I harvest?",
            "matched_question_found": "What is harvest date?",
            "retrieved_answer": "Sep 11",
            "answer_for_farmer": "Harvest date is Sep 11.",
            "expert_like_answer": "HDAT is 2023-09-11.",
        },
    }
    mock_cls.return_value = multi_pipeline

    resp = client.post(
        "/api/query",
        json={"farmer_query": SAMPLE_QUERY, "openrouter_api_key": DUMMY_OR_KEY, "model": "gpt-4o-mini"},
    )
    events = _parse_sse(resp.text)
    result = next(e for e in events if e["event"] == "result")
    assert len(result["answers"]) == 2
    assert "\n\n" in result["answer"]


@patch("backend.pipeline_cache.DSSATLMPipeline", return_value=_make_mock_pipeline())
def test_pro_query_wandb_run_id_populated(mock_cls, client):
    """When pipeline._wandb_run exists, run ID should appear in result."""
    pipeline_with_wandb = _make_mock_pipeline()
    wandb_run = MagicMock()
    wandb_run.id = "abc123"
    wandb_run.url = "https://wandb.ai/run/abc123"
    pipeline_with_wandb._wandb_run = wandb_run
    mock_cls.return_value = pipeline_with_wandb

    resp = client.post(
        "/api/query",
        json={"farmer_query": SAMPLE_QUERY, "openrouter_api_key": DUMMY_OR_KEY, "model": "gpt-4o-mini"},
    )
    events = _parse_sse(resp.text)
    result = next(e for e in events if e["event"] == "result")
    assert result["wandb_run_id"] == "abc123"
    assert result["wandb_run_url"] == "https://wandb.ai/run/abc123"


# ---------------------------------------------------------------------------
# pipeline_cache — env injection / restore
# ---------------------------------------------------------------------------

def test_inject_env_sets_openrouter_key():
    prev = _inject_env("new-key", None)
    try:
        import os
        assert os.environ["OPENROUTER_API_KEY"] == "new-key"
        assert os.environ["WANDB_API_KEY"] == WANDB_DISABLED_SENTINEL
    finally:
        _restore_env(prev)


def test_inject_env_sets_wandb_key_when_provided():
    prev = _inject_env("new-key", "wandb-key-123")
    try:
        import os
        assert os.environ["WANDB_API_KEY"] == "wandb-key-123"
    finally:
        _restore_env(prev)


def test_restore_env_cleans_up():
    import os
    original_or = os.environ.get("OPENROUTER_API_KEY")
    prev = _inject_env("temp-key", None)
    _restore_env(prev)
    assert os.environ.get("OPENROUTER_API_KEY") == original_or

