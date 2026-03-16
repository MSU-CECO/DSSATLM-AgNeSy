"""
Eval query endpoint tests.

Mirrors test_pro_query.py but exercises the /api/eval/query route:
  - X-Eval-Key auth guard
  - Dual-model parallel execution (like how we did in the paper)
  - raw_dssat_output included in result events
  - Both models present in the stream
"""

from unittest.mock import MagicMock, patch

from tests.conftest import DUMMY_OR_KEY, SAMPLE_QUERY
from tests.test_pro_query import MOCK_LOGS, MOCK_OUTPUTS, _make_mock_pipeline, _parse_sse


def test_eval_query_no_key_returns_401(client, eval_env):
    resp = client.post(
        "/api/eval/query",
        json={"farmer_query": SAMPLE_QUERY, "openrouter_api_key": DUMMY_OR_KEY, "models": ["gpt-4o", "llama-3.3-70b"]},
    )
    assert resp.status_code == 401


def test_eval_query_wrong_key_returns_401(client, eval_env):
    resp = client.post(
        "/api/eval/query",
        json={"farmer_query": SAMPLE_QUERY, "openrouter_api_key": DUMMY_OR_KEY, "models": ["gpt-4o", "llama-3.3-70b"]},
        headers={"X-Eval-Key": "wrong-key"},
    )
    assert resp.status_code == 401


def test_eval_query_single_model_returns_422(client, eval_env):
    resp = client.post(
        "/api/eval/query",
        json={"farmer_query": SAMPLE_QUERY, "openrouter_api_key": DUMMY_OR_KEY, "models": ["gpt-4o"]},
        headers={"X-Eval-Key": eval_env},
    )
    assert resp.status_code == 422


@patch("backend.pipeline_cache.DSSATLMPipeline", side_effect=lambda **kw: _make_mock_pipeline())
def test_eval_query_returns_event_stream(mock_cls, client, eval_env):
    resp = client.post(
        "/api/eval/query",
        json={"farmer_query": SAMPLE_QUERY, "openrouter_api_key": DUMMY_OR_KEY, "models": ["gpt-4o", "llama-3.3-70b"]},
        headers={"X-Eval-Key": eval_env},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]


@patch("backend.pipeline_cache.DSSATLMPipeline", side_effect=lambda **kw: _make_mock_pipeline())
def test_eval_query_all_events_have_model_field(mock_cls, client, eval_env):
    resp = client.post(
        "/api/eval/query",
        json={"farmer_query": SAMPLE_QUERY, "openrouter_api_key": DUMMY_OR_KEY, "models": ["gpt-4o", "llama-3.3-70b"]},
        headers={"X-Eval-Key": eval_env},
    )
    events = _parse_sse(resp.text)
    assert all("model" in e for e in events)


@patch("backend.pipeline_cache.DSSATLMPipeline", side_effect=lambda **kw: _make_mock_pipeline())
def test_eval_query_both_models_emit_result(mock_cls, client, eval_env):
    resp = client.post(
        "/api/eval/query",
        json={"farmer_query": SAMPLE_QUERY, "openrouter_api_key": DUMMY_OR_KEY, "models": ["gpt-4o", "llama-3.3-70b"]},
        headers={"X-Eval-Key": eval_env},
    )
    events = _parse_sse(resp.text)
    result_models = {e["model"] for e in events if e["event"] == "result"}
    assert result_models == {"gpt-4o", "llama-3.3-70b"}


@patch("backend.pipeline_cache.DSSATLMPipeline", side_effect=lambda **kw: _make_mock_pipeline())
def test_eval_query_result_includes_raw_dssat(mock_cls, client, eval_env):
    resp = client.post(
        "/api/eval/query",
        json={"farmer_query": SAMPLE_QUERY, "openrouter_api_key": DUMMY_OR_KEY, "models": ["gpt-4o", "llama-3.3-70b"]},
        headers={"X-Eval-Key": eval_env},
    )
    events = _parse_sse(resp.text)
    results = [e for e in events if e["event"] == "result"]
    assert len(results) == 2
    assert all(e["raw_dssat_output"] is not None for e in results)


@patch("backend.pipeline_cache.DSSATLMPipeline", side_effect=lambda **kw: _make_mock_pipeline())
def test_eval_query_three_models_all_present(mock_cls, client, eval_env):
    resp = client.post(
        "/api/eval/query",
        json={
            "farmer_query": SAMPLE_QUERY,
            "openrouter_api_key": DUMMY_OR_KEY,
            "models": ["gpt-4o", "llama-3.3-70b", "claude-sonnet"],
        },
        headers={"X-Eval-Key": eval_env},
    )
    events = _parse_sse(resp.text)
    result_models = {e["model"] for e in events if e["event"] == "result"}
    assert result_models == {"gpt-4o", "llama-3.3-70b", "claude-sonnet"}

