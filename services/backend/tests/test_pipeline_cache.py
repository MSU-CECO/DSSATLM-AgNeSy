from unittest.mock import MagicMock, patch
from backend.pipeline_cache import MAX_SIZE, cache_size, clear_cache, get_pipeline


def _mock_pipeline(*args, **kwargs):
    return MagicMock()


@patch("backend.pipeline_cache.DSSATLMPipeline", side_effect=_mock_pipeline)
def test_cache_miss_creates_pipeline(mock_cls):
    _ = get_pipeline("key-a", None, "gpt-4o-mini")
    assert mock_cls.called
    assert cache_size() == 1


@patch("backend.pipeline_cache.DSSATLMPipeline", side_effect=_mock_pipeline)
def test_cache_hit_returns_same_instance(mock_cls):
    p1 = get_pipeline("key-a", None, "gpt-4o-mini")
    p2 = get_pipeline("key-a", None, "gpt-4o-mini")
    assert p1 is p2
    assert mock_cls.call_count == 1


@patch("backend.pipeline_cache.DSSATLMPipeline", side_effect=_mock_pipeline)
def test_different_keys_create_different_instances(mock_cls):
    p1 = get_pipeline("key-a", None, "gpt-4o-mini")
    p2 = get_pipeline("key-b", None, "gpt-4o-mini")
    assert p1 is not p2
    assert cache_size() == 2


@patch("backend.pipeline_cache.DSSATLMPipeline", side_effect=_mock_pipeline)
def test_different_models_create_different_instances(mock_cls):
    p1 = get_pipeline("key-a", None, "gpt-4o")
    p2 = get_pipeline("key-a", None, "llama-3.3-70b")
    assert p1 is not p2
    assert cache_size() == 2


@patch("backend.pipeline_cache.DSSATLMPipeline", side_effect=_mock_pipeline)
def test_eviction_at_max_size(mock_cls):
    for i in range(MAX_SIZE + 2):
        get_pipeline(f"key-{i}", None, "gpt-4o-mini")
    assert cache_size() == MAX_SIZE


def test_clear_cache_empties():
    clear_cache()
    assert cache_size() == 0
