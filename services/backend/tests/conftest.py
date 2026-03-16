import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.pipeline_cache import clear_cache


@pytest.fixture(autouse=True)
def reset_cache():
    """Flush the pipeline cache before every test."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def eval_env(monkeypatch):
    """Set a known EVAL_KEY for eval route tests."""
    monkeypatch.setenv("EVAL_KEY", "test-eval-secret")
    yield "test-eval-secret"


DUMMY_OR_KEY = "sk-or-v1-dummy-key-for-testing"
DUMMY_WANDB_KEY = "dummy-wandb-key"
SAMPLE_QUERY = (
    "My farm is at N 42.29, W 85.59. I planted maize on 2023-05-01. "
    "No irrigation. What yield can I expect?"
)
