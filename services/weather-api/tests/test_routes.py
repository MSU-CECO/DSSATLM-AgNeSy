"""
Basic tests for the Weather API routes.
Uses a small in-memory DataFrame to avoid needing the real parquet file.
"""
import os
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Patch settings before importing the app
os.environ.setdefault("WEATHER_PARQUET_PATH", "/tmp/dummy.parquet")
os.environ.setdefault("WEATHER_API_KEY", "test-key")

from src.weather_api.main import app  # noqa: E402

MOCK_DF = pd.DataFrame([
    {
        "date": date(2023, 11, 21),
        "latitude": 42.4241716982,
        "longitude": -85.7479303509,
        "tmax": 6.54,
        "tmin": 2.20,
        "prcp": 5.39,
        "srad": 74.65,
        "dayl": 33641.72,
        "swe": 0.0,
        "vp": 715.67,
    },
    {
        "date": date(2023, 11, 22),
        "latitude": 42.4241716982,
        "longitude": -85.7479303509,
        "tmax": 6.99,
        "tmin": 1.13,
        "prcp": 0.0,
        "srad": 133.57,
        "dayl": 33538.02,
        "swe": 0.0,
        "vp": 662.81,
    },
])

HEADERS = {"X-API-Key": "test-key"}


@pytest.fixture(autouse=True)
def mock_data():
    with patch("src.weather_api.data._df", MOCK_DF):
        with patch("src.weather_api.data._unique_coords.cache_clear"):
            yield


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["parquet_loaded"] is True


def test_weather_success():
    response = client.get(
        "/weather",
        params={
            "lat": 42.42,
            "lon": -85.75,
            "start_date": "2023-11-21",
            "end_date": "2023-11-22",
        },
        headers=HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["record_count"] == 2
    assert len(data["records"]) == 2


def test_weather_no_api_key():
    response = client.get(
        "/weather",
        params={
            "lat": 42.42,
            "lon": -85.75,
            "start_date": "2023-11-21",
            "end_date": "2023-11-22",
        },
    )
    assert response.status_code == 401


def test_weather_invalid_date_range():
    response = client.get(
        "/weather",
        params={
            "lat": 42.42,
            "lon": -85.75,
            "start_date": "2023-11-25",
            "end_date": "2023-11-21",
        },
        headers=HEADERS,
    )
    assert response.status_code == 422
