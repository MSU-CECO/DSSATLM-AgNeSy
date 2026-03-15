"""Integration tests for weather and soil API clients.

These tests hit the live APIs and require:
  - Weather API running at WEATHER_API_BASE_URL (default: http://localhost:8001)
  - Soil API reachable at SOIL_API_BASE_URL (default: https://soil-query-production.up.railway.app)

Run with:
    pytest tests/integration/test_integration_clients.py -v

"""

import pytest
import pandas as pd
from DSSATTools.soil import SoilProfile

from dssatsim.clients.weather import fetch_weather, WeatherAPIError
from dssatsim.clients.soil import fetch_soil_profile, SoilAPIError


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# KBS / Kalamazoo area — canonical test location used throughout the project
KBS_LAT = 42.24
KBS_LON = -85.24


# ---------------------------------------------------------------------------
# Weather client
# ---------------------------------------------------------------------------

class TestFetchWeatherIntegration:
    """Live calls to the Weather API."""

    def test_returns_dataframe(self):
        df = fetch_weather(KBS_LAT, KBS_LON, "2023-05-01", "2023-05-31")
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        df = fetch_weather(KBS_LAT, KBS_LON, "2023-05-01", "2023-05-31")
        assert list(df.columns) == ["date", "srad", "tmax", "tmin", "rain"]

    def test_index_is_datetime(self):
        df = fetch_weather(KBS_LAT, KBS_LON, "2023-05-01", "2023-05-31")
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_record_count_matches_date_range(self):
        df = fetch_weather(KBS_LAT, KBS_LON, "2023-05-01", "2023-05-31")
        assert len(df) == 31

    def test_values_are_numeric(self):
        df = fetch_weather(KBS_LAT, KBS_LON, "2023-05-01", "2023-05-31")
        numeric_cols = ["srad", "tmax", "tmin", "rain"]
        assert df[numeric_cols].select_dtypes(include="number").shape[1] == 4

    def test_srad_values_are_non_negative(self):
        df = fetch_weather(KBS_LAT, KBS_LON, "2023-05-01", "2023-05-31")
        assert (df["srad"] >= 0).all()

    def test_tmax_always_gte_tmin(self):
        df = fetch_weather(KBS_LAT, KBS_LON, "2023-05-01", "2023-05-31")
        assert (df["tmax"] >= df["tmin"]).all()

    def test_rain_is_non_negative(self):
        df = fetch_weather(KBS_LAT, KBS_LON, "2023-05-01", "2023-05-31")
        assert (df["rain"] >= 0).all()

    def test_two_year_window_used_in_run(self):
        """Simulate the two-year fetch window that setup_weather() uses."""
        df = fetch_weather(KBS_LAT, KBS_LON, "2023-01-01", "2024-12-31")
        assert len(df) >= 300
        assert df["date"].dt.year.min() >= 2023

    def test_api_returns_data_for_valid_coordinates(self):
        """Weather API should return records for any valid coordinate in dataset."""
        df = fetch_weather(KBS_LAT, KBS_LON, "2023-05-01", "2023-05-31")
        assert len(df) > 0


# ---------------------------------------------------------------------------
# Soil client
# ---------------------------------------------------------------------------

class TestFetchSoilProfileIntegration:
    """Live calls to the soil-query API."""

    def test_returns_soil_profile_instance(self):
        profile = fetch_soil_profile(KBS_LAT, KBS_LON)
        assert isinstance(profile, SoilProfile)

    def test_profile_has_layers(self):
        profile = fetch_soil_profile(KBS_LAT, KBS_LON)
        # SoilProfile is table-like; must have at least one layer
        assert len(profile) > 0

    def test_raises_on_server_error(self, monkeypatch):
        """Smoke-test error handling by pointing at a bogus base URL."""
        import dssatsim.clients.soil as soil_module
        monkeypatch.setattr(soil_module, "SOIL_API_BASE_URL", "http://localhost:9999")
        with pytest.raises((SoilAPIError, Exception)):
            fetch_soil_profile(KBS_LAT, KBS_LON)

