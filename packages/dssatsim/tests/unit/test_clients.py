"""Tests for weather and soil API clients."""

import pytest
import respx
import httpx
import pandas as pd
from unittest.mock import patch

from dssatsim.clients.weather import fetch_weather, WeatherAPIError
from dssatsim.clients.soil import fetch_soil_profile, SoilAPIError


# ---------------------------------------------------------------------------
# Weather client
# ---------------------------------------------------------------------------

MOCK_WEATHER_RESPONSE = {
    "query_lat": 42.24,
    "query_lon": -85.24,
    "nearest_lat": 42.242,
    "nearest_lon": -85.296,
    "distance_km": 4.6,
    "start_date": "2023-05-01",
    "end_date": "2023-05-03",
    "record_count": 3,
    "records": [
        {"date": "2023-05-01", "tmax": 6.86, "tmin": 2.45, "prcp": 10.0, "srad": 141.87,
         "latitude": 42.242, "longitude": -85.296, "dayl": 49913.0, "swe": 0.0, "vp": 728.69},
        {"date": "2023-05-02", "tmax": 6.90, "tmin": 2.59, "prcp": 2.30, "srad": 141.62,
         "latitude": 42.242, "longitude": -85.296, "dayl": 50062.0, "swe": 0.0, "vp": 735.65},
        {"date": "2023-05-03", "tmax": 13.39, "tmin": 3.61, "prcp": 0.0,  "srad": 383.52,
         "latitude": 42.242, "longitude": -85.296, "dayl": 50210.0, "swe": 0.0, "vp": 790.88},
    ],
}


@respx.mock
def test_fetch_weather_returns_dataframe():
    respx.get("http://localhost:8001/weather").mock(
        return_value=httpx.Response(200, json=MOCK_WEATHER_RESPONSE)
    )
    df = fetch_weather(42.24, -85.24, "2023-05-01", "2023-05-03")

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["date", "srad", "tmax", "tmin", "rain"]
    assert len(df) == 3
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


@respx.mock
def test_fetch_weather_raises_on_error():
    respx.get("http://localhost:8001/weather").mock(
        return_value=httpx.Response(403, json={"detail": "Forbidden"})
    )
    with pytest.raises(WeatherAPIError):
        fetch_weather(42.24, -85.24, "2023-05-01", "2023-05-03")


@respx.mock
def test_fetch_weather_rain_column_mapped_from_prcp():
    respx.get("http://localhost:8001/weather").mock(
        return_value=httpx.Response(200, json=MOCK_WEATHER_RESPONSE)
    )
    df = fetch_weather(42.24, -85.24, "2023-05-01", "2023-05-03")
    assert "rain" in df.columns
    assert "prcp" not in df.columns
    assert df["rain"].iloc[0] == 10.0


# ---------------------------------------------------------------------------
# Soil client
# ---------------------------------------------------------------------------

MOCK_SOL_RESPONSE = """\
*US02476497     USA              Loam   200    ISRIC soilgrids + HC27
@SITE        COUNTRY          LAT     LONG SCS Family
 -99              US      42.208   -85.208     HC_GEN0011
@ SCOM  SALB  SLU1  SLDR  SLRO  SLNF  SLPF  SMHB  SMPX  SMKE
    BK  0.10  6.00  0.50 75.00  1.00  1.00 SA001 SA001 SA001
@  SLB  SLMH  SLLL  SDUL  SSAT  SRGF  SSKS  SBDM  SLOC  SLCL  SLSI  SLCF  SLNI  SLHW  SLHB  SCEC  SADC
     5 A     0.119 0.261 0.397  1.00  0.74  1.49  3.01 19.30 45.97 -99.0  0.12  5.72 -99.0 21.10 -99.0
    15 A     0.130 0.272 0.401  0.85  0.62  1.51  2.55 21.18 45.12 -99.0  0.09  5.79 -99.0 18.50 -99.0
"""


@respx.mock
def test_fetch_soil_profile_extracts_profile_name():
    respx.get("https://soil-query-production.up.railway.app/soil").mock(
        return_value=httpx.Response(200, text=MOCK_SOL_RESPONSE)
    )
    # Mock SoilProfile.from_file to avoid needing DSSAT installed in CI
    with patch("dssatsim.clients.soil.SoilProfile.from_file") as mock_from_file:
        mock_from_file.return_value = object()
        fetch_soil_profile(42.24, -85.24)
        called_profile = mock_from_file.call_args[0][0]
        assert called_profile == "US02476497"


@respx.mock
def test_fetch_soil_profile_raises_on_error():
    respx.get("https://soil-query-production.up.railway.app/soil").mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )
    with pytest.raises(SoilAPIError):
        fetch_soil_profile(42.24, -85.24)


