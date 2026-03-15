"""Weather API client.

Fetches daily weather records from the DSSATLM-AgNeSy Weather API service
(services/weather-api) and returns a DataFrame ready for WeatherStation().
"""

import httpx
import pandas as pd
from dssatsim.config import WEATHER_API_BASE_URL, WEATHER_API_KEY


class WeatherAPIError(Exception):
    """Raised when the Weather API returns an unexpected response."""


def fetch_weather(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch daily weather records for the nearest grid point.

    Args:
        lat: Latitude of the target location.
        lon: Longitude of the target location.
        start_date: Start date in YYYY-MM-DD format (YYYY-MM-DD).
        end_date: End date in YYYY-MM-DD format (YYYY-MM-DD).

    Returns:
        DataFrame with columns: date, srad, tmax, tmin, rain.
        Column names are lowercase to match DSSATTools v3 WeatherRecord
        parameter names. srad is converted from W/m² (API) to MJ/m²/day
        (DSSAT requirement) by multiplying by 0.0864.

    Raises:
        WeatherAPIError: If the API returns a non-200 response.
    """
    url = f"{WEATHER_API_BASE_URL}/weather"
    headers = {"X-API-Key": WEATHER_API_KEY}
    params = {
        "lat": lat,
        "lon": lon,
        "start_date": start_date,
        "end_date": end_date,
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, headers=headers, params=params)

    if response.status_code != 200:
        raise WeatherAPIError(
            f"Weather API returned {response.status_code}: {response.text}"
        )

    payload = response.json()
    df = pd.DataFrame(payload["records"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"prcp": "rain"})
    
    # DSSAT expects SRAD in MJ/m²/day; the API returns W/m² — convert
    df["srad"] = df["srad"] * 0.0864
    df = df[["date", "srad", "tmax", "tmin", "rain"]]

    return df
