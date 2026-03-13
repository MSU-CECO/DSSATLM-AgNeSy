"""
Data layer — loads the DAYMET parquet file once at startup
and exposes query helpers used by the route handlers.
"""
import math
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import settings

# Columns to keep from the parquet (drops geom, dms strings, metadata)
_KEEP_COLS = [
    "date", "latitude", "longitude",
    "tmax", "tmin", "prcp", "srad", "dayl", "swe", "vp",
]

_df: pd.DataFrame | None = None


def load_parquet() -> None:
    """Load parquet into memory. Called once at app startup."""
    global _df
    path = Path(settings.weather_parquet_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Weather parquet not found at: {path}\n"
            "Check WEATHER_PARQUET_PATH in your .env file."
        )
    raw = pd.read_parquet(path, columns=_KEEP_COLS)
    raw["date"] = pd.to_datetime(raw["date"]).dt.date
    _df = raw


def get_df() -> pd.DataFrame:
    if _df is None:
        raise RuntimeError("Parquet not loaded. Call load_parquet() at startup.")
    return _df


@lru_cache(maxsize=512)
def _unique_coords() -> list[tuple[float, float]]:
    """Return deduplicated (lat, lon) pairs from the dataset."""
    df = get_df()
    return list(df[["latitude", "longitude"]].drop_duplicates().itertuples(index=False, name=None))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def find_nearest(lat: float, lon: float) -> tuple[float, float, float]:
    """
    Return (nearest_lat, nearest_lon, distance_km) for the closest
    grid point in the dataset to the supplied coordinates.
    """
    coords = _unique_coords()
    best_lat, best_lon, best_dist = None, None, float("inf")
    for g_lat, g_lon in coords:
        d = _haversine_km(lat, lon, g_lat, g_lon)
        if d < best_dist:
            best_dist = d
            best_lat, best_lon = g_lat, g_lon
    return best_lat, best_lon, best_dist


def query_weather(
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
) -> tuple[float, float, float, pd.DataFrame]:
    """
    Find the nearest grid point and return filtered weather records.

    Returns:
        (nearest_lat, nearest_lon, distance_km, filtered_dataframe)
    """
    nearest_lat, nearest_lon, distance_km = find_nearest(lat, lon)
    df = get_df()
    mask = (
        (df["latitude"] == nearest_lat)
        & (df["longitude"] == nearest_lon)
        & (df["date"] >= start_date)
        & (df["date"] <= end_date)
    )
    return nearest_lat, nearest_lon, distance_km, df[mask].sort_values("date")


def dataset_summary() -> dict:
    """Return summary stats for the /health endpoint."""
    df = get_df()
    dates = df["date"]
    lats = df["latitude"]
    lons = df["longitude"]
    return {
        "total_records": len(df),
        "date_range": {
            "min": str(dates.min()),
            "max": str(dates.max()),
        },
        "spatial_coverage": {
            "lat_min": float(lats.min()),
            "lat_max": float(lats.max()),
            "lon_min": float(lons.min()),
            "lon_max": float(lons.max()),
        },
    }
