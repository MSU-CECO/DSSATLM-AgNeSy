from datetime import date
from typing import Optional
from pydantic import BaseModel, Field


class WeatherRecord(BaseModel):
    date: date
    latitude: float
    longitude: float
    tmax: float = Field(description="Maximum temperature, °C")
    tmin: float = Field(description="Minimum temperature, °C")
    prcp: float = Field(description="Precipitation, mm")
    srad: float = Field(description="Solar radiation, W/m²")
    dayl: float = Field(description="Day length, seconds")
    swe: float = Field(description="Snow water equivalent, mm")
    vp: float = Field(description="Vapor pressure, Pa")


class WeatherResponse(BaseModel):
    query_lat: float
    query_lon: float
    nearest_lat: float
    nearest_lon: float
    distance_km: float
    start_date: date
    end_date: date
    record_count: int
    records: list[WeatherRecord]


class HealthResponse(BaseModel):
    status: str
    version: str
    parquet_loaded: bool
    total_records: Optional[int] = None
    date_range: Optional[dict] = None
    spatial_coverage: Optional[dict] = None
