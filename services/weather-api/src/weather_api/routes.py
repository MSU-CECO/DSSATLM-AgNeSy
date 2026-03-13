from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from starlette import status

from .auth import require_api_key
from .data import dataset_summary, get_df, query_weather
from .schemas import HealthResponse, WeatherRecord, WeatherResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """Public health check — no API key required."""
    try:
        summary = dataset_summary()
        return HealthResponse(
            status="ok",
            version="0.1.0",
            parquet_loaded=True,
            **summary,
        )
    except RuntimeError:
        return HealthResponse(
            status="degraded",
            version="0.1.0",
            parquet_loaded=False,
        )


@router.get(
    "/weather",
    response_model=WeatherResponse,
    tags=["weather"],
    dependencies=[Depends(require_api_key)],
)
async def get_weather(
    lat: float = Query(..., description="Latitude in decimal degrees", ge=-90, le=90),
    lon: float = Query(..., description="Longitude in decimal degrees", ge=-180, le=180),
    start_date: date = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: date = Query(..., description="End date (YYYY-MM-DD)"),
) -> WeatherResponse:
    """
    Return daily weather records for the nearest grid point to (lat, lon)
    within the requested date range.

    Requires `X-API-Key` header.
    """
    if end_date < start_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="end_date must be >= start_date.",
        )
    if (end_date - start_date).days > 3650:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Date range cannot exceed 10 years (3650 days).",
        )

    nearest_lat, nearest_lon, distance_km, df = query_weather(lat, lon, start_date, end_date)

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No weather records found for the nearest grid point "
                f"({nearest_lat}, {nearest_lon}) in the requested date range "
                f"{start_date} to {end_date}."
            ),
        )

    records = [
        WeatherRecord(
            date=row.date,
            latitude=row.latitude,
            longitude=row.longitude,
            tmax=row.tmax,
            tmin=row.tmin,
            prcp=row.prcp,
            srad=row.srad,
            dayl=row.dayl,
            swe=row.swe,
            vp=row.vp,
        )
        for row in df.itertuples(index=False)
    ]

    return WeatherResponse(
        query_lat=lat,
        query_lon=lon,
        nearest_lat=nearest_lat,
        nearest_lon=nearest_lon,
        distance_km=round(distance_km, 3),
        start_date=start_date,
        end_date=end_date,
        record_count=len(records),
        records=records,
    )
