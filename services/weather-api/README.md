# weather-api

FastAPI microservice for DAYMET daily weather data at 1km resolution (currently limited to the Kalamazoo area, year 2023).
Part of the **DSSATLM-AgNeSy** monorepo.

---

## Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | None | Service health + dataset summary |
| `GET` | `/weather` | `X-API-Key` | Weather records for nearest grid point |

### `GET /weather` — Query Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `lat` | float | ✅ | Latitude in decimal degrees |
| `lon` | float | ✅ | Longitude in decimal degrees |
| `start_date` | date | ✅ | Start date (YYYY-MM-DD) |
| `end_date` | date | ✅ | End date (YYYY-MM-DD) |

### Example Request

```bash
curl -H "X-API-Key: your-key-here" \
  "http://localhost:8001/weather?lat=42.42&lon=-85.75&start_date=2023-11-21&end_date=2023-11-25"
```

### Example Response

```json
{
  "query_lat": 42.42,
  "query_lon": -85.75,
  "nearest_lat": 42.4241716982,
  "nearest_lon": -85.7479303509,
  "distance_km": 0.412,
  "start_date": "2023-11-21",
  "end_date": "2023-11-25",
  "record_count": 5,
  "records": [
    {
      "date": "2023-11-21",
      "latitude": 42.4241716982,
      "longitude": -85.7479303509,
      "tmax": 6.54,
      "tmin": 2.2,
      "prcp": 5.39,
      "srad": 74.65,
      "dayl": 33641.72,
      "swe": 0.0,
      "vp": 715.67
    }
  ]
}
```

---

## Setup

### 1. Copy and configure `.env`

```bash
cp .env.example .env
# Edit .env with your actual values
```

### 2. Install dependencies (using uv)

```bash
uv venv
uv pip install -e ".[dev]"
```

### 3. Run

```bash
python run.py
# or
uvicorn src.weather_api.main:app --host 0.0.0.0 --port 8001
```

Interactive docs available at: `http://localhost:8001/docs`

### 4. Run tests

```bash
pytest tests/
```

---

## Deployment (Linux server)

The service runs on port `8001` by default. Windows IIS reverse proxies
`/api/weather/` -> `http://<linux-private-ip>:8001/`.

Make sure port `8001` is open in the Linux firewall:

```bash
sudo ufw allow 8001/tcp
```
