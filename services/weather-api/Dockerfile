FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY services/weather-api/ .

RUN uv pip install --system --no-cache -e .

CMD ["uvicorn", "weather_api.main:app", "--host", "0.0.0.0", "--port", "8001"]
