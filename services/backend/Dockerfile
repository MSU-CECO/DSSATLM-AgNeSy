FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy full monorepo (needed for workspace deps)
COPY . .

# Install backend with workspace deps
RUN uv pip install --system --no-cache \
    -e packages/dssatsim \
    -e packages/dssatlm \
    -e services/backend

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8002"]
