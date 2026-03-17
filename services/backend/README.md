# DSSATLM-AgNeSy's Backend Service

FastAPI backend for the DSSATLM-AgNeSy web application. It simply exposes a streaming HTTP API that orchestrates the `dssatlm` pipeline, that we suggest in our paper:
    * parsing a farmer's natural-language query, 
    * running a DSSAT crop simulation, and 
    * interpreting the results via an LLM 
It then streams the progress and results back to the frontend as Server-Sent Events (SSE).

---

## Quick start

```bash
# From the repo root
cd services/backend

# Create .env (see Environment variables below)
cp .env.example .env
# edit .env and fill in your keys

# Install dependencies
uv sync

# Run development server
uv run uvicorn backend.main:app --port 8002 --reload
```

Server starts at `http://localhost:8002`. Interactive docs at `http://localhost:8002/docs`.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key for LLM calls |
| `WANDB_API_KEY` | No | WandB key for experiment logging. If unset, logging is skipped |
| `EVAL_KEY` | Yes (eval mode) | Secret passphrase for the `/api/eval/query` endpoint |
| `CORS_ORIGINS` | No | Comma-separated allowed origins. Defaults to `*` (dev only) |
| `DEBUG` | No | Set to `true` to include full tracebacks in SSE error events. Never set in production. |

Create a `.env` file in `services/backend/`:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
WANDB_API_KEY=...           # optional
EVAL_KEY=your-secret-here
CORS_ORIGINS=*              # tighten in production, e.g. https://your-iis-host.com
```

> **Note:** API keys sent by external users in request bodies are injected into the environment at pipeline instantiation time. They are never stored on disk or logged.

---

## Endpoints

### `GET /health`

Liveness check.

```bash
curl http://localhost:8002/health
# {"status": "ok", "version": "0.1.0"}
```

---

### `GET /api/models`

Returns the list of supported model slugs for Pro mode.

```bash
curl http://localhost:8002/api/models
# {"models": ["gpt-4o", "gpt-4o-mini", "claude-sonnet", "llama-3.3-70b", "dsr1-llama-70b"]}
```

---

### `POST /api/query` aka 'Pro mode'

Runs a single-model DSSAT simulation and streams SSE events.

**Request body:**

```json
{
  "farmer_query": "My farm is at latitude 42.263, longitude -85.648. I planted Maize on May 1st 2023. What yield can I expect?",
  "openrouter_api_key": "sk-or-v1-...",
  "wandb_api_key": null,
  "model": "gpt-4o-mini",
  "wandb_project": null
}
```

**SSE event stream:**

```
data: {"event": "stage", "model": "gpt-4o-mini", "stage": "parsing",      "status": "start"}
data: {"event": "stage", "model": "gpt-4o-mini", "stage": "simulating",   "status": "start"}
data: {"event": "stage", "model": "gpt-4o-mini", "stage": "interpreting", "status": "start"}
data: {"event": "stage", "model": "gpt-4o-mini", "stage": "parsing",      "status": "done"}
data: {"event": "parsed", "model": "gpt-4o-mini", "lat": 42.263, "lon": -85.648, "crop": "Maize", ...}
data: {"event": "stage", "model": "gpt-4o-mini", "stage": "simulating",   "status": "done"}
data: {"event": "stage", "model": "gpt-4o-mini", "stage": "interpreting", "status": "done"}
data: {"event": "result", "model": "gpt-4o-mini", "answer": "...", "yield_kg_ha": 9063.0, "harvest_date": "2023-09-11", ...}
```

On error, a final `error` event is emitted instead of `result`:

```
data: {"event": "error", "model": "gpt-4o-mini", "friendly": "...", "detail": "<traceback>"}
```

---

### `POST /api/eval/query` aka "Eval mode" (similar to the paper's eval methodologies)

Runs multiple models in parallel and streams interleaved SSE events. Requires the `X-Eval-Key` header matching the server-side `EVAL_KEY` environment variable.

**Request body:**

```json
{
  "farmer_query": "...",
  "openrouter_api_key": "sk-or-v1-...",
  "wandb_api_key": null,
  "models": ["gpt-4o", "llama-3.3-70b"],
  "wandb_project": null
}
```

Every SSE event includes a `"model"` field so the frontend can route events to the correct column. `result` events additionally include `raw_dssat_output` (the expert-like ground-truth answer) for use in the evaluator survey.

```bash
# Auth guard — returns 401 with wrong key
curl -s -o /dev/null -w "%{http_code}" \
  -H "X-Eval-Key: wrong" \
  -X POST http://localhost:8002/api/eval/query ...
# 401

# Correct key
curl -N -X POST http://localhost:8002/api/eval/query \
  -H "Content-Type: application/json" \
  -H "X-Eval-Key: your-secret-here" \
  -d '{...}'
```

---

## Running tests

```bash
cd services/backend
uv run pytest -v
```

Expected: **47 tests**, 2 upstream deprecation warnings (DSSATTools, requests). All tests are fully mocked — no real API keys or DSSAT binary required.

```
tests/test_health.py           — liveness endpoint
tests/test_pipeline_cache.py   — LRU cache, eviction, env injection
tests/test_pro_query.py        — SSE event sequence, validation, error handling
tests/test_eval_query.py       — auth guard, multi-model, raw DSSAT output
tests/test_edge_cases.py       — extraction helpers, multi-question, WandB run ID
```

---

## Currently supported language models

| Slug | Provider | Notes |
|---|---|---|
| `gpt-4o` | OpenAI via OpenRouter | Highest quality |
| `gpt-4o-mini` | OpenAI via OpenRouter | Faster, lower cost |
| `claude-sonnet` | Anthropic via OpenRouter | |
| `llama-3.3-70b` | Meta via OpenRouter | Open source |
| `dsr1-llama-70b` | DeepSeek via OpenRouter | Open source |

---

## Deployment

```bash
# Production — bind to all interfaces, no reload
uv run uvicorn backend.main:app \
  --host 0.0.0.0 \
  --port 8002 \
  --workers 2

```

Set `CORS_ORIGINS` to your IIS hostname before deploying:

```bash
CORS_ORIGINS=https://your-iis-host.example.com
```