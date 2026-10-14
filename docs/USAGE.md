# HEMS-MCP: Closed-Loop Forecast Reasoning

HEMS-MCP forecasts household electricity demand and solar production, summarizes energy and surplus windows, and explores EV charging schedules through a local MCP service. The offline demo uses reproducible synthetic data and a seasonal-naive forecast. An optional Chronos-Bolt backend executes an actual model on CPU.

The system supports decisions. It does not operate a charger.

## Install and run

Python 3.12 and [uv](https://docs.astral.sh/uv/) are recommended.

```sh
uv sync --python 3.12 --group dev
uv run hems-mcp demo --output-dir artifacts/demo --days 14 --seed 42
uv run hems-mcp forecast --input artifacts/demo/history.csv --backend seasonal --horizon 24 --output artifacts/forecast.json
uv run hems-mcp summarize --input artifacts/forecast.json --output artifacts/summary.json
uv run hems-mcp schedule --input artifacts/forecast.json --energy-kwh 10 --output artifacts/schedule.json
uv run hems-mcp evaluate --input artifacts/demo/history.csv --origins 3 --output-dir artifacts/evaluation --output artifacts/evaluation.json
uv run python -m pytest tests -q
uv run ruff check .
```

The demo writes history, forecasts, summaries, schedules, and evaluation reports. Commands accept a positional input path or `--input`. `-` reads JSON from stdin. History inputs accept CSV or JSON records. Outputs are JSON; omission of `--output` writes to stdout. `python -m hems_mcp` also works.

Each history row has `timestamp`, `load_kw`, and `solar_kw`. Power is the average kW during the interval starting at the timestamp. Energy is `sum(power) * interval_minutes / 60` kWh. Timestamps require timezone offsets and normalize to UTC. Input must be ordered, unique, regular, finite, and nonnegative. Validation discards extra columns. Maximum history is 35,040 rows. Seasonal forecasts support up to 168 slots; Chronos supports up to 64. Evaluation supports 1–48 slots and 1–20 non-overlapping origins, with a complete daily training window before the first origin. Evaluation requires sampling intervals that divide one day.

## MCP

```sh
uv run hems-mcp serve --transport stdio
```

Register this command in an MCP client with this project as its working directory. Tools are `generate_demo_history`, `forecast_energy`, `summarize_energy`, and `explore_ev_schedule`. The service uses the official MCP Python SDK v1 maintenance API. Stdio owns stdout. Model work runs outside the event loop with bounded admission. The server accepts history records; it does not expose arbitrary filesystem reads.

Scheduling prioritizes available solar, then lower household net demand. Charging power is AC input; requested energy is battery energy after constant efficiency. Arrival is inclusive and departure is exclusive. Infeasible requests return a safe partial allocation and shortfall. Existing household grid-cap violations remain visible. The greedy policy does not prove tariff or peak optimality.

## Optional Chronos

The backend uses [amazon/chronos-bolt-tiny](https://huggingface.co/amazon/chronos-bolt-tiny), float32 CPU inference, median predictions, and up to 2,048 historical slots. Model initialization is lazy and cached once per process. Requested Chronos failures terminate the forecast without substituting the seasonal backend. Negative model values are clipped and recorded.

```sh
uv sync --python 3.12 --group dev --extra chronos
uv run --extra chronos hems-mcp forecast artifacts/demo/history.csv --backend chronos --output artifacts/chronos.json
```

PyPI's Linux torch installation can include large CUDA dependencies. For CPU-only experimentation, install the optional packages directly into the existing environment with the CPU index and run the environment's Python to avoid uv resynchronizing it:

```sh
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python 'chronos-forecasting>=2,<3'
.venv/bin/python -m hems_mcp forecast artifacts/demo/history.csv --backend chronos --output artifacts/chronos.json
```

A first real forecast downloads checkpoint weights. Dependency installation or mocked inference does not establish real model execution.

## Configured LLM experiment

No credentials are required for the offline workflow. Evaluation skips every LLM condition unless a configuration file is explicitly supplied. No real provider calls run in tests.

Create `llm-config.json` with a provider model and a base URL **including `/v1`** when the provider requires it:

```json
{
  "base_url": "https://api.openai.com/v1",
  "model": "YOUR_MODEL",
  "api_key_env": "HEMS_LLM_API_KEY",
  "max_completion_tokens": 2048,
  "temperature": 0
}
```

Set the referenced environment variable through your secret manager or shell environment. Never put a key in this JSON. `base_url` and `model` may instead come from `HEMS_LLM_BASE_URL` and `HEMS_LLM_MODEL` when the explicit JSON file is `{}`. A local endpoint can run without a key. Set `temperature` to `null` if the model does not support temperature.

```sh
uv run hems-mcp evaluate artifacts/demo/history.csv --origins 3 --llm-config llm-config.json --output-dir artifacts/llm-evaluation
```

This command makes paid calls when the configured provider charges. Each origin runs three conditions using identical EV constraints and completion budgets:

| Condition | Input representation | Required output |
|---|---|---|
| `llm_only` | Last 512 permitted historical observations, future timestamps | Load, solar, and charging arrays |
| `numeric_curves` | Backend forecast arrays | Charging array |
| `semantic_summary` | Computed energy, peaks, and surplus-window summaries | Charging array |

The backend and LLM-only forecaster can use different context lengths. Numeric and semantic conditions share one backend forecast. The semantic condition receives less slot detail. Payload bytes measure serialization size; they are not token counts. These differences limit causal claims about representation quality.

The adapter uses the [OpenAI-compatible Chat Completions request shape and usage fields](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create). Other endpoints must support `max_completion_tokens`. Responses must contain a JSON object without markdown. The adapter bounds completion tokens, response bytes, connect/read timeouts, and checks a 90-second elapsed budget between response chunks. This check can overrun by a read timeout while waiting for a chunk. Requests do not retry automatically.

Each original proposal retains its forecast-based and held-out realized feasibility checks. An infeasible valid proposal receives at most one feedback request. Failed or still-infeasible corrections retain their outcomes and expose a separately labeled deterministic fallback. Constraints remain fixed in this LLM feedback loop. Malformed proposals fail explicitly. The offline constraint-feedback example separately extends an insufficient departure window and recomputes the numerical schedule. Neither loop represents hardware control.

## Evaluation artifacts and interpretation

Expanding-window origins train strictly before their next held-out window. Later origins may use observations from previous windows. Reports contain:

- `results.json`: provenance, versions, original plans, feedback attempts, forecast and realized feasibility, summaries, timing, and nullable provider usage.
- `metrics.csv`: original per-origin/condition/target errors. Assisted conditions without their own forecast arrays have blank forecast metrics; their forecast source is the shared backend.
- `curves.csv`: actual and predicted per-slot powers for evaluated forecasts.
- `report.md`: aggregate backend errors and limitations.

MAE and RMSE use kW; signed energy bias uses kWh. Aggregate errors include every held-out slot with explicit sample counts. Solar zeros do not require MAPE. Wall time and local process CPU seconds are measured separately for forecasts, summaries, schedules, and HTTP calls. Local CPU excludes provider compute. RSS is the entire process lifetime peak, measured in bytes; it cannot isolate model memory. Missing provider usage stays null. Skipped arms have null metrics. Provider compute remains null. Synthetic data does not establish real-household accuracy.

The initial system models one EV, continuous charging power, fixed availability and constant efficiency. It excludes tariffs, minimum current, phases, battery state-of-charge curves, weather forecasts, and grid uncertainty. Forecast-feasible plans can violate the actual cap when demand differs from forecasts. Detailed design and public contracts appear in [the implementation handoff](IMPLEMENTATION_HANDOFF.md).
