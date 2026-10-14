# HEMS-MCP implementation handoff

## Outcome and scope

Build a runnable Python project in `hems-mcp/` named **HEMS-MCP: Closed-Loop Forecast Reasoning**. The project exposes household electricity-load and solar-production forecasts through an asyncio MCP service. Pandas and NumPy calculate peaks, energy totals, and solar surplus. A deterministic scheduler explores feasible EV charging plans. A rolling-origin evaluation compares a forecasting backend, an LLM that sees history alone, and LLM reasoning that receives numeric forecast curves or semantic summaries.

The first release must run offline with deterministic synthetic data and a clearly identified seasonal-naive backend. The Chronos path must execute a real installed model when explicitly selected. Optional dependency absence, download failure, and missing LLM configuration must produce honest diagnostics. No function may silently substitute seasonal forecasts for a requested Chronos model. No report may invent LLM tokens, forecast accuracy, latency, provider compute, or successful model execution.

The product is an experiment and decision-support service. It does not control a charger. Its closed loop consists of forecast, summarize, explore a constrained plan, inspect feasibility, revise constraints, and recompute. Hardware telemetry, charger actuation, billing, distributed deployment, model training, weather acquisition, and a web dashboard fall outside the initial release.

## Runtime and upstream decisions

Use Python 3.12 through uv. Set `requires-python = ">=3.11,<3.14"` because the machine's default Python 3.14 may outrun optional ML dependency support. Use a `src/hems_mcp/` package, argparse, NumPy, Pandas, Pydantic, httpx, pytest, and the official MCP SDK. Keep PyTorch and `chronos-forecasting` in an optional `chronos` extra. Keep a resolved lockfile after installation and scoped tests establish compatibility.

The official Chronos repository supports Chronos-2 and earlier model families. Select `amazon/chronos-bolt-tiny` for the initial CPU-oriented adapter because it has 9 million parameters. Chronos-2 remains a later adapter option. The model identity must appear in every forecast artifact. Source: [official Chronos repository](https://github.com/amazon-science/chronos-forecasting).

The model card exposes `ChronosBoltPipeline.from_pretrained` and dataframe forecasting. The implementation may use tensor quantile prediction to avoid dataframe version differences. It must validate the actual return shape against the installed version. Source: [official Chronos Bolt tiny model card](https://huggingface.co/amazon/chronos-bolt-tiny).

Use `mcp>=1.28,<2` and `mcp.server.fastmcp.FastMCP` for this implementation. The v1 branch is a supported maintenance line; v2 is now the current stable release. The version bound deliberately preserves the chosen API. Do not describe v1 as the latest release. Source: [official MCP v1 README](https://raw.githubusercontent.com/modelcontextprotocol/python-sdk/v1.x/README.md). A future migration can adopt MCPServer independently of domain functions.

## Ownership and dependency boundaries

Three implementation workers can operate independently after agreeing to the public signatures below. Assign GPT-6.1 Sol medium to core numerical work, GPT-6.1 Sol low or medium to service plumbing, and GPT-6.1 Sol medium to evaluation. Every worker must preserve other workers' edits. Workers must communicate contract changes before implementing them.

| Owner | Files and responsibility | Dependency |
|---|---|---|
| Core | `src/hems_mcp/schemas.py`, `data.py`, `forecast.py`, `summary.py`, `schedule.py`, `tests/test_core.py` | NumPy, Pandas; optional lazy Chronos imports |
| Service | `pyproject.toml`, `src/hems_mcp/__init__.py`, `__main__.py`, `cli.py`, `server.py`, `tests/test_service.py` | Public core functions; lazy evaluation import |
| Evaluation and documentation | `src/hems_mcp/evaluation.py`, `llm.py`, `tests/test_evaluation.py`, `README.md`, `.env.example` | Public core functions; httpx |
| Integrator | Lockfile, integration fixes, final acceptance evidence, generated ignored artifacts | Completed modules |

Only core owns common data validation. Service must not duplicate forecast or scheduling math. Evaluation must not modify package metadata independently. Imports must point toward core; core must never import service or evaluation. Keep all tool transport objects out of core APIs.

## Canonical module contracts

All functions below return JSON-compatible dictionaries unless their signature specifies DataFrame. Public results contain native Python floats, integers, lists, strings, booleans, and nulls. JSON serialization must reject NaN and infinity.

```python
# data.py
DEFAULT_COLUMNS = ("timestamp", "load_kw", "solar_kw")
def generate_demo(days: int = 14, seed: int = 42,
                  interval_minutes: int = 60) -> pd.DataFrame: ...
def validate_history(df: pd.DataFrame) -> pd.DataFrame: ...

# forecast.py
def forecast_frame(history: pd.DataFrame, horizon_steps: int = 24,
                   backend: str = "seasonal",
                   model_id: str = "amazon/chronos-bolt-tiny") -> dict: ...

# summary.py
def summarize_forecast(forecast: dict) -> dict: ...

# schedule.py
def explore_schedule(forecast: dict, energy_kwh: float = 10.0,
                     max_charge_kw: float = 7.2,
                     grid_limit_kw: float = 10.0,
                     efficiency: float = 0.9,
                     arrival_step: int = 0,
                     departure_step: int | None = None) -> dict: ...

# evaluation.py
def run_evaluation(history: pd.DataFrame, output_dir: str | Path,
                   backend: str = "seasonal", horizon_steps: int = 24,
                   origins: int = 3, llm_config: dict | None = None) -> dict: ...
```

These signatures form the integration contract. Optional keyword-only extensions are acceptable after coordination. Preserve the listed default arguments. `departure_step=None` means the end of the horizon. The departure index is exclusive.

A forecast has these mandatory keys:

```json
{
  "backend": "seasonal",
  "model_id": null,
  "interval_minutes": 60,
  "timestamps": ["2026-01-15T00:00:00+00:00"],
  "load_kw": [1.2],
  "solar_kw": [0.0],
  "warnings": [],
  "timing": {"wall_seconds": 0.001, "cpu_seconds": 0.001}
}
```

The illustrative numbers above describe shape only. Generated reports must contain measured values. Add `schema_version`, `history_end`, `horizon_steps`, `context_steps`, and `device` when useful. Optional quantiles must declare their probabilities and preserve the same horizon. Do not require optional quantiles for the service or scheduler.

Exceptions should distinguish invalid input (`ValueError` or a shared subclass), unavailable backend (`RuntimeError` subclass), and transient provider failure. Do not expose local credentials or authorization headers in exception strings. User-facing errors should identify the invalid field and accepted range.

## Time-series and unit rules

A row represents average nonnegative power during the interval that starts at `timestamp`. `load_kw` is household demand before EV charging. `solar_kw` is solar generation. Every timestamp must include a timezone and must normalize to UTC. A naive timestamp must fail validation. A UTC timeline avoids ambiguous local daylight-saving intervals.

Require all three columns, at least two records, finite numeric values, strictly increasing unique timestamps, and a constant positive interval. Reject duplicate, shuffled, missing-interval, negative, NaN, and infinite observations. Reject booleans as numeric configuration values. Validation returns a defensive normalized copy and never mutates caller data. Extra columns may remain unused or be explicitly discarded; document the choice.

Support intervals that evenly divide 24 hours. Cap input history at 35,040 rows. Seasonal forecasts allow 168 steps, Chronos allows 64 steps, and evaluation allows 48 steps. A daily seasonal forecast requires at least one full day. Use the final daily cycle repeatedly for horizons beyond one cycle. The forecast starts exactly one interval after the final observed timestamp. Forecasting must never read rows after the requested origin.

Energy equals `sum(power_kw) * interval_minutes / 60`. Never sum kW values and label the result kWh without multiplying by interval duration. Define signed net demand as `load_kw - solar_kw`, grid import as `max(net, 0)`, and surplus as `max(-net, 0)`.

Synthetic generation uses an explicit fixed start timestamp, a local NumPy random generator, smooth daily demand and solar profiles, and bounded stochastic variation. Clamp generated demand and solar at zero. A fixed seed, duration, and interval must reproduce the same data. Include enough variability to exercise zero solar, daytime surplus, and evening demand. Label every fixture and report as synthetic. Do not imply performance on real households.

## Forecasting implementation

Seasonal mode repeats the most recent complete daily cycle separately for load and solar. Set `backend="seasonal"`, `model_id=null`, and a warning or description stating that this is a seasonal-naive offline baseline. This path must not import torch, initialize a model, access the network, or inspect API credentials.

Chronos mode lazily imports PyTorch and the Chronos package. Load `amazon/chronos-bolt-tiny` on CPU in float32 by default. Reuse a single loaded pipeline per process and serialize model inference. Bound model caching to one configured identity. Do not let MCP callers supply arbitrary remote model identifiers. Use a configured allowlist or a fixed server model.

Batch the two target series as a tensor with shape `[2, context_steps]`. Use at most 2,048 historical steps. The shipped ChronosBoltPipeline.predict returns its built-in quantiles; select the configured median. Verify outputs have two series, the requested horizon, finite values, and the expected quantile axis. Use the median for `load_kw` and `solar_kw`. Avoid interpreting a quantile tensor as a sample tensor. The official adapter source defines the callable behavior: [Chronos Bolt pipeline source](https://raw.githubusercontent.com/amazon-science/chronos-forecasting/main/src/chronos/chronos_bolt.py).

Clamp negative model predictions to zero and record that physical postprocessing occurred. Do not invent an astronomical night mask without location and timezone metadata. Report model loading time separately from warm inference when available. Record truncation of context. Dependency, checkpoint, device, and output-shape errors must terminate that forecast with a clear diagnostic. A requested backend must remain visible in failed evaluation rows.

## Numeric and semantic summaries

Calculate load energy, solar energy, imported energy, surplus energy, peak load and its earliest timestamp, peak import and its earliest timestamp, and contiguous positive-surplus windows. Use NumPy for vectorized calculations and Pandas timestamps for window boundaries. A window starts at its first slot timestamp and ends one interval after its last slot timestamp.

The summary dictionary should expose machine-readable fields such as `load_energy_kwh`, `solar_energy_kwh`, `import_energy_kwh`, `surplus_energy_kwh`, `peak_load_kw`, `peak_load_timestamp`, and `surplus_windows`. Include a concise `text` string generated from the same computed fields. Specify whether rounding is presentation-only. Use original precision for all scheduling math.

Summary mode must omit full arrays from the LLM payload so token comparisons measure an actual representation difference. Include enough timing information to support a charging decision, such as surplus window start/end, available surplus kWh, and peak periods. Do not claim that a compressed summary contains all information in the numeric curves.

## EV scheduling and invariants

`energy_kwh` denotes energy delivered to the EV battery. `max_charge_kw` denotes power drawn at the household AC boundary. Charging efficiency must satisfy `0 < efficiency <= 1`. The charging power array has one value per forecast interval. Require nonnegative finite energy, charger power and grid import limit, and `0 <= arrival_step <= departure_step <= horizon_steps`. Empty availability is an explicit infeasible case for positive energy. A zero-energy request returns a valid zero schedule.

For slot i, compute `base_net_i = load_i - solar_i`. The maximum additional charging power permitted by the import cap is `max(0, grid_limit_kw - base_net_i)`. Slot charging capacity is the minimum of that value and charger power, with zero capacity outside the availability window.

If household base import already exceeds the cap in any slot, report that condition explicitly. EV charging cannot repair it. A zero charging schedule must not be labeled globally grid-feasible when the household already breaches the cap.

Allocate energy deterministically. First consume available solar surplus in chronological order, then use remaining capacity in ascending base-net-demand order with chronological tie-breaking. Split the final allocation exactly to avoid overcharging. This greedy policy prioritizes solar and low demand. It does not prove a globally minimal peak or tariff cost. No optimizer dependency is required.

Return the agreed integration keys: `status` (`feasible` or `infeasible`), `requested_energy_kwh`, `delivered_energy_kwh`, `unmet_energy_kwh`, `charging_energy_kwh`, `charging_kw`, `timestamps`, `baseline_import_energy_kwh`, `scheduled_import_energy_kwh`, `baseline_peak_import_kw`, `scheduled_peak_import_kw`, and `warnings`. `charging_energy_kwh` measures AC input; delivered energy measures battery energy. Optional fields include `grid_import_kw`, `baseline_grid_violations`, `strategy`, `arrival_step`, `departure_step`, and efficiency. Feasibility requires sufficient delivered battery energy and no grid-cap violations. An infeasible request returns the maximal safe partial plan and its shortfall. Do not alter the request to manufacture success.

Validate these invariants with tolerance near 1e-8:

1. Every charge value is finite, nonnegative, and no greater than charger power.
2. Charge is zero before arrival and at or after departure.
3. Grid import equals `max(load + charge - solar, 0)`.
4. No schedule-induced grid violation occurs.
5. Battery energy equals `efficiency * sum(charge_kw) * dt_hours`.
6. Delivered energy does not exceed requested energy beyond tolerance.
7. `unmet_energy_kwh = max(requested_energy_kwh - delivered_energy_kwh, 0)`.
8. `status` reflects both energy completion and baseline cap violations.

Evaluation must also perform a bounded correction cycle: validate each proposed schedule, preserve its original metrics, issue deterministic feasibility feedback, and recompute a corrected schedule with the numeric scheduler. Store original and corrected plans under distinct labels. A configured LLM may instead receive one feedback request, with its additional tokens and latency recorded separately. Never silently repair the original arm.

A closed-loop example must make an infeasible short availability request, inspect the shortfall, extend departure, and obtain a recomputed result. Save both outcomes as evidence. Do not describe this as physical charger control.

## Async MCP service and CLI

Expose typed MCP tools named `forecast_energy`, `summarize_energy`, and `explore_ev_schedule` through FastMCP. The fourth tool is `generate_demo_history`. Accept bounded history records or a configured dataset handle. Do not expose arbitrary server filesystem reads through MCP. CLI users may provide explicit local CSV paths.

`forecast` accepts history records, horizon, and the allowed backend. `summarize` accepts a forecast object. `schedule` accepts a forecast and the scheduler parameters. Tool descriptions must state units, timezone rules, exclusive departure, and whether data is synthetic. Preserve structured tool errors so callers can revise invalid requests.

Async handlers run blocking work outside the event loop. Serialize forecast work with a one-worker executor or an equivalent mechanism. Bound admission to prevent an unbounded pending queue; reject excess jobs with a retryable busy error. Use a small request bound such as four active-plus-waiting jobs and a 120-second response budget. Cancellation of `asyncio.to_thread` does not stop its thread. Keep the model slot occupied until the actual worker finishes, or explicitly document that inference is allowed to complete after client cancellation. Do not release a semaphore early and accidentally execute several model calls concurrently.

Default transport is stdio. Protocol output owns stdout; logs and progress belong on stderr. Optional Streamable HTTP must bind to loopback by default. Remote public deployment and authorization are outside the first release. Never block service startup with an unsolicited checkpoint download; initialize lazily when Chronos is requested.

CLI entry point is `hems-mcp`, also available as `python -m hems_mcp`. Define these commands:

- `demo --output-dir PATH --days 14 --seed 42 --interval-minutes 60` writes synthetic CSV plus forecast, summary, schedule, evaluation, and report artifacts.
- `forecast --input PATH --backend seasonal --horizon 24 --output PATH` writes forecast JSON.
- `summarize --input FORECAST_JSON --output PATH` writes summary JSON.
- `schedule --input FORECAST_JSON --energy-kwh 10 --max-charge-kw 7.2 --grid-limit-kw 10 --output PATH` writes a plan.
- `evaluate --input PATH --backend seasonal --horizon 24 --origins 3 --output-dir PATH` writes evaluation artifacts.
- `serve --transport stdio` launches the MCP service.

Allow optional output omission to write JSON to stdout. Exit nonzero on invalid input and requested backend failure. Missing optional LLM configuration may produce a successful partial evaluation with explicit skipped rows. CLI help and README must agree on the exact flags before integration completes.

## LLM HTTP adapter and experiment design

Provide a configurable OpenAI-compatible chat-completions HTTP adapter through httpx. Keep the URL, model, and API key configuration outside code. Suggested environment names are `HEMS_LLM_BASE_URL`, `HEMS_LLM_MODEL`, and `HEMS_LLM_API_KEY`. The README provides a placeholder JSON configuration with an environment-variable key reference; no .env loader is used. Local unauthenticated endpoints may omit a key when explicitly configured. Define whether the base URL includes `/v1`; normalize it once and avoid duplicated paths.

The request includes system instructions, a bounded JSON task payload, deterministic temperature when supported, and a bounded completion budget. Do not require a provider-specific JSON-schema option for basic interoperability. Request JSON in the prompt and validate the returned object strictly. Set 10-second connect and 60-second read timeouts, cap response bytes at 1 MiB, and check a 90-second elapsed budget between chunks. The elapsed check can overrun by one read timeout. Avoid automatic retries. Each arm uses one original request and at most one feasibility-feedback request per origin. Do not log the API key, full headers, or raw provider error bodies that may contain credentials.

For each forecast origin, run three comparable LLM decision arms when configured:

1. `llm_only`: receives the historical series, timestamps, horizon, and EV constraints. It must predict future load/solar arrays and propose an EV plan.
2. `numeric_curves`: receives the exact numeric forecast and identical EV constraints. It proposes an EV plan.
3. `semantic_summary`: receives only the computed summary and identical EV constraints. It proposes an EV plan.

Use the same provider, model, completion budget, and decision task across arms. The numeric and summary arms share one underlying forecast per origin. Record that full forecast error belongs to the backend for those two arms. If an assisted arm supplies different forecast arrays, score them separately and label their origin. Do not assign zero error when an arm omits arrays.

Minimum accepted LLM JSON includes `charge_kw` of length H. The baseline additionally requires `load_kw` and `solar_kw`, each length H. A `reason` string is optional. Validate exact lengths, finite nonnegative values, charger bounds, availability, battery delivery, and grid constraints. Use held-out actual load and solar to report realized cap violations separately from forecast-based feasibility. An LLM proposal must be scored as returned; repair may produce an additional result with a distinct label, but must not conceal the raw proposal's failure.

When configuration is missing, write status `skipped` with reason `missing_llm_configuration` and null metrics. When HTTP or parsing fails, write status `failed`, a sanitized failure category, elapsed time, and available usage. Never replace failed LLM outputs with a seasonal baseline and label them LLM results. Mock HTTP tests verify adapter plumbing only; their values must never appear as measured model benchmark results.

## Evaluation splits, metrics, and artifacts

Use expanding-window rolling origins. Select the final `origins * horizon_steps` rows as non-overlapping held-out windows when enough history exists. At origin k, call the forecast with rows strictly before that origin; compare against the next H rows only. Require at least a full daily history before the first origin. Bound origins at a small explicit maximum such as 20. All arms receive the same origin and no future observations. LLM-only receives at most 512 recent history slots; Chronos uses at most 2,048 and seasonal uses one daily cycle. Do not impute across the train/test boundary or tune parameters on held-out targets.

For each target report MAE in kW, RMSE in kW, signed energy bias in kWh, and optional absolute energy error in kWh. Define `MAE=mean(abs(pred-actual))`, `RMSE=sqrt(mean((pred-actual)^2))`, and energy bias as `sum(pred-actual)*dt_hours`. Avoid MAPE for solar because night values are zero. Aggregate by explicit sample counts; do not average incompatible horizon lengths without weights.

Measure wall latency with `perf_counter` and local process CPU time with `process_time`. Separate forecast construction, summary construction, scheduler execution, and LLM HTTP latency. Local CPU time does not describe remote model compute. Report provider compute as null unless the provider explicitly supplies it. Chronos time-series tokens are not comparable to LLM text tokens; do not combine them.

Read prompt and completion tokens from the provider's usage object. Leave absent values null. Serialized payload bytes or character count may supplement usage under clearly named fields, but never masquerade as token counts. A local no-LLM path legitimately uses zero remote LLM calls; skipped LLM requests have unknown or inapplicable token metrics.

Write `results.json` with configuration, software versions, seed, data provenance, origins, per-arm statuses, metrics, and limitations. Write a flat `metrics.csv` with one row per origin/arm/target or a documented equivalent. Write a readable `report.md` from the same data. Optional CSV forecast curves support plotting without a plotting dependency. Write outputs atomically where practical and avoid embedding credentials in configuration snapshots.

The report must distinguish these states: offline baseline evaluated; Chronos evaluated; Chronos unavailable; LLM skipped; LLM failed; LLM evaluated. Report improvements only when both compared quantities exist. Synthetic results demonstrate execution and consistency; they do not establish household forecast accuracy.

## Acceptance tests and completion gates

Core tests cover deterministic synthetic output, timezone validation, duplicate and missing intervals, nonfinite values, daily seasonal alignment, timestamp advancement, non-hourly kWh conversion, surplus window boundaries, exact EV energy delivery, availability limits, efficiency, zero-energy requests, infeasible demand, and existing household cap breaches. Use hand-calculated small arrays for numerical assertions.

A leakage test alters every held-out value and confirms unchanged predictions for the same origin. An adapter test with a fake pipeline verifies input and output dimensions without downloading weights. A real optional Chronos smoke test executes a small two-series forecast only when dependencies and weights are available; report whether it ran.

Service tests initialize a real MCP client over stdio or the SDK's in-memory transport, list tools, call forecast/summarize/schedule, and exercise a rejected input. Directly calling decorated functions alone does not establish protocol integration. Verify stdout contains no diagnostic logging. Test overload or cancellation behavior if the service claims those bounds.

Evaluation tests use an httpx mock transport for successful usage reporting, missing usage, malformed JSON, wrong horizons, HTTP failure, and missing configuration. Check that skipped metrics remain null and forecasts never see held-out rows. Check that report artifacts parse and preserve backend identity.

Run these from the project directory after aligning CLI flags with this contract:

```bash
uv sync --python 3.12 --group dev
uv run pytest -q
uv run hems-mcp --help
uv run hems-mcp demo --output-dir artifacts/demo --days 14
uv run hems-mcp forecast --input artifacts/demo/history.csv --backend seasonal --horizon 24 --output artifacts/forecast.json
uv run hems-mcp summarize --input artifacts/forecast.json --output artifacts/summary.json
uv run hems-mcp schedule --input artifacts/forecast.json --energy-kwh 10 --output artifacts/schedule.json
uv run hems-mcp evaluate --input artifacts/demo/history.csv --backend seasonal --horizon 24 --origins 3 --output-dir artifacts/evaluation
```

Optional real-model acceptance:

```bash
uv sync --python 3.12 --group dev --extra chronos
uv run hems-mcp forecast --input artifacts/demo/history.csv --backend chronos --horizon 24 --output artifacts/chronos.json
```

An authenticated LLM evaluation uses the same `evaluate` command with the documented environment populated. Installation, checkpoint downloads, and slow tests must run as tracked background jobs with progress logs. Do not claim optional execution when only mocks ran. Generated artifacts, caches, virtual environments, and secrets belong in `.gitignore`.

Phase 1 passes when core tests establish units, temporal alignment, and feasibility. Phase 2 passes when the installed CLI completes an offline workflow and a protocol client calls the MCP tools. Phase 3 passes when evaluation writes reproducible artifacts with honest optional statuses. Phase 4 passes when the integrator runs the full scoped suite, documents actual dependency versions, exercises the closed-loop example, and records whether real Chronos and LLM requests ran.

## Explicit limitations and next steps

The first version models one household and one EV, constant charger efficiency, continuous charging power, deterministic point forecasts, and fixed availability. It excludes vehicle minimum-current constraints, discrete phases, battery state-of-charge curves, tariffs, export caps, distribution voltage, and weather forecast uncertainty. The solar series uses historical behavior only. Schedule feasibility depends on predicted demand and cannot guarantee physical grid limits under forecast error.

A later release can add real meter adapters, Chronos-2 covariates, calibrated quantiles, robust scheduling, tariff-aware optimization, repeated observed-state replanning, and MCP v2 migration. Each extension must preserve explicit units, cutoff times, model provenance, and evidence-backed metrics.
