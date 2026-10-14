"""Command-line JSON/CSV boundaries for the local energy workflow."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .data import generate_demo, validate_history
from .forecast import forecast_frame
from .schedule import explore_schedule
from .summary import summarize_forecast


def _read_json(source: str):
    if source == "-":
        return json.load(sys.stdin)
    with Path(source).open() as stream:
        return json.load(stream)


def _read_history(source: str):
    if source != "-" and Path(source).suffix.lower() == ".csv":
        return validate_history(pd.read_csv(source))
    records = _read_json(source)
    if not isinstance(records, list):
        raise TypeError("History JSON must be an array of timestamp/load_kw/solar_kw records")
    return validate_history(pd.DataFrame(records))


def _write_json(value, destination: str | Path | None = None):
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if destination is None or str(destination) == "-":
        sys.stdout.write(encoded)
    else:
        Path(destination).write_text(encoded)


def _forecast_options(parser, *, include_model=True):
    parser.add_argument("--backend", choices=("seasonal", "chronos"), default="seasonal")
    if include_model:
        parser.add_argument("--model-id", default="amazon/chronos-bolt-tiny")
    parser.add_argument("--horizon-steps", "--horizon", type=int, default=24)


def _schedule_options(parser):
    parser.add_argument("--energy-kwh", type=float, default=10)
    parser.add_argument("--max-charge-kw", type=float, default=7.2)
    parser.add_argument("--grid-limit-kw", type=float, default=10)
    parser.add_argument("--efficiency", type=float, default=0.9)
    parser.add_argument("--arrival-step", type=int, default=0)
    parser.add_argument("--departure-step", type=int)


def _schedule(forecast, args):
    return explore_schedule(
        forecast, energy_kwh=args.energy_kwh, max_charge_kw=args.max_charge_kw,
        grid_limit_kw=args.grid_limit_kw, efficiency=args.efficiency,
        arrival_step=args.arrival_step, departure_step=args.departure_step,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Local household energy forecasts and EV charging scenarios.",
        epilog="History: CSV file or JSON record array; '-' reads JSON from stdin. "
        "Forecast inputs and all reports use JSON. Power is kW; energy is kWh.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run an offline seasonal workflow and save artifacts")
    demo.add_argument("--output-dir", "--output", default="artifacts/demo")
    demo.add_argument("--days", type=int, default=14)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--interval-minutes", type=int, default=60)
    demo.add_argument("--horizon-steps", type=int, default=24)
    _schedule_options(demo)
    forecast = commands.add_parser("forecast", help="Forecast history from CSV or JSON")
    forecast.add_argument("input", nargs="?", help="History CSV/JSON path, or '-' for JSON stdin")
    _forecast_options(forecast)
    summary = commands.add_parser("summarize", help="Summarize forecast JSON")
    summary.add_argument("input", nargs="?", help="Forecast JSON path or '-'")
    schedule = commands.add_parser("schedule", help="Explore solar-first EV charging")
    schedule.add_argument("input", nargs="?", help="Forecast JSON path or '-'")
    _schedule_options(schedule)
    evaluate = commands.add_parser("evaluate", help="Run rolling-origin evaluation")
    evaluate.add_argument("input", nargs="?", help="History CSV/JSON path, or '-' for JSON stdin")
    evaluate.add_argument("--output-dir", default="artifacts/evaluation")
    evaluate.add_argument("--origins", type=int, default=3)
    evaluate.add_argument("--llm-config", help="Optional JSON config path; keep keys in environment")
    _forecast_options(evaluate, include_model=False)
    serve = commands.add_parser("serve", help="Run the MCP stdio service")
    serve.add_argument("--transport", choices=("stdio",), default="stdio")
    for command in (forecast, summary, schedule, evaluate):
        command.add_argument("--input", dest="input_flag", help="Input path (alternative to positional)")
        command.add_argument("--output", default="-", help="JSON destination (default stdout)")
    args = parser.parse_args(argv)
    if args.command not in ("demo", "serve"):
        if args.input and args.input_flag:
            parser.error("provide an input path either positionally or with --input")
        args.input = args.input or args.input_flag
        if args.input is None:
            parser.error("an input path or '-' is required")
    try:
        if args.command == "serve":
            from .server import serve
            serve()
            return
        if args.command == "demo":
            from .evaluation import run_evaluation
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            history = generate_demo(args.days, args.seed, args.interval_minutes)
            forecast = forecast_frame(history, args.horizon_steps, backend="seasonal")
            result = {
                "forecast": forecast, "summary": summarize_forecast(forecast),
                "schedule": _schedule(forecast, args),
                "evaluation": run_evaluation(
                    history, output_dir / "evaluation", horizon_steps=args.horizon_steps,
                ),
            }
            history.to_csv(output_dir / "history.csv", index=False)
            for name, value in result.items():
                _write_json(value, output_dir / f"{name}.json")
            result["artifacts_dir"] = str(output_dir.resolve())
            _write_json(result, output_dir / "report.json")
            _write_json(result)
            return
        if args.command == "forecast":
            result = forecast_frame(
                _read_history(args.input), args.horizon_steps, args.backend, args.model_id,
            )
        elif args.command == "summarize":
            result = summarize_forecast(_read_json(args.input))
        elif args.command == "schedule":
            result = _schedule(_read_json(args.input), args)
        else:
            from .evaluation import run_evaluation
            result = run_evaluation(
                _read_history(args.input), args.output_dir, backend=args.backend,
                horizon_steps=args.horizon_steps, origins=args.origins,
                llm_config=_read_json(args.llm_config) if args.llm_config else None,
            )
        _write_json(result, args.output)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError) as exc:
        print(f"hems-mcp: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
