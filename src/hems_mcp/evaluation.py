"""Expanding-window evaluation with preserved proposals and measured local costs."""

import importlib.metadata
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import validate_history
from .forecast import forecast_frame
from .llm import EMPTY_USAGE, request_plan, resolve_config
from .schedule import explore_schedule
from .schemas import integer
from .summary import summarize_forecast

ARMS = ("llm_only", "numeric_curves", "semantic_summary")
CONSTRAINTS = {"energy_kwh": 10.0, "max_charge_kw": 7.2, "grid_limit_kw": 10.0,
               "efficiency": 0.9, "arrival_step": 0}


def measured(function, *args, **kwargs):
    start, cpu = time.perf_counter(), time.process_time()
    value = function(*args, **kwargs)
    return value, {"wall_seconds": time.perf_counter() - start,
                   "local_cpu_seconds": time.process_time() - cpu}


def metrics(prediction, actual, hours):
    error = np.asarray(prediction, dtype=float) - np.asarray(actual, dtype=float)
    scale = float(np.abs(error).max())
    return {"sample_count": len(error), "mae_kw": float(np.abs(error).mean()),
            "rmse_kw": float(scale * np.sqrt(np.mean((error / scale) ** 2))) if scale else 0.0,
            "energy_bias_kwh": float(error.sum() * hours)}


def _array(values, horizon, name):
    if not isinstance(values, list) or len(values) != horizon or any(isinstance(v, bool) or not isinstance(v, (float, int)) for v in values):
        raise ValueError(f"invalid {name}")
    try:
        result = np.asarray(values, dtype=float)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"invalid {name}") from exc
    if not np.isfinite(result).all() or (result < 0).any() or (result > 1e12).any():
        raise ValueError(f"invalid {name}")
    return result


def check_plan(charge, forecast, constraints):
    """Check an unmodified proposal against a forecast, including baseline violations."""
    charge = _array(charge, len(forecast["load_kw"]), "charge_kw")
    hours = forecast["interval_minutes"] / 60
    net = np.asarray(forecast["load_kw"]) - np.asarray(forecast["solar_kw"])
    arrival, departure = constraints["arrival_step"], constraints["departure_step"]
    delivered = float(charge.sum() * hours * constraints["efficiency"])
    violations = {
        "charger_slots": int((charge > constraints["max_charge_kw"] + 1e-8).sum()),
        "availability_slots": int((charge[:arrival] > 1e-8).sum() + (charge[departure:] > 1e-8).sum()),
        "grid_slots": int((np.maximum(net + charge, 0) > constraints["grid_limit_kw"] + 1e-8).sum()),
        "baseline_grid_slots": int((np.maximum(net, 0) > constraints["grid_limit_kw"] + 1e-8).sum()),
        "overdelivery": delivered > constraints["energy_kwh"] + 1e-8,
        "underdelivery": delivered < constraints["energy_kwh"] - 1e-8,
    }
    return {"feasible": not any(violations.values()), "violations": violations,
            "delivered_energy_kwh": delivered,
            "unmet_energy_kwh": max(0, constraints["energy_kwh"] - delivered),
            "peak_import_kw": float(np.maximum(net + charge, 0).max())}


def _proposal(call, arm, forecast, actual, constraints):
    if call["status"] != "evaluated":
        return
    try:
        proposal = call["proposal"]
        used = dict(forecast)
        if arm == "llm_only" or any(name in proposal for name in ("load_kw", "solar_kw")):
            for name in ("load_kw", "solar_kw"):
                used[name] = _array(proposal.get(name), len(forecast[name]), name).tolist()
            call["forecast_metrics"] = {name: metrics(used[name], actual[name], used["interval_minutes"] / 60) for name in ("load_kw", "solar_kw")}
            call["forecast_source"] = "llm_prediction"
        else:
            call["forecast_metrics"] = None
            call["forecast_source"] = "shared_backend"
        call["forecast_feasibility"] = check_plan(proposal.get("charge_kw"), used, constraints)
        call["realized_feasibility"] = check_plan(proposal["charge_kw"], actual, constraints)
    except (ValueError, TypeError, KeyError):
        call.update(status="failed", error_category="invalid_proposal")


def _atomic(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def run_evaluation(history, output_dir, backend="seasonal", horizon_steps=24, origins=3, llm_config=None):
    history = validate_history(history)
    horizon = integer(horizon_steps, "horizon_steps", 1, 48)
    origins = integer(origins, "origins", 1, 20)
    config = resolve_config(llm_config)
    minutes = int((history.timestamp.iloc[1] - history.timestamp.iloc[0]).total_seconds() / 60)
    if 1440 % minutes:
        raise ValueError("evaluation requires an interval that evenly divides one day")
    first = len(history) - origins * horizon
    if first < 1440 // minutes:
        raise ValueError("evaluation requires a full daily training history before held-out windows")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "backend": backend, "data_provenance": "caller_supplied; demo inputs are synthetic",
              "seed": None, "configuration": {"horizon_steps": horizon, "origins": origins,
              "llm": {"model": config["model"], "max_completion_tokens": config["max_completion_tokens"]} if config else None},
              "software": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "httpx")},
              "python": platform.python_version(), "origins": [], "remote_calls": 0,
              "limitations": ["Synthetic demo results do not establish accuracy on real households.",
                  "Local CPU seconds exclude remote provider compute; provider_compute is null.",
                  "RSS is the whole process lifetime high-water mark, not incremental model memory.",
                  "Forecast-feasible schedules may violate actual grid limits; realized checks use held-out observations.",
                  "Continuous charging, constant efficiency, one EV, no tariffs or hardware actuation.",
                  "Semantic summaries omit slot detail. Timing measurements are not reproducible constants."]}
    flat, curves = [], []
    for origin, cutoff in enumerate(range(first, len(history), horizon)):
        train = history.iloc[:cutoff].copy()
        actual_rows = history.iloc[cutoff:cutoff + horizon]
        forecast, forecast_timing = measured(forecast_frame, train, horizon, backend)
        actual = {**forecast, **{name: actual_rows[name].tolist() for name in ("load_kw", "solar_kw")}}
        summary, summary_timing = measured(summarize_forecast, forecast)
        constraints = {**CONSTRAINTS, "departure_step": horizon}
        schedule, schedule_timing = measured(explore_schedule, forecast, **constraints)
        target_metrics = {name: metrics(forecast[name], actual[name], minutes / 60) for name in ("load_kw", "solar_kw")}
        row = {"origin": origin, "train_steps": cutoff, "train_end": train.timestamp.iloc[-1].isoformat(),
               "test_start": actual_rows.timestamp.iloc[0].isoformat(), "forecast": forecast,
               "summary": summary, "forecast_metrics": target_metrics,
               "timing": {"forecast": forecast_timing, "summary": summary_timing, "schedule": schedule_timing},
               "backend_schedule": {"plan": schedule,
                   "forecast_feasibility": check_plan(schedule["charging_kw"], forecast, constraints),
                   "realized_feasibility": check_plan(schedule["charging_kw"], actual, constraints)}, "arms": {}}
        # A deterministic constraint-feedback loop remains observable without an LLM.
        short_constraints = {**constraints, "energy_kwh": 10.0, "departure_step": 1,
                             "max_charge_kw": 1.0}
        short = explore_schedule(forecast, **short_constraints)
        extended_constraints = {**short_constraints, "departure_step": horizon}
        extended = explore_schedule(forecast, **extended_constraints)
        row["constraint_feedback_loop"] = {"original_constraints": short_constraints, "original_plan": short,
            "feedback": check_plan(short["charging_kw"], forecast, short_constraints),
            "revised_constraints": extended_constraints, "recomputed_plan": extended,
            "outcome": check_plan(extended["charging_kw"], forecast, extended_constraints)}
        for name, values in target_metrics.items():
            flat.append({"origin": origin, "arm": backend, "target": name, "status": "evaluated", **values})
        for step, stamp in enumerate(forecast["timestamps"]):
            curves.append({"origin": origin, "arm": backend, "timestamp": stamp,
                           **{f"{kind}_{name}": obj[name][step] for kind, obj in (("predicted", forecast), ("actual", actual)) for name in ("load_kw", "solar_kw")}})
        for arm in ARMS:
            if config is None:
                row["arms"][arm] = {"status": "skipped", "reason": "missing_llm_configuration",
                                     "forecast_metrics": None, "usage": EMPTY_USAGE.copy()}
                flat.append({"origin": origin, "arm": arm, "target": None, "status": "skipped"})
                continue
            payload = {"condition": arm, "horizon_steps": horizon, "interval_minutes": minutes,
                       "timestamps": forecast["timestamps"], "constraints": constraints,
                       "required_output": ["charge_kw", "load_kw", "solar_kw"] if arm == "llm_only" else ["charge_kw"]}
            if arm == "llm_only":
                payload["history"] = [{**record, "timestamp": record["timestamp"].isoformat()} for record in train.iloc[-512:].to_dict("records")]
            elif arm == "numeric_curves":
                payload["forecast"] = forecast
            else:
                payload["summary"] = summary
            call = request_plan(config, payload)
            result["remote_calls"] += 1
            _proposal(call, arm, forecast, actual, constraints)
            arm_result = {"status": call["status"], "original": call, "forecast_metrics": call.get("forecast_metrics")}
            if call["status"] == "evaluated":
                final = call
                if not call["forecast_feasibility"]["feasible"]:
                    feedback_payload = {**payload, "previous_proposal": call["proposal"],
                                        "feasibility_feedback": call["forecast_feasibility"],
                                        "instruction": "Correct the proposal using this feasibility feedback. Constraints are unchanged."}
                    correction = request_plan(config, feedback_payload)
                    result["remote_calls"] += 1
                    _proposal(correction, arm, forecast, actual, constraints)
                    arm_result["feedback_attempt"] = correction
                    final = correction
                arm_result["final_status"] = final["status"]
                arm_result["final_feasibility"] = final.get("forecast_feasibility")
                if final["status"] != "evaluated" or not final["forecast_feasibility"]["feasible"]:
                    arm_result["deterministic_fallback"] = row["backend_schedule"]
            row["arms"][arm] = arm_result
            for name in ("load_kw", "solar_kw"):
                values = (call.get("forecast_metrics") or {}).get(name)
                flat.append({"origin": origin, "arm": arm, "target": name, "status": call["status"], **(values or {})})
                if values:
                    for step, stamp in enumerate(forecast["timestamps"]):
                        curves.append({"origin": origin, "arm": arm, "timestamp": stamp,
                                       f"predicted_{name}": call["proposal"][name][step],
                                       f"actual_{name}": actual[name][step]})
        result["origins"].append(row)
    result["aggregate_backend_metrics"] = {name: metrics(
        np.concatenate([row["forecast"][name] for row in result["origins"]]),
        history.iloc[first:][name].to_numpy(), minutes / 60) for name in ("load_kw", "solar_kw")}
    result["compute"] = {"process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                         "provider_compute": None}
    _atomic(output / "results.json", json.dumps(result, indent=2, allow_nan=False) + "\n")
    _atomic(output / "metrics.csv", pd.DataFrame(flat).to_csv(index=False))
    _atomic(output / "curves.csv", pd.DataFrame(curves).to_csv(index=False))
    lines = ["# Rolling-origin evaluation", "", f"Backend: {backend}. Origins: {origins}. Horizon: {horizon} slots.",
             f"Observed HTTP calls: {result['remote_calls']}. LLM status: {'configured' if config else 'skipped: missing_llm_configuration'}.", "",
             "Power errors are kW; signed energy bias is kWh. Training ends before each held-out window.", "",
             "| Target | Samples | MAE kW | RMSE kW | Energy bias kWh |", "|---|---:|---:|---:|---:|"]
    for name, values in result["aggregate_backend_metrics"].items():
        lines.append(f"| {name} | {values['sample_count']} | {values['mae_kw']:.6f} | {values['rmse_kw']:.6f} | {values['energy_bias_kwh']:.6f} |")
    lines.extend(["", "| Origin | LLM condition | Original status | Original feasible | Final feasible |", "|---:|---|---|---|---|"])
    for row in result["origins"]:
        for arm, outcome in row["arms"].items():
            original = outcome.get("original", {})
            initial_feasible = original.get("forecast_feasibility", {}).get("feasible")
            final_feasible = (outcome.get("final_feasibility") or {}).get("feasible")
            lines.append(f"| {row['origin']} | {arm} | {outcome['status']} | {initial_feasible} | {final_feasible} |")
    lines.extend(["", "Original proposals, one bounded LLM feedback attempt, and separately labeled deterministic fallbacks appear in results.json.", ""])
    lines.extend(f"- {limitation}" for limitation in result["limitations"])
    _atomic(output / "report.md", "\n".join(lines) + "\n")
    return result
