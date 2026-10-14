"""MCP stdio tools. All power and energy computations stay in local Python."""

import asyncio
import functools
import json
from typing import Any

import pandas as pd
from mcp.server.fastmcp import FastMCP
from pydantic import StrictFloat, StrictInt

from .data import generate_demo, validate_history
from .forecast import forecast_frame
from .schedule import explore_schedule
from .summary import summarize_forecast

mcp = FastMCP("HEMS local energy tools")
# ponytail: one CPU job at a time; use worker processes if throughput requires it.
_work_limit = asyncio.Semaphore(1)
_admitted = 0
_jobs: set[asyncio.Task] = set()


async def _run(function, *args, **kwargs):
    global _admitted
    if _admitted >= 4:
        raise RuntimeError("Service busy: four jobs are active or waiting; retry later")
    _admitted += 1

    async def work():
        global _admitted
        try:
            async with _work_limit:
                return await asyncio.to_thread(functools.partial(function, *args, **kwargs))
        finally:
            _admitted -= 1

    def finished(task):
        _jobs.discard(task)
        if not task.cancelled():
            task.exception()  # Consume failures after the requesting client has disconnected.

    task = asyncio.create_task(work())
    _jobs.add(task)
    task.add_done_callback(finished)
    # Shield retains the job and its CPU slot after timeout or client cancellation.
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=120)
    except TimeoutError as exc:
        raise RuntimeError("Job exceeded 120 seconds; it continues locally; retry later") from exc


@mcp.tool(structured_output=True)
async def generate_demo_history(
    days: StrictInt = 14, seed: StrictInt = 42, interval_minutes: StrictInt = 60,
) -> dict[str, Any]:
    """Generate synthetic local history as timestamp/load_kw/solar_kw records (not meter data)."""
    def run():
        if not 1 <= days <= 366:
            raise ValueError("days must be between 1 and 366")
        frame = generate_demo(days, seed, interval_minutes)
        return {"records": json.loads(frame.to_json(orient="records", date_format="iso"))}
    return await _run(run)


@mcp.tool(structured_output=True)
async def forecast_energy(
    history: list[dict], horizon_steps: StrictInt = 24, backend: str = "seasonal",
    model_id: str = "amazon/chronos-bolt-tiny",
) -> dict[str, Any]:
    """Forecast household load and solar kW from regular UTC history. Chronos requires its extra."""
    def run():
        if len(history) > 100_000:
            raise ValueError("history exceeds the 100000-record service limit")
        return forecast_frame(validate_history(pd.DataFrame(history)), horizon_steps, backend, model_id)
    return await _run(run)


@mcp.tool(structured_output=True)
async def summarize_energy(forecast: dict) -> dict[str, Any]:
    """Calculate deterministic kWh totals and peak kW from forecast JSON."""
    return await _run(summarize_forecast, forecast)


@mcp.tool(structured_output=True)
async def explore_ev_schedule(
    forecast: dict, energy_kwh: StrictFloat = 10, max_charge_kw: StrictFloat = 7.2,
    grid_limit_kw: StrictFloat = 10, efficiency: StrictFloat = 0.9,
    arrival_step: StrictInt = 0, departure_step: StrictInt | None = None,
) -> dict[str, Any]:
    """Explore solar-first EV charging. Arrival is inclusive; departure is exclusive. No actuation."""
    return await _run(
        explore_schedule, forecast, energy_kwh=energy_kwh, max_charge_kw=max_charge_kw,
        grid_limit_kw=grid_limit_kw, efficiency=efficiency,
        arrival_step=arrival_step, departure_step=departure_step,
    )


def serve():
    mcp.run(transport="stdio")
