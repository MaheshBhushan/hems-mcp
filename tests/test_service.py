"""Real CLI boundaries and the official MCP client's stdio lifecycle."""

import asyncio
import json
import subprocess
import sys
import threading
from datetime import timedelta

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from hems_mcp.data import generate_demo


def cli(*args, stdin=None):
    return subprocess.run(
        [sys.executable, "-m", "hems_mcp", *args], input=stdin,
        capture_output=True, text=True, timeout=40, check=False,
    )


def test_cli_json_csv_and_demo(tmp_path):
    for command in (None, "demo", "forecast", "summarize", "schedule", "evaluate", "serve"):
        result = cli(*([command] if command else []), "--help")
        assert result.returncode == 0, result.stderr
    history = generate_demo()
    records = history.to_json(orient="records", date_format="iso")
    forecast_result = cli("forecast", "-", stdin=records)
    assert forecast_result.returncode == 0, forecast_result.stderr
    forecast = json.loads(forecast_result.stdout)
    assert len(forecast["load_kw"]) == 24
    summary = cli("summarize", "-", stdin=forecast_result.stdout)
    assert summary.returncode == 0, summary.stderr
    assert json.loads(summary.stdout)["load_energy_kwh"] > 0
    schedule = cli("schedule", "-", stdin=forecast_result.stdout)
    assert schedule.returncode == 0, schedule.stderr
    assert json.loads(schedule.stdout)["status"] == "feasible"
    history.to_csv(tmp_path / "history.csv", index=False)
    csv_result = cli("forecast", str(tmp_path / "history.csv"))
    assert csv_result.returncode == 0, csv_result.stderr
    assert json.loads(csv_result.stdout)["load_kw"] == pytest.approx(forecast["load_kw"], abs=1e-9)
    demo = cli("demo", "--output-dir", str(tmp_path / "demo"))
    assert demo.returncode == 0, demo.stderr
    report = json.loads(demo.stdout)
    assert report["forecast"]["backend"] == "seasonal"
    assert report["schedule"]["unmet_energy_kwh"] < 1e-6
    assert report["evaluation"]
    for name in ("history.csv", "forecast.json", "summary.json", "schedule.json", "report.json"):
        assert (tmp_path / "demo" / name).is_file()


def test_cli_rejects_malformed_input():
    for payload in ("not json", "{}", '[{"timestamp":"bad"}]'):
        result = cli("forecast", "-", stdin=payload)
        assert result.returncode == 2
        assert result.stdout == ""
        assert result.stderr.startswith("hems-mcp:")


def test_real_stdio_client():
    async def run():
        parameters = StdioServerParameters(
            command=sys.executable, args=["-m", "hems_mcp", "serve"],
        )
        async with stdio_client(parameters) as (read, write):  # noqa: SIM117 - streams create session
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30)) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert {tool.name for tool in tools.tools} == {
                    "generate_demo_history", "forecast_energy", "summarize_energy", "explore_ev_schedule",
                }
                generated = await session.call_tool("generate_demo_history", {"days": 3})
                assert not generated.isError
                records = generated.structuredContent["records"]
                result = await session.call_tool("forecast_energy", {"history": records})
                assert not result.isError
                forecast = result.structuredContent
                assert len(forecast["timestamps"]) == 24
                summary = await session.call_tool("summarize_energy", {"forecast": forecast})
                assert not summary.isError
                assert summary.structuredContent["load_energy_kwh"] > 0
                schedule = await session.call_tool("explore_ev_schedule", {"forecast": forecast})
                assert not schedule.isError
                assert schedule.structuredContent["status"] == "feasible"
                invalid = await session.call_tool("forecast_energy", {"history": []})
                assert invalid.isError
                invalid_bool = await session.call_tool(
                    "forecast_energy", {"history": records, "horizon_steps": True},
                )
                assert invalid_bool.isError
    asyncio.run(run())


def test_admission_and_cancelled_client_keeps_cpu_slot(monkeypatch):
    from hems_mcp import server

    async def run():
        monkeypatch.setattr(server, "_work_limit", asyncio.Semaphore(1))
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        ran = []

        def slow():
            loop.call_soon_threadsafe(started.set)
            if not release.wait(timeout=5):
                raise RuntimeError("test release did not arrive")
            return "slow"

        first = asyncio.create_task(server._run(slow))
        await asyncio.wait_for(started.wait(), timeout=5)
        pending = [asyncio.create_task(server._run(lambda: ran.append(True))) for _ in range(3)]
        await asyncio.sleep(0)  # Admit pending jobs without waiting for their CPU slots.
        try:
            try:
                await server._run(lambda: None)
            except RuntimeError as exc:
                assert "busy" in str(exc)
            else:
                raise AssertionError("fifth request must be rejected")
            first.cancel()
            try:
                await first
            except asyncio.CancelledError:
                pass
            assert not ran
            assert server._admitted == 4
        finally:
            release.set()
        await asyncio.gather(*pending)
        assert len(ran) == 3
        assert server._admitted == 0
    asyncio.run(run())
