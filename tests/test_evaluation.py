"""Evaluation tests use synthetic data and mocked HTTP; they incur no API charges."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np

from hems_mcp.data import generate_demo
from hems_mcp.evaluation import metrics, run_evaluation
from hems_mcp.forecast import forecast_frame
from hems_mcp.llm import request_plan, resolve_config


class EvaluationTests(unittest.TestCase):
    def test_metrics_units(self):
        result = metrics([2, 4], [1, 2], 0.5)
        self.assertEqual(result["mae_kw"], 1.5)
        self.assertEqual(result["energy_bias_kwh"], 1.5)
        self.assertAlmostEqual(result["rmse_kw"], np.sqrt(2.5))

    def test_split_skip_and_artifacts(self):
        history = generate_demo(days=5)
        cutoffs = []

        def forecast(train, horizon, backend):
            cutoffs.append(train.timestamp.iloc[-1])
            return forecast_frame(train, horizon, backend)

        with tempfile.TemporaryDirectory() as directory:
            with patch("hems_mcp.evaluation.forecast_frame", side_effect=forecast):
                result = run_evaluation(history, directory, origins=2)
            self.assertEqual(cutoffs, [history.timestamp.iloc[71], history.timestamp.iloc[95]])
            self.assertEqual(result["remote_calls"], 0)
            arm = result["origins"][0]["arms"]["llm_only"]
            self.assertEqual(arm["status"], "skipped")
            self.assertIsNone(arm["forecast_metrics"])
            self.assertIsNone(arm["usage"]["total_tokens"])
            self.assertFalse(result["origins"][0]["constraint_feedback_loop"]["feedback"]["feasible"])
            self.assertTrue(result["origins"][0]["constraint_feedback_loop"]["outcome"]["feasible"])
            self.assertEqual(json.loads(Path(directory, "results.json").read_text())["backend"], "seasonal")
            for artifact in ("metrics.csv", "curves.csv", "report.md"):
                self.assertTrue(Path(directory, artifact).stat().st_size)
            changed = history.copy()
            changed.loc[72:, "load_kw"] += 50
            second = run_evaluation(changed, directory, origins=2)
            self.assertEqual(result["origins"][0]["forecast"]["load_kw"], second["origins"][0]["forecast"]["load_kw"])
            self.assertNotEqual(result["origins"][0]["forecast_metrics"], second["origins"][0]["forecast_metrics"])

    def test_feedback_and_representation(self):
        history = generate_demo(days=3)
        payloads = []

        def fake(config, payload):
            payloads.append(payload)
            proposal = {"charge_kw": [0.0] * 24}
            if payload["condition"] == "llm_only":
                proposal.update(load_kw=[1.0] * 24, solar_kw=[0.0] * 24)
            if "feasibility_feedback" in payload:
                proposal["charge_kw"][0] = 10 / 0.9 / 2
                proposal["charge_kw"][1] = 10 / 0.9 / 2
            return {"status": "evaluated", "proposal": proposal, "usage": {"total_tokens": None}}

        with tempfile.TemporaryDirectory() as directory:  # noqa: SIM117
            with patch("hems_mcp.evaluation.request_plan", side_effect=fake):
                result = run_evaluation(history, directory, origins=1,
                    llm_config={"base_url": "http://localhost:9000/v1", "model": "fake"})
        self.assertEqual(result["remote_calls"], 6)
        arm = result["origins"][0]["arms"]["llm_only"]
        self.assertFalse(arm["original"]["forecast_feasibility"]["feasible"])
        self.assertTrue(arm["feedback_attempt"]["forecast_feasibility"]["feasible"])
        semantic = next(p for p in payloads if p["condition"] == "semantic_summary")
        self.assertNotIn("forecast", semantic)
        self.assertNotIn("history", semantic)
        self.assertNotIn("load_kw", semantic["summary"])
        self.assertIn("forecast", next(p for p in payloads if p["condition"] == "numeric_curves"))

    def test_adapter_http_usage_and_failures(self):
        config = resolve_config({"base_url": "http://localhost/v1/", "model": "fake"})
        real_client = httpx.Client
        responses = [
            httpx.Response(200, json={"choices": [{"message": {"content": '{"charge_kw":[0]}'}}],
                                     "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}}),
            httpx.Response(200, json={"choices": [{"message": {"content": '{"charge_kw":[0]}'}}]}),
            httpx.Response(200, json={"choices": [{"message": {"content": 'not JSON'}}], "usage": {"total_tokens": 8}}),
            httpx.Response(401, text="secret provider diagnostic"),
        ]
        for index, response in enumerate(responses):
            def handler(request, response=response):
                self.assertEqual(str(request.url), "http://localhost/v1/chat/completions")
                self.assertEqual(json.loads(request.content)["model"], "fake")
                return response
            with patch("hems_mcp.llm.httpx.Client", side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)):
                result = request_plan(config, {"task": "test"})
            self.assertGreaterEqual(result["timing"]["wall_seconds"], 0)
            if index == 0:
                self.assertEqual(result["usage"]["total_tokens"], 19)
            if index == 1:
                self.assertIsNone(result["usage"]["total_tokens"])
            if index == 2:
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["usage"]["total_tokens"], 8)
            if index == 3:
                self.assertEqual(result["error_category"], "http_401")
                self.assertNotIn("secret", json.dumps(result))

    def test_wrong_horizon_is_failed(self):
        from hems_mcp.evaluation import check_plan
        with self.assertRaises(ValueError):
            check_plan([0], forecast_frame(generate_demo(2)), {"departure_step": 24})

    def test_huge_proposal_is_failed(self):
        with tempfile.TemporaryDirectory() as directory:  # noqa: SIM117
            with patch("hems_mcp.evaluation.request_plan", return_value={
                "status": "evaluated", "proposal": {"charge_kw": [1e308] * 24}, "usage": {}}):
                result = run_evaluation(generate_demo(3), directory, origins=1,
                    llm_config={"base_url": "http://localhost/v1", "model": "fake"})
        for arm in result["origins"][0]["arms"].values():
            self.assertEqual(arm["status"], "failed")
            self.assertEqual(arm["original"]["error_category"], "invalid_proposal")


if __name__ == "__main__":
    unittest.main()
