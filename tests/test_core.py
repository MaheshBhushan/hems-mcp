import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from hems_mcp.data import generate_demo, validate_history
from hems_mcp.forecast import forecast_frame
from hems_mcp.schedule import explore_schedule
from hems_mcp.summary import summarize_forecast


def fixture(load, solar, interval=60):
    return {"timestamps": [stamp.isoformat() for stamp in pd.date_range("2025-01-01", periods=len(load), freq=f"{interval}min", tz="UTC")], "interval_minutes": interval, "load_kw": load, "solar_kw": solar}


class CoreTests(unittest.TestCase):
    def test_demo_and_seasonal_repeat(self):
        history = generate_demo()
        pd.testing.assert_frame_equal(history, generate_demo())
        forecast = forecast_frame(history)
        self.assertEqual(forecast["backend"], "seasonal")
        self.assertIsNone(forecast["model_id"])
        np.testing.assert_array_equal(forecast["load_kw"], history.load_kw.iloc[-24:])
        self.assertEqual(pd.Timestamp(forecast["timestamps"][0]), history.timestamp.iloc[-1] + pd.Timedelta(hours=1))

    def test_summary_power_and_energy(self):
        result = summarize_forecast(fixture([4, 1], [1, 3], 30))
        self.assertEqual(result["load_energy_kwh"], 2.5)
        self.assertEqual(result["import_energy_kwh"], 1.5)
        self.assertEqual(result["export_energy_kwh"], 1)
        self.assertEqual(result["peak_import_kw"], 3)
        self.assertEqual(result["load_energy_kwh"] - result["solar_energy_kwh"], result["import_energy_kwh"] - result["export_energy_kwh"])

    def test_fractional_solar_first_efficiency(self):
        result = explore_schedule(fixture([1, 1, 1], [0, 5, 0]), energy_kwh=1, efficiency=0.8, max_charge_kw=2)
        self.assertEqual(result["status"], "feasible")
        np.testing.assert_allclose(result["charging_kw"], [0, 1.25, 0])
        self.assertAlmostEqual(result["delivered_energy_kwh"], 1)
        self.assertEqual(result["baseline_import_energy_kwh"], result["scheduled_import_energy_kwh"])

    def test_capacity_and_availability(self):
        forecast = fixture([8, 8, 8], [0, 0, 0], 30)
        result = explore_schedule(forecast, energy_kwh=5, efficiency=0.9, grid_limit_kw=10, arrival_step=1, departure_step=2)
        self.assertEqual(result["status"], "infeasible")
        np.testing.assert_allclose(result["charging_kw"], [0, 2, 0])
        self.assertAlmostEqual(result["unmet_energy_kwh"], 4.1)
        self.assertLessEqual(result["scheduled_peak_import_kw"], 10)

    def test_existing_grid_violation(self):
        result = explore_schedule(fixture([12, 1], [0, 0]), energy_kwh=0, grid_limit_kw=10)
        self.assertEqual(result["status"], "infeasible")
        self.assertEqual(result["unmet_energy_kwh"], 0)
        self.assertEqual(result["charging_kw"], [0, 0])

    def test_grid_allocation_prefers_low_demand_then_earliest(self):
        result = explore_schedule(fixture([3, 1, 1], [0, 0, 0]), energy_kwh=1, efficiency=1)
        self.assertEqual(result["charging_kw"], [0, 1, 0])
        self.assertEqual(result["status"], "feasible")

    def test_surplus_windows_and_earliest_peak_tie(self):
        result = summarize_forecast(fixture([2, 2, 1, 1, 0], [0, 4, 3, 0, 2], 30))
        self.assertEqual(result["peak_load_timestamp"], "2025-01-01T00:00:00+00:00")
        self.assertEqual(len(result["surplus_windows"]), 2)
        self.assertEqual(result["surplus_windows"][0], {
            "start": "2025-01-01T00:30:00+00:00", "end": "2025-01-01T01:30:00+00:00",
            "start_step": 1, "end_step": 3, "surplus_kwh": 2.0})
        self.assertEqual(sum(window["surplus_kwh"] for window in result["surplus_windows"]), result["surplus_energy_kwh"])
        self.assertIn("end exclusive", result["text"])

    def test_chronos_negative_predictions_are_clipped(self):
        # Simulate the median output without downloading optional model weights.
        with patch("hems_mcp.forecast._chronos", return_value=(np.array([[-1, 2], [-0.5, 0]]), 0, 0)):
            result = forecast_frame(generate_demo(days=1), horizon_steps=2, backend="chronos")
        self.assertEqual(result["backend"], "chronos")
        self.assertEqual(result["load_kw"], [0, 2])
        self.assertEqual(result["solar_kw"], [0, 0])
        self.assertTrue(any("clipped" in warning for warning in result["warnings"]))

    def test_schedule_invalid_arguments(self):
        forecast = fixture([1, 1], [0, 0])
        for kwargs in ({"energy_kwh": -1}, {"energy_kwh": np.nan}, {"energy_kwh": np.inf},
                       {"energy_kwh": True}, {"energy_kwh": "1"}, {"efficiency": 1.1},
                       {"max_charge_kw": -1}, {"grid_limit_kw": -1}, {"grid_limit_kw": 0}, {"arrival_step": -1},
                       {"max_charge_kw": np.finfo(float).max},
                       {"arrival_step": 3}, {"arrival_step": 0.5}, {"departure_step": 3},
                       {"arrival_step": 2}, {"arrival_step": 1, "departure_step": 1},
                       {"arrival_step": 2, "departure_step": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                explore_schedule(forecast, **kwargs)

    def test_invalid_boundary_inputs(self):
        history = generate_demo(days=1)
        for value in (-1, np.nan, np.inf, np.finfo(float).max, "1", True):
            bad = history.copy()
            bad["load_kw"] = value
            with self.assertRaises(ValueError):
                validate_history(bad)
        bad = history.copy()
        bad.loc[1, "timestamp"] = bad.loc[0, "timestamp"]
        with self.assertRaises(ValueError):
            validate_history(bad)
        bad = history.copy()
        bad["timestamp"] = bad.timestamp.dt.tz_localize(None)
        with self.assertRaises(ValueError):
            validate_history(bad)
        for horizon in (0, 169, True):
            with self.assertRaises(ValueError):
                forecast_frame(history, horizon_steps=horizon)
        with self.assertRaises(ValueError):
            explore_schedule(fixture([1], [0]), efficiency=0)
        with self.assertRaises(ValueError):
            forecast_frame(history, backend="fake")


if __name__ == "__main__":
    unittest.main()
