"""Reproducible synthetic data and strict CSV history validation."""

import numpy as np
import pandas as pd

from .schemas import MAX_HISTORY_STEPS, integer, powers, utc_timestamps


def validate_history(df):
    if not isinstance(df, pd.DataFrame) or not {"timestamp", "load_kw", "solar_kw"}.issubset(df.columns):
        raise ValueError("history must contain timestamp, load_kw, and solar_kw columns")
    if not 2 <= len(df) <= MAX_HISTORY_STEPS:
        raise ValueError(f"history must contain 2 to {MAX_HISTORY_STEPS} rows")
    timestamps = utc_timestamps(df["timestamp"])
    deltas = np.diff(timestamps.asi8)
    minute = 60_000_000_000
    if deltas[0] < minute or deltas[0] > 1440 * minute or deltas[0] % minute or not (deltas == deltas[0]).all():
        raise ValueError("history timestamps must increase at a regular whole-minute interval from 1 to 1440 minutes")
    return pd.DataFrame({"timestamp": timestamps, "load_kw": powers(df["load_kw"], "load_kw"), "solar_kw": powers(df["solar_kw"], "solar_kw")})


def generate_demo(days=14, seed=42, interval_minutes=60):
    days = integer(days, "days", 1, 365)
    seed = integer(seed, "seed", 0, 2**32 - 1)
    interval_minutes = integer(interval_minutes, "interval_minutes", 1, 1440)
    steps = days * 1440 // interval_minutes
    if not 2 <= steps <= MAX_HISTORY_STEPS:
        raise ValueError(f"demo must contain 2 to {MAX_HISTORY_STEPS} intervals")
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-06-01", periods=steps, freq=f"{interval_minutes}min", tz="UTC")
    hour = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60
    morning = np.exp(-0.5 * ((hour - 7) / 1.5) ** 2)
    evening = np.exp(-0.5 * ((hour - 19) / 2) ** 2)
    load = np.maximum(0.05, 0.55 + 1.2 * morning + 1.8 * evening + rng.normal(0, 0.12, steps))
    daylight = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
    daylight[(hour < 6) | (hour > 18)] = 0
    daily_cloud = rng.uniform(0.55, 1, days + 1)[np.arange(steps) * interval_minutes // 1440]
    solar = 4.8 * daylight * daily_cloud
    return validate_history(pd.DataFrame({"timestamp": timestamps, "load_kw": load, "solar_kw": solar}))
