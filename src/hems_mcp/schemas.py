"""Shared validation at the tool boundary."""

import math
from numbers import Integral, Real

import numpy as np
import pandas as pd

MAX_HISTORY_STEPS = 35040
MAX_HORIZON_STEPS = 168


def integer(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return int(value)


def number(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    return float(value)


def utc_timestamps(values):
    try:
        parsed = [pd.Timestamp(value) for value in values]
        if any(pd.isna(value) or value.tzinfo is None for value in parsed):
            raise ValueError("timestamps must include a timezone")
        return pd.DatetimeIndex(pd.to_datetime(parsed, utc=True)).as_unit("ns")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("timestamps must be valid timezone-aware datetimes") from exc


def powers(values, name):
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain numeric powers")
    result = raw.astype(float)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError(f"{name} must contain finite nonnegative powers")
    if (result > np.finfo(float).max / (MAX_HISTORY_STEPS * 24)).any():
        raise ValueError(f"{name} values are too large for finite energy arithmetic")
    return result


def forecast_arrays(forecast):
    """Validate a forecast payload and return timestamps, load, solar, slot hours."""
    if not isinstance(forecast, dict):
        raise ValueError("forecast must be an object")  # noqa: TRY004 - tool validation contract
    try:
        interval = integer(forecast["interval_minutes"], "interval_minutes", 1, 1440)
        timestamps = utc_timestamps(forecast["timestamps"])
        load = powers(forecast["load_kw"], "load_kw")
        solar = powers(forecast["solar_kw"], "solar_kw")
    except KeyError as exc:
        raise ValueError(f"forecast is missing {exc.args[0]}") from exc
    if not 1 <= len(load) <= MAX_HORIZON_STEPS or len(load) != len(solar) or len(load) != len(timestamps):
        raise ValueError("forecast arrays must have matching lengths between 1 and 168")
    if len(timestamps) > 1 and not (np.diff(timestamps.asi8) == interval * 60_000_000_000).all():
        raise ValueError("forecast timestamps must be regular and match interval_minutes")
    return timestamps, load, solar, interval / 60
