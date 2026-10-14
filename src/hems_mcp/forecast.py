"""Offline seasonal baseline and optional real Chronos-Bolt inference."""

import threading
import time

import numpy as np
import pandas as pd

from .data import validate_history
from .schemas import MAX_HORIZON_STEPS, integer

# ponytail: one CPU model and one lock; use process workers if throughput matters.
_LOCK = threading.Lock()
_PIPELINE = None
MODEL_ID = "amazon/chronos-bolt-tiny"


def _chronos(values, horizon):
    global _PIPELINE
    try:
        import torch
        from chronos import ChronosBoltPipeline
    except ImportError as exc:
        raise RuntimeError("Chronos backend requires the optional chronos dependencies; install with uv sync --extra chronos") from exc
    with _LOCK:
        load_start = time.perf_counter()
        if _PIPELINE is None:
            torch.set_num_threads(min(4, torch.get_num_threads()))
            _PIPELINE = ChronosBoltPipeline.from_pretrained(MODEL_ID, device_map="cpu", torch_dtype=torch.float32)
        load_seconds = time.perf_counter() - load_start
        inference_start = time.perf_counter()
        with torch.inference_mode():
            prediction = _PIPELINE.predict(torch.tensor(values[:, -2048:], dtype=torch.float32), prediction_length=horizon, limit_prediction_length=True)
        # Official Bolt output is [batch, quantile, horizon]. Use its median.
        result = prediction[:, _PIPELINE.quantiles.index(0.5), :].detach().cpu().numpy().astype(float)
        inference_seconds = time.perf_counter() - inference_start
    if result.shape != (2, horizon) or not np.isfinite(result).all():
        raise RuntimeError("Chronos returned invalid forecast values")
    return result, load_seconds, inference_seconds


def forecast_frame(history, horizon_steps=24, backend="seasonal", model_id=MODEL_ID):
    start = time.perf_counter()
    history = validate_history(history)
    horizon = integer(horizon_steps, "horizon_steps", 1, MAX_HORIZON_STEPS)
    if backend not in ("seasonal", "chronos"):
        raise ValueError("backend must be 'seasonal' or 'chronos'")
    interval = int((history.timestamp.iloc[1] - history.timestamp.iloc[0]).total_seconds() / 60)
    values = history[["load_kw", "solar_kw"]].to_numpy().T
    warnings = []
    load_seconds = 0.0
    if backend == "chronos":
        if model_id != MODEL_ID:
            raise ValueError(f"only {MODEL_ID} is supported to bound model resources")
        if horizon > 64:
            raise ValueError("Chronos-Bolt horizon_steps must be <= 64")
        result, load_seconds, inference_seconds = _chronos(values, horizon)
        if (result < 0).any():
            warnings.append("Negative model predictions were clipped to zero.")
        result = np.maximum(result, 0)
        warnings.append("Chronos-Bolt median forecasts are uncertain; no site-specific training or calibration was performed.")
    else:
        inference_start = time.perf_counter()
        period = min(len(history), max(1, round(1440 / interval)))
        result = values[:, -period:][:, np.arange(horizon) % period]
        inference_seconds = time.perf_counter() - inference_start
        warnings.append("Offline seasonal baseline repeats the latest daily cycle; this is not Chronos inference.")
        if len(history) * interval < 1440:
            warnings.append("History is shorter than one day; the available history is repeated.")
        if 1440 % interval:
            warnings.append("Sampling interval does not divide one day; the repeated daily period is approximate.")
    timestamps = pd.date_range(history.timestamp.iloc[-1] + pd.Timedelta(minutes=interval), periods=horizon, freq=f"{interval}min")
    return {"backend": backend, "model_id": MODEL_ID if backend == "chronos" else None,
            "interval_minutes": interval, "timestamps": [stamp.isoformat() for stamp in timestamps],
            "load_kw": result[0].tolist(), "solar_kw": result[1].tolist(), "warnings": warnings,
            "timing": {"model_load_seconds": load_seconds, "inference_seconds": inference_seconds, "total_seconds": time.perf_counter() - start}}
