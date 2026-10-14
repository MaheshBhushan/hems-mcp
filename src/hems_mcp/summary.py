"""Energy integrals and power peaks for slot-average forecasts."""

import numpy as np
import pandas as pd

from .schemas import forecast_arrays


def summarize_forecast(forecast):
    timestamps, load, solar, hours = forecast_arrays(forecast)
    imports = np.maximum(load - solar, 0)
    exports = np.maximum(solar - load, 0)
    boundaries = np.diff(np.r_[False, exports > 0, False].astype(int))
    windows = [
        {"start": timestamps[start].isoformat(),
         "end": (timestamps[end - 1] + pd.Timedelta(hours=hours)).isoformat(),
         "start_step": int(start), "end_step": int(end),
         "surplus_kwh": float(exports[start:end].sum() * hours)}
        for start, end in zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1))
    ]
    result = {"horizon_steps": len(load), "duration_hours": len(load) * hours,
            "interval_minutes": int(forecast["interval_minutes"]),
            "start": timestamps[0].isoformat(),
            "end": (timestamps[-1] + pd.Timedelta(hours=hours)).isoformat(),
            "load_energy_kwh": float(load.sum() * hours), "solar_energy_kwh": float(solar.sum() * hours),
            "import_energy_kwh": float(imports.sum() * hours), "export_energy_kwh": float(exports.sum() * hours),
            "peak_load_kw": float(load.max()), "peak_import_kw": float(imports.max()), "peak_export_kw": float(exports.max()),
            "peak_load_timestamp": timestamps[int(load.argmax())].isoformat(),
            "peak_import_timestamp": timestamps[int(imports.argmax())].isoformat(),
            "peak_export_timestamp": timestamps[int(exports.argmax())].isoformat(),
            "surplus_energy_kwh": float(exports.sum() * hours), "surplus_windows": windows,
            "solar_self_consumption_kwh": float(np.minimum(load, solar).sum() * hours)}
    window_text = "; ".join(f"{window['start']} to {window['end']}: {window['surplus_kwh']:.3f} kWh" for window in windows) or "none"
    result["text"] = (
        f"Forecast spans {result['start']} to {result['end']} (end exclusive). "
        f"Load {result['load_energy_kwh']:.3f} kWh; solar {result['solar_energy_kwh']:.3f} kWh; "
        f"import {result['import_energy_kwh']:.3f} kWh; surplus {result['surplus_energy_kwh']:.3f} kWh. "
        f"Earliest load peak {result['peak_load_kw']:.3f} kW at {result['peak_load_timestamp']}; "
        f"earliest import peak {result['peak_import_kw']:.3f} kW at {result['peak_import_timestamp']}. "
        f"Surplus windows (end exclusive): {window_text}. Text rounding is presentation-only."
    )
    return result
