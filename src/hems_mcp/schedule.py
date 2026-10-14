"""Continuous charging exploration that minimizes extra grid-import energy."""

import numpy as np

from .schemas import forecast_arrays, integer, number


def explore_schedule(forecast, energy_kwh=10, max_charge_kw=7.2, grid_limit_kw=10, efficiency=0.9, arrival_step=0, departure_step=None):
    timestamps, load, solar, hours = forecast_arrays(forecast)
    energy = number(energy_kwh, "energy_kwh")
    max_charge = number(max_charge_kw, "max_charge_kw")
    grid_limit = number(grid_limit_kw, "grid_limit_kw")
    safe_power = np.finfo(float).max / (len(load) * max(1, hours) * 2)
    if max_charge > safe_power or grid_limit > safe_power:
        raise ValueError("charger and grid powers are too large for finite energy arithmetic")
    if grid_limit <= 0:
        raise ValueError("grid_limit_kw must be greater than zero")
    efficiency = number(efficiency, "efficiency")
    if not 0 < efficiency <= 1:
        raise ValueError("efficiency must be greater than zero and <= 1")
    arrival = integer(arrival_step, "arrival_step", 0, len(load))
    departure = len(load) if departure_step is None else integer(departure_step, "departure_step", 0, len(load))
    if departure <= arrival:
        raise ValueError("departure_step must be > arrival_step")
    net = load - solar
    capacity = np.minimum(max_charge, np.maximum(grid_limit - net, 0))
    capacity[:arrival] = 0
    capacity[departure:] = 0
    charge = np.zeros(len(load))
    remaining = energy / efficiency
    # All solar-surplus kWh have zero incremental import; other kWh have equal cost.
    free_solar = np.minimum(capacity, np.maximum(-net, 0))
    low_demand_order = sorted(range(arrival, departure), key=lambda step: (net[step], step))
    for allowance, order in ((free_solar, range(arrival, departure)), (capacity, low_demand_order)):
        for step in order:
            extra = min(max(0, allowance[step] - charge[step]), remaining / hours)
            charge[step] += extra
            remaining = max(0, remaining - extra * hours)
    delivered = min(energy, float(charge.sum() * hours * efficiency))
    unmet = max(0, energy - delivered)
    baseline_import = np.maximum(net, 0)
    scheduled_import = np.maximum(net + charge, 0)
    warnings = ["Exploration only: constant import cost, no tariffs, hardware controls, or battery-state model."]
    baseline_violation = bool((baseline_import > grid_limit + 1e-9).any())
    if baseline_violation:
        warnings.append("Household baseline already exceeds the grid limit; EV charging cannot remove that violation.")
    if unmet > 1e-8:
        warnings.append("Available charging capacity cannot deliver the requested battery energy.")
    return {"status": "infeasible" if unmet > 1e-8 or baseline_violation else "feasible",
            "requested_energy_kwh": energy, "delivered_energy_kwh": delivered, "unmet_energy_kwh": unmet,
            "charging_energy_kwh": float(charge.sum() * hours), "charging_kw": charge.tolist(),
            "timestamps": [stamp.isoformat() for stamp in timestamps],
            "baseline_import_energy_kwh": float(baseline_import.sum() * hours),
            "scheduled_import_energy_kwh": float(scheduled_import.sum() * hours),
            "baseline_peak_import_kw": float(baseline_import.max()), "scheduled_peak_import_kw": float(scheduled_import.max()),
            "warnings": warnings}
