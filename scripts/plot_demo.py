"""Plot real demo artifacts. Requires matplotlib in the plotting environment."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
forecast = json.loads((root / "artifacts/demo/forecast.json").read_text())
schedule = json.loads((root / "artifacts/demo/schedule.json").read_text())
with plt.style.context("dark_background"):
    fig, ax = plt.subplots(figsize=(12, 4.8), layout="constrained")
    fig.set_facecolor("#101820")
    ax.set_facecolor("#101820")
    hours = range(len(forecast["load_kw"]))
    ax.plot(hours, forecast["load_kw"], label="Household load", color="#ffbd69", linewidth=2.5)
    ax.plot(hours, forecast["solar_kw"], label="Solar generation", color="#6be5b0", linewidth=2.5)
    ax.step(hours, schedule["charging_kw"], where="post", label="EV charging (AC)", color="#71baff", linewidth=2)
    ax.set(title="HEMS-MCP · Synthetic 24-hour forecast and charging plan", xlabel="Hour (UTC)", ylabel="Average power (kW)")
    ax.set_xlim(0, 23)
    ax.grid(alpha=0.15)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(root / "assets/demo.png", dpi=150, facecolor=fig.get_facecolor())
