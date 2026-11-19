#!/usr/bin/env python3
"""Plot Part 3 contraction scaling and the Factor-15 Boltzmann histogram."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "artifacts" / "factoring.png"

ks = list(range(2, 9))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ax = axes[0]
for order, marker in (("greedy", "o"), ("treesa", "s")):
    xs, ys = [], []
    for k in ks:
        p = RUNS / f"plan_k{k}_{order}" / "run.json"
        if p.exists():
            d = json.loads(p.read_text())
            xs.append(k)
            ys.append(float(d["sc"]))
    if xs:
        ax.plot(xs, ys, marker=marker, linewidth=2, label=order)
ax.set_xlabel("bits per factor k")
ax.set_ylabel("space complexity sc (log₂ entries)")
ax.set_title("Contraction memory cost")
ax.set_xticks(ks)
ax.grid(True, alpha=0.25)
ax.legend()

sample_path = RUNS / "sample_f15" / "run.json"
if sample_path.exists():
    d = json.loads(sample_path.read_text())
    hist = d.get("histogram", [])
    energies = [int(x["energy"]) for x in hist]
    counts = [int(x["count"]) for x in hist]
    ax = axes[1]
    ax.bar([str(x) for x in energies], counts, width=0.8)
    ax.set_xlabel("spin-glass energy E")
    ax.set_ylabel("samples (out of {})".format(d.get("samples_requested", 2000)))
    ax.set_title("Factor 15 sampled states (β = {})".format(d.get("beta", 2.0)))
    ax.tick_params(axis="x", rotation=45)
    emin = d.get("energy_min")
    predicted = d.get("predicted_ground_state_samples")
    if emin is not None and predicted is not None:
        ax.annotate("ground state E={}: observed {}, predicted {:.0f}".format(
            emin, dict(zip(energies, counts)).get(int(emin), 0), predicted),
            xy=(0.02, 0.96), xycoords="axes fraction", va="top", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
else:
    axes[1].text(0.5, 0.5, "Run sample_f15 before plotting", ha="center", va="center")
    axes[1].set_axis_off()

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Saved {OUT}")
