from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/order"
OUT = ROOT / "evidence/order.png"

runs = [
    (0.4, "rk4-dt0.4"),
    (0.25, "rk4-dt0.25"),
    (0.2, "rk4-dt0.2"),
]

with open(ART / "exact-t2.json") as f:
    exact = json.load(f)

ue = np.asarray(exact["u"], dtype=float)
ve = np.asarray(exact["v"], dtype=float)

den = np.sum(ue**2 + ve**2)

dts = []
errors = []

for dt, folder in runs:

    with open(ART / folder / "fields.jsonl") as f:
        frames = [
            json.loads(line)
            for line in f
        ]

    final = frames[-1]

    u = np.asarray(final["u"], dtype=float)
    v = np.asarray(final["v"], dtype=float)

    num = np.sum(
        (u - ue)**2
        + (v - ve)**2
    )

    error = np.sqrt(num / den)

    dts.append(dt)
    errors.append(error)

    print(
        f"dt={dt:<6} "
        f"final t={final['t']} "
        f"error={error:.8e}"
    )

dts = np.asarray(dts)
errors = np.asarray(errors)

slope, intercept = np.polyfit(
    np.log(dts),
    np.log(errors),
    1,
)

print()
print("RK4 fitted slope =", slope)

# Sort for plotting
order = np.argsort(dts)

x = dts[order]
y = errors[order]

fig, ax = plt.subplots(
    figsize=(7.0, 5.0),
    constrained_layout=True,
)

ax.loglog(
    x,
    y,
    marker="o",
    linewidth=2,
    label=f"RK4, slope {slope:.2f}",
)

# fourth-order reference line
reference = (
    y[0]
    * (x / x[0])**4
)

ax.loglog(
    x,
    reference,
    linestyle="--",
    label="slope 4 reference",
)

ax.axhline(
    7e-7,
    linestyle=":",
    label="6-decimal storage floor",
)

ax.set_xlabel("time step dt")

ax.set_ylabel(
    "relative velocity error at t = 2"
)

ax.set_title(
    "Taylor-Green RK4 time accuracy"
)

ax.grid(
    True,
    which="both",
    alpha=0.25,
)

ax.legend()

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fig.savefig(
    OUT,
    dpi=180,
)

plt.close(fig)

print(f"Wrote {OUT}")
