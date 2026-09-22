from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "evidence" / "line-accuracy.png"


# ============================================================
# Load pulse-race data
# ============================================================
race = np.genfromtxt(
    ART / "line-race.csv",
    delimiter=",",
    names=True,
)

# ============================================================
# Load order data
# ============================================================
order = np.genfromtxt(
    ART / "line-order.csv",
    delimiter=",",
    names=True,
    dtype=None,
    encoding="utf-8",
)

methods = [
    ("euler", "Euler"),
    ("rk2", "Midpoint / RK2"),
    ("rk4", "RK4"),
    ("rk4-broken", "equal-weight RK4"),
]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.5, 4.8),
    constrained_layout=True,
)

# ============================================================
# Left: pulse race
# ============================================================
ax = axes[0]

ax.plot(
    race["x"],
    race["exact"],
    linestyle="--",
    linewidth=2,
    label="exact",
)

ax.plot(
    race["x"],
    race["fourier_rk4"],
    label="RK4 + Fourier, dt=0.02",
)

ax.plot(
    race["x"],
    race["centered_rk4"],
    label="RK4 + centred FD, dt=0.02",
)

ax.plot(
    race["x"],
    race["fourier_euler"],
    label="Euler + Fourier, dt=0.005",
)

ax.set_xlabel("x")
ax.set_ylabel("u")
ax.set_title("One lap of the Gaussian pulse")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)


# ============================================================
# Right: convergence order
# ============================================================
ax = axes[1]

for method, label in methods:
    mask = order["method"] == method

    dt = np.asarray(order["dt"][mask], dtype=float)
    err = np.asarray(order["error"][mask], dtype=float)

    sort = np.argsort(dt)
    dt = dt[sort]
    err = err[sort]

    slope, intercept = np.polyfit(
        np.log(dt),
        np.log(err),
        1,
    )

    ax.loglog(
        dt,
        err,
        marker="o",
        label=f"{label}, slope={slope:.2f}",
    )

    print(
        f"{label:20s} slope = {slope:.5f}"
    )

ax.set_xlabel("time step dt")
ax.set_ylabel("maximum error at t=1")
ax.set_title("Time-integrator accuracy")
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=8)

OUT.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(
    OUT,
    dpi=180,
)

plt.close(fig)

print(f"Wrote {OUT}")
