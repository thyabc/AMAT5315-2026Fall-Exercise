from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

FIELDS = ROOT / "artifacts/random/fields.jsonl"
OUT = ROOT / "evidence/random.png"

with open(FIELDS) as f:
    frames = [json.loads(line) for line in f]

wanted = [0.0, 2.0, 5.0, 10.0]

selected = []

for target in wanted:
    frame = min(
        frames,
        key=lambda f: abs(f["t"] - target),
    )
    selected.append(frame)

n = int(np.sqrt(len(selected[0]["omega"])))

# One shared colour scale based on t = 0.
limit = np.max(
    np.abs(
        np.asarray(selected[0]["omega"])
    )
)

fig, axes = plt.subplots(
    1,
    4,
    figsize=(16, 4.1),
    constrained_layout=True,
)

for ax, frame in zip(axes, selected):
    omega = np.asarray(
        frame["omega"]
    ).reshape(n, n)

    u = np.asarray(frame["u"])
    v = np.asarray(frame["v"])

    E = 0.5 * np.mean(
        u*u + v*v
    )

    Z = 0.5 * np.mean(
        omega.ravel() ** 2
    )

    im = ax.imshow(
        omega,
        origin="lower",
        extent=[
            0,
            2*np.pi,
            0,
            2*np.pi,
        ],
        vmin=-limit,
        vmax=limit,
        aspect="equal",
    )

    ax.set_title(
        f"t = {frame['t']:.0f}\n"
        f"E = {E:.3f}, Z = {Z:.3f}"
    )

    ax.set_xlabel("x")

axes[0].set_ylabel("y")

fig.colorbar(
    im,
    ax=axes,
    label="vorticity omega",
    shrink=0.88,
)

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
