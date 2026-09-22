from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

FIELDS = ROOT / "artifacts/taylor-green/fields.jsonl"
OUT = ROOT / "evidence/taylor-green.png"

with open(FIELDS) as f:
    frames = [json.loads(line) for line in f]

f0 = frames[0]
f1 = frames[-1]

n = int(np.sqrt(len(f0["omega"])))

x = np.linspace(0.0, 2*np.pi, n, endpoint=False)
y = np.linspace(0.0, 2*np.pi, n, endpoint=False)

omega0 = np.array(f0["omega"]).reshape(n, n)
omega1 = np.array(f1["omega"]).reshape(n, n)

u0 = np.array(f0["u"]).reshape(n, n)
v0 = np.array(f0["v"]).reshape(n, n)

u1 = np.array(f1["u"]).reshape(n, n)
v1 = np.array(f1["v"]).reshape(n, n)

limit = max(
    np.max(np.abs(omega0)),
    np.max(np.abs(omega1)),
)

fig, axes = plt.subplots(
    1, 2,
    figsize=(11, 4.8),
    constrained_layout=True,
)

skip = 4

for ax, om, u, v, t in [
    (axes[0], omega0, u0, v0, f0["t"]),
    (axes[1], omega1, u1, v1, f1["t"]),
]:
    im = ax.imshow(
        om,
        origin="lower",
        extent=[0, 2*np.pi, 0, 2*np.pi],
        vmin=-limit,
        vmax=limit,
        aspect="equal",
    )

    ax.quiver(
        x[::skip],
        y[::skip],
        u[::skip, ::skip],
        v[::skip, ::skip],
angles="xy",
    scale_units="xy",
    scale=2.5,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"t = {t:.1f}, max|omega| = {np.max(np.abs(om)):.3f}"
    )

fig.colorbar(
    im,
    ax=axes,
    label="vorticity omega",
)

OUT.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(
    OUT,
    dpi=180,
)

plt.close(fig)

print(f"Wrote {OUT}")
