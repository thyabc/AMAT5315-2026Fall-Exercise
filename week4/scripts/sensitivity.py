from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/sensitivity"
OUT = ROOT / "evidence/sensitivity.png"


def read_fields(folder):
    frames = []

    with open(ART / folder / "fields.jsonl") as f:
        for line in f:
            frames.append(json.loads(line))

    return frames


def distances(base_name, pert_name):
    base = read_fields(base_name)
    pert = read_fields(pert_name)

    assert len(base) == len(pert)

    times = []
    distances = []

    for a, b in zip(base, pert):
        assert a["step"] == b["step"]

        wa = np.asarray(a["omega"], dtype=float)
        wb = np.asarray(b["omega"], dtype=float)

        numerator = np.linalg.norm(
            wb - wa
        )

        denominator = np.linalg.norm(
            wa
        )

        times.append(a["t"])

        distances.append(
            numerator / denominator
        )

    return (
        np.asarray(times),
        np.asarray(distances),
    )


tr, dr = distances(
    "random-base",
    "random-perturbed",
)

tt, dtg = distances(
    "tg-base",
    "tg-perturbed",
)

fig, ax = plt.subplots(
    figsize=(7.5, 5),
    constrained_layout=True,
)

positive_r = dr > 0
positive_t = dtg > 0

ax.semilogy(
    tr[positive_r],
    dr[positive_r],
    label="random flow",
)

ax.semilogy(
    tt[positive_t],
    dtg[positive_t],
    label="Taylor-Green",
)

ax.axhline(
    1.0e-6,
    linestyle="--",
    linewidth=1,
    label="6-decimal storage floor",
)

ax.set_xlabel("time t")

ax.set_ylabel(
    r"$||\omega_1-\omega_2||/||\omega_1||$"
)

ax.set_title(
    "Sensitivity to a small initial perturbation"
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

print("random initial =", dr[0])
print("random final   =", dr[-1])
print("random growth  =", dr[-1] / dr[0])

print("TG initial     =", dtg[0])
print("TG final       =", dtg[-1])

print(f"Wrote {OUT}")
