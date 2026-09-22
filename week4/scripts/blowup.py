from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/scan"
OUT = ROOT / "evidence/blowup.png"


def load_tsv(name):
    path = ART / name

    data = np.genfromtxt(
        path,
        delimiter="\t",
        names=True,
    )

    return data


tg32 = load_tsv("tg-0032.tsv")
tg33 = load_tsv("tg-0033.tsv")

r30 = load_tsv("random-rk4-0030.tsv")
r32 = load_tsv("random-rk4-0032.tsv")
reu = load_tsv("random-euler-001.tsv")


fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4.8),
    constrained_layout=True,
)

# ============================================================
# Taylor-Green
# ============================================================
ax = axes[0]

for d, label in [
    (tg32, "RK4, dt=0.032"),
    (tg33, "RK4, dt=0.033"),
]:
    finite = (
        np.isfinite(d["E"])
        & (d["E"] > 0)
    )

    ax.semilogy(
        d["t"][finite],
        d["E"][finite],
        marker="o",
        markersize=3,
        label=label,
    )

    if np.any(~finite):
        first_bad = np.argmax(~finite)

        ax.axvline(
            d["t"][first_bad],
            linestyle=":",
            linewidth=1,
        )

# Exact Taylor-Green energy
t = np.linspace(0.0, 8.0, 400)

exact = 0.25 * np.exp(-0.4 * t)

ax.semilogy(
    t,
    exact,
    linestyle="--",
    label="exact",
)

ax.set_xlabel("time t")
ax.set_ylabel("energy E(t)")
ax.set_title(
    "Taylor-Green: diffusive stability"
)
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=8)


# ============================================================
# Random
# ============================================================
ax = axes[1]

for d, label in [
    (r30, "RK4, dt=0.030"),
    (r32, "RK4, dt=0.032"),
    (reu, "Euler, dt=0.01"),
]:
    finite = (
        np.isfinite(d["E"])
        & (d["E"] > 0)
    )

    ax.semilogy(
        d["t"][finite],
        d["E"][finite],
        marker="o",
        markersize=3,
        label=label,
    )

    if np.any(~finite):
        first_bad = np.argmax(~finite)

        ax.axvline(
            d["t"][first_bad],
            linestyle=":",
            linewidth=1,
        )

ax.set_xlabel("time t")
ax.set_ylabel("energy E(t)")
ax.set_title(
    "Random flow: advective stability"
)
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=8)

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
