from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "evidence" / "line-stability.png"


def load_map():
    data = np.genfromtxt(
        ART / "line-stability-map.csv",
        delimiter=",",
        names=True,
    )

    re_unique = np.unique(data["re"])
    im_unique = np.unique(data["im"])

    growth = data["growth"].reshape(
        len(im_unique),
        len(re_unique),
    )

    return re_unique, im_unique, growth


def load_boundaries():
    data = np.genfromtxt(
        ART / "stability-boundaries.csv",
        delimiter=",",
        names=True,
    )

    re_unique = np.unique(data["re"])
    im_unique = np.unique(data["im"])

    shape = (len(im_unique), len(re_unique))

    return (
        re_unique,
        im_unique,
        data["euler"].reshape(shape),
        data["rk2"].reshape(shape),
        data["rk4"].reshape(shape),
    )


def load_pulse(filename):
    data = np.genfromtxt(
        ART / filename,
        delimiter=",",
        names=True,
    )

    t = np.unique(data["t"])
    x = np.unique(data["x"])

    u = data["u"].reshape(len(t), len(x))

    return x, t, u


re, im, growth = load_map()

(
    bre,
    bim,
    euler_r,
    rk2_r,
    rk4_r,
) = load_boundaries()

modes = np.genfromtxt(
    ART / "line-modes.csv",
    delimiter=",",
    names=True,
)

x45, t45, u45 = load_pulse("pulse-dt-0.045.csv")
x56, t56, u56 = load_pulse("pulse-dt-0.056.csv")

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4.8),
    constrained_layout=True,
)

# --------------------------------------------------------------
# Panel 1: stability map
# --------------------------------------------------------------
ax = axes[0]

log_growth = np.log10(np.maximum(growth, 1.0e-4))

image = ax.pcolormesh(
    re,
    im,
    log_growth,
    shading="auto",
)

ax.contour(
    bre,
    bim,
    rk4_r,
    levels=[1.0],
    linewidths=2.0,
)

ax.contour(
    bre,
    bim,
    euler_r,
    levels=[1.0],
    linestyles="--",
)

ax.contour(
    bre,
    bim,
    rk2_r,
    levels=[1.0],
    linestyles=":",
)

for dt, marker, label in [
    (0.045, "o", "dt = 0.045"),
    (0.056, "x", "dt = 0.056"),
]:
    mask = np.isclose(modes["dt"], dt)

    ax.scatter(
        modes["re"][mask],
        modes["im"][mask],
        s=14,
        marker=marker,
        label=label,
    )

ax.axhline(0.0, linewidth=0.6)
ax.axvline(0.0, linewidth=0.6)

ax.set_xlabel("Re(z)")
ax.set_ylabel("Im(z)")
ax.set_title("Measured RK4 growth")

ax.set_xlim(-4.0, 1.0)
ax.set_ylim(-4.0, 4.0)
ax.set_aspect("equal", adjustable="box")

ax.legend(fontsize=8)

fig.colorbar(
    image,
    ax=ax,
    label="log10 growth per step",
)

# --------------------------------------------------------------
# Panel 2: stable pulse
# --------------------------------------------------------------
ax = axes[1]

im1 = ax.imshow(
    u45,
    aspect="auto",
    origin="upper",
    extent=[
        x45.min(),
        x45.max(),
        t45.max(),
        t45.min(),
    ],
 vmin=-1.0,
    vmax=1.0,
)

ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_title("RK4, dt = 0.045")

fig.colorbar(im1, ax=ax, label="u")

# --------------------------------------------------------------
# Panel 3: unstable pulse
# --------------------------------------------------------------
ax = axes[2]

im2 = ax.imshow(
    u56,
    aspect="auto",
    origin="upper",
    extent=[
        x56.min(),
        x56.max(),
        t56.max(),
        t56.min(),
    ],
 vmin=-1.0,
    vmax=1.0,
)

ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_title("RK4, dt = 0.056")

fig.colorbar(im2, ax=ax, label="u")

OUT.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(
    OUT,
    dpi=180,
)

plt.close(fig)

print(f"Wrote {OUT}")
