import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    with open("inputs/reflector.json", "r") as f:
        exp = json.load(f)

    nx = exp["nx"]
    nz = exp["nz"]
    dx = exp["dx"]
    dt = exp["dt"]
    steps = exp["steps"]

    # Reduced length unit -> km
    length_km = exp["length_unit_m"] / 1000.0
    time_s = exp["time_unit_s"]

    background = np.asarray(exp["background"], dtype=float)
    perturbation = np.asarray(exp["perturbation"], dtype=float)

    shots = np.asarray(exp["shots"], dtype=float)
    receivers = np.asarray(exp["receivers"], dtype=float)

    x = np.arange(nx) * dx * length_km
    z = np.arange(nz) * dx * length_km

    X, Z = np.meshgrid(x, z)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5)
    )

    # ======================================================
    # Geometry
    # ======================================================

    ax = axes[0]

    pcm = ax.pcolormesh(
        X,
        Z,
        background,
        shading="nearest"
    )

    cbar = fig.colorbar(
        pcm,
        ax=ax
    )

    cbar.set_label("Speed (km/s)")

    # Thin reflector / perturbation
    levels = [
        0.02 * np.max(perturbation)
    ]

    ax.contour(
        X,
        Z,
        perturbation,
        levels=levels,
        linewidths=2
    )

    # Shot positions
    ax.scatter(
        shots[:, 0] * dx * length_km,
        shots[:, 1] * dx * length_km,
        marker="*",
        s=130,
        label="Sources"
    )

    # Receiver positions
    ax.scatter(
        receivers[:, 0] * dx * length_km,
        receivers[:, 1] * dx * length_km,
        marker="v",
        s=40,
        label="Receivers"
    )

    # Sponge inner boundary
    w = exp["sponge_width"]

    left = w * dx * length_km
    right = (nx - 1 - w) * dx * length_km
    top = w * dx * length_km
    bottom = (nz - 1 - w) * dx * length_km

    ax.plot(
        [left, right, right, left, left],
        [top, top, bottom, bottom, top],
        "--",
        linewidth=1,
        label="Sponge inner edge"
    )

    ax.set_xlabel("Horizontal position (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_title("Seismic acquisition")

    # Depth increases downward
    ax.invert_yaxis()

    ax.legend(
        loc="lower right"
    )

    # ======================================================
    # Ricker pulse
    # ======================================================

    ax = axes[1]

    n = np.arange(steps)

    # Reduced time
    t_reduced = n * dt

    theta = (
        np.pi
        * exp["source_frequency"]
        * (t_reduced - exp["source_peak_time"])
    )

    pulse = (
        1.0 - 2.0 * theta**2
    ) * np.exp(-theta**2)

    # physical seconds
    t_seconds = (
        t_reduced * time_s
    )

    ax.plot(
        t_seconds,
        pulse,
        linewidth=2
    )

    ax.axhline(
        0.0,
        linewidth=0.8
    )

    peak_frequency_hz = (
        exp["source_frequency"]
        / time_s
    )

    peak_time_s = (
        exp["source_peak_time"]
        * time_s
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Source amplitude")

    ax.set_title(
        f"Ricker pulse; peak frequency "
        f"{peak_frequency_hz:.1f} Hz, "
        f"peak time {peak_time_s:.1f} s"
    )

    fig.tight_layout()

    out = Path(
        "artifacts/inputs.png"
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        out,
        dpi=200
    )

    plt.close(fig)

    print("Saved:", out)

    print(
        "background speed =",
        np.min(background),
        "to",
        np.max(background)
    )

    print(
        "maximum perturbation =",
        np.max(perturbation)
    )

    print(
        "pulse maximum =",
        np.max(pulse)
    )

    print(
        "pulse maximum time =",
        t_seconds[np.argmax(pulse)],
        "s"
    )


if __name__ == "__main__":
    main()
