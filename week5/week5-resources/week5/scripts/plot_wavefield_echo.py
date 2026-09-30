import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_field(
    field,
    title,
    output,
    extent,
    label,
):
    vmax = np.max(np.abs(field))

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    im = ax.imshow(
        field,
        extent=extent,
        origin="upper",
        aspect="equal",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_xlabel(
        "Horizontal position (km)"
    )

    ax.set_ylabel(
        "Depth (km)"
    )

    ax.set_title(title)

    cbar = fig.colorbar(
        im,
        ax=ax
    )

    cbar.set_label(label)

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=200
    )

    plt.close(fig)


def main():
    wavefield = np.load(
        "artifacts/forward/wavefield.npy"
    )

    echo = np.load(
        "artifacts/forward/echo.npy"
    )

    with open(
        "inputs/reflector.json",
        "r"
    ) as f:
        exp = json.load(f)

    every = 3

    target_step = 150

    # Recorded frames are at:
    # step = 3, 6, 9, ...
    frame_index = target_step // every - 1

    print(
        "frame index for step 150 =",
        frame_index
    )

    w = wavefield[frame_index]
    e = echo[frame_index]

    length_km = (
        exp["length_unit_m"]
        / 1000.0
    )

    xmax = (
        (exp["nx"] - 1)
        * exp["dx"]
        * length_km
    )

    zmax = (
        (exp["nz"] - 1)
        * exp["dx"]
        * length_km
    )

    extent = [
        0.0,
        xmax,
        zmax,
        0.0,
    ]

    time_s = (
        target_step
        * exp["dt"]
        * exp["time_unit_s"]
    )

    out_dir = Path(
        "artifacts/forward"
    )

    plot_field(
        w,
        f"Forward wavefield; shot 0, t = {time_s:.2f} s",
        out_dir / "wavefield.png",
        extent,
        "Pressure (arbitrary units)",
    )

    plot_field(
        e,
        f"Reflector echo; shot 0, t = {time_s:.2f} s",
        out_dir / "echo.png",
        extent,
        "Pressure difference (arbitrary units)",
    )

    print()
    print(
        "wavefield max abs =",
        np.max(np.abs(w))
    )

    print(
        "echo max abs =",
        np.max(np.abs(e))
    )

    print()
    print(
        "Saved:",
        out_dir / "wavefield.png"
    )

    print(
        "Saved:",
        out_dir / "echo.png"
    )


if __name__ == "__main__":
    main()
