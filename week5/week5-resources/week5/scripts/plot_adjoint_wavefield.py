import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    wavefield = np.load(
        "artifacts/adjoint/wavefield.npy"
    )

    with open(
        "artifacts/adjoint/run.json",
        "r"
    ) as f:
        run = json.load(f)

    with open(
        "inputs/reflector.json",
        "r"
    ) as f:
        exp = json.load(f)

    steps = run["recording"]["steps"]

    target_step = 132

    if target_step not in steps:
        raise RuntimeError(
            f"step {target_step} not found; "
            f"recorded steps include {steps[:5]}..."
        )

    frame_index = steps.index(
        target_step
    )

    field = wavefield[
        frame_index
    ]

    dx_km = (
        exp["dx"]
        * exp["length_unit_m"]
        / 1000.0
    )

    xmax = (
        exp["nx"] - 1
    ) * dx_km

    zmax = (
        exp["nz"] - 1
    ) * dx_km

    time_s = (
        target_step
        * exp["dt"]
        * exp["time_unit_s"]
    )

    true_depth = 21 * dx_km

    vmax = np.max(
        np.abs(field)
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    im = ax.imshow(
        field,
        extent=[
            0,
            xmax,
            zmax,
            0
        ],
        origin="upper",
        aspect="equal",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax
    )

    ax.axhline(
        true_depth,
        linestyle="--",
        linewidth=1,
        label="Reflector"
    )

    ax.set_xlabel(
        "Horizontal position (km)"
    )

    ax.set_ylabel(
        "Depth (km)"
    )

    ax.set_title(
        f"Adjoint field at step {target_step}; "
        f"{time_s:.2f} s"
    )

    ax.legend()

    cbar = fig.colorbar(
        im,
        ax=ax
    )

    cbar.set_label(
        "Pressure adjoint (arbitrary units)"
    )

    fig.tight_layout()

    out = Path(
        "artifacts/adjoint/wavefield.png"
    )

    fig.savefig(
        out,
        dpi=200
    )

    plt.close(fig)

    print(
        "frame index =",
        frame_index
    )

    print(
        "time =",
        time_s,
        "s"
    )

    print(
        "max abs adjoint =",
        vmax
    )

    print(
        "Saved:",
        out
    )


if __name__ == "__main__":
    main()
