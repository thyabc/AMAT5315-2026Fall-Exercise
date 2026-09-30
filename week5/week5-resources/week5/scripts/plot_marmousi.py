import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    # ======================================================
    # Load experiment and results
    # ======================================================

    with open("inputs/marmousi.json", "r") as f:
        exp = json.load(f)

    background = np.asarray(
        exp["background"],
        dtype=float
    )

    perturbation = np.asarray(
        exp["perturbation"],
        dtype=float
    )

    born = np.load(
        "artifacts/marmousi-born/born_data.npy"
    )

    image = np.load(
        "artifacts/marmousi-image/image.npy"
    )

    with open(
        "artifacts/marmousi-image/result.json",
        "r"
    ) as f:
        result = json.load(f)

    # ======================================================
    # Physical coordinates
    # ======================================================

    length_unit_m = exp.get(
        "length_unit_m",
        1000.0
    )

    time_unit_s = exp.get(
        "time_unit_s",
        1.0
    )

    dx_km = (
        exp["dx"]
        * length_unit_m
        / 1000.0
    )

    dt_s = (
        exp["dt"]
        * time_unit_s
    )

    nx = exp["nx"]
    nz = exp["nz"]
    steps = exp["steps"]

    xmax = (nx - 1) * dx_km
    zmax = (nz - 1) * dx_km

    # ======================================================
    # Find shot closest to x = 10 km
    # ======================================================

    shot_x = np.asarray([
        s[0] * dx_km
        for s in exp["shots"]
    ])

    shot_index = int(
        np.argmin(
            np.abs(
                shot_x - 10.0
            )
        )
    )

    selected_shot_x = shot_x[
        shot_index
    ]

    gather = born[
        shot_index
    ]

    receiver_x = np.asarray([
        r[0] * dx_km
        for r in exp["receivers"]
    ])

    times = (
        np.arange(steps)
        * dt_s
    )

    # ======================================================
    # Numerical checks
    # ======================================================

    image_l2 = np.linalg.norm(image)

    stats = result["statistics"]

    peak_states = stats[
        "peak_saved_states"
    ]

    peak_bytes = stats[
        "peak_saved_bytes"
    ]

    reference_l2 = 6.7037741e-4

    relative_l2_error = abs(
        image_l2 - reference_l2
    ) / reference_l2

    print(
        "Marmousi shot index =",
        shot_index
    )

    print(
        "selected shot x =",
        selected_shot_x,
        "km"
    )

    print(
        "image L2 norm =",
        image_l2
    )

    print(
        "reference L2 =",
        reference_l2
    )

    print(
        "relative L2 error =",
        relative_l2_error
    )

    print(
        "peak saved states =",
        peak_states
    )

    print(
        "peak saved bytes =",
        peak_bytes
    )

    # ======================================================
    # Figure
    # ======================================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10)
    )

    model_extent = [
        0.0,
        xmax,
        zmax,
        0.0,
    ]

    # ------------------------------------------------------
    # Panel 1: background
    # ------------------------------------------------------

    im0 = axes[0, 0].imshow(
        background,
        extent=model_extent,
        origin="upper",
        aspect="auto"
    )

    axes[0, 0].set_title(
        "Smoothed Marmousi background"
    )

    axes[0, 0].set_xlabel(
        "Horizontal position (km)"
    )

    axes[0, 0].set_ylabel(
        "Depth (km)"
    )

    cb0 = fig.colorbar(
        im0,
        ax=axes[0, 0]
    )

    cb0.set_label(
        "Speed (km/s)"
    )

    # mark 8-14 km, upper 3 km comparison region
    axes[0, 0].add_patch(
        Rectangle(
            (8.0, 0.0),
            6.0,
            3.0,
            fill=False,
            linestyle="--",
            linewidth=1.2
        )
    )

    # ------------------------------------------------------
    # Panel 2: perturbation
    # ------------------------------------------------------

    vmax_p = np.max(
        np.abs(perturbation)
    )

    im1 = axes[0, 1].imshow(
        perturbation,
        extent=model_extent,
        origin="upper",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax_p,
        vmax=vmax_p
    )

    axes[0, 1].set_title(
        "Short-wavelength perturbation"
    )

    axes[0, 1].set_xlabel(
        "Horizontal position (km)"
    )

    axes[0, 1].set_ylabel(
        "Depth (km)"
    )

    cb1 = fig.colorbar(
        im1,
        ax=axes[0, 1]
    )

    cb1.set_label(
        "Velocity perturbation (km/s)"
    )

    axes[0, 1].add_patch(
        Rectangle(
            (8.0, 0.0),
            6.0,
            3.0,
            fill=False,
            linestyle="--",
            linewidth=1.2
        )
    )

    # ------------------------------------------------------
    # Panel 3: Born gather
    # ------------------------------------------------------

    vmax_g = np.max(
        np.abs(gather)
    )

    gather_extent = [
        receiver_x.min(),
        receiver_x.max(),
        times[-1],
        times[0],
    ]

    im2 = axes[1, 0].imshow(
        gather,
        extent=gather_extent,
        origin="upper",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax_g,
        vmax=vmax_g
    )

    axes[1, 0].set_title(
        f"Born gather; source x = "
        f"{selected_shot_x:.1f} km"
    )

    axes[1, 0].set_xlabel(
        "Receiver position (km)"
    )

    axes[1, 0].set_ylabel(
        "Time (s)"
    )

    cb2 = fig.colorbar(
        im2,
        ax=axes[1, 0]
    )

    cb2.set_label(
        "Scattered pressure (arbitrary units)"
    )

    # ------------------------------------------------------
    # Panel 4: raw image
    #
    # IMPORTANT:
    # one amplitude scale over all depths,
    # no depth gain.
    # ------------------------------------------------------

    vmax_i = np.max(
        np.abs(image)
    )

    im3 = axes[1, 1].imshow(
        image,
        extent=model_extent,
        origin="upper",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax_i,
        vmax=vmax_i
    )

    axes[1, 1].set_title(
        "Checkpointed migration image"
    )

    axes[1, 1].set_xlabel(
        "Horizontal position (km)"
    )

    axes[1, 1].set_ylabel(
        "Depth (km)"
    )

    cb3 = fig.colorbar(
        im3,
        ax=axes[1, 1]
    )

    cb3.set_label(
        "Adjoint image (arbitrary units)"
    )

    axes[1, 1].add_patch(
        Rectangle(
            (8.0, 0.0),
            6.0,
            3.0,
            fill=False,
            linestyle="--",
            linewidth=1.2
        )
    )

    fig.suptitle(
        "Marmousi Born modeling and "
        "Treeverse reverse-time migration"
    )

    fig.tight_layout()

    out = Path(
        "artifacts/marmousi.png"
    )

    fig.savefig(
        out,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "Saved:",
        out
    )

    # ======================================================
    # Assignment checks
    # ======================================================

    assert (
        relative_l2_error < 1e-4
    ), (
        "Marmousi image L2 does not "
        "match reference"
    )

    assert (
        peak_states == 6
    ), (
        f"expected 6 saved states, "
        f"got {peak_states}"
    )

    assert (
        peak_bytes == 20788320
    ), (
        f"expected 20788320 bytes, "
        f"got {peak_bytes}"
    )

    print(
        "PASS: Marmousi verification."
    )


if __name__ == "__main__":
    main()
