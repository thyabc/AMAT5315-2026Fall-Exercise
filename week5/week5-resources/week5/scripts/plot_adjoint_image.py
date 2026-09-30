import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    image = np.load(
        "artifacts/adjoint/image.npy"
    )

    with open("inputs/reflector.json", "r") as f:
        exp = json.load(f)

    perturbation = np.asarray(
        exp["perturbation"],
        dtype=float
    )

    dx_km = (
        exp["dx"]
        * exp["length_unit_m"]
        / 1000.0
    )

    # Assignment window:
    # x = 7..33, z = 10..33 inclusive
    x0, x1 = 7, 33
    z0, z1 = 10, 33

    pert_win = perturbation[
        z0:z1 + 1,
        x0:x1 + 1
    ]

    image_win = image[
        z0:z1 + 1,
        x0:x1 + 1
    ]

    x = np.arange(
        x0,
        x1 + 1
    ) * dx_km

    z = np.arange(
        z0,
        z1 + 1
    ) * dx_km

    # Row L2 norm of RTM image
    profile = np.sqrt(
        np.sum(
            image_win ** 2,
            axis=1
        )
    )

    peak_local = np.argmax(profile)
    peak_z_index = z0 + peak_local

    true_z_index = 21

    peak_depth = peak_z_index * dx_km
    true_depth = true_z_index * dx_km

    depth_error = abs(
        peak_depth - true_depth
    )

    print(
        "true reflector depth =",
        true_depth,
        "km"
    )

    print(
        "image peak depth =",
        peak_depth,
        "km"
    )

    print(
        "depth error =",
        depth_error,
        "km"
    )

    print(
        "image L2 norm =",
        np.linalg.norm(image)
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 5),
        sharey=True
    )

    extent = [
        x[0],
        x[-1],
        z[-1],
        z[0],
    ]

    # ======================================================
    # 1. Known perturbation
    # ======================================================

    vmax_p = np.max(
        np.abs(pert_win)
    )

    im0 = axes[0].imshow(
        pert_win,
        extent=extent,
        origin="upper",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax_p,
        vmax=vmax_p
    )

    axes[0].axhline(
        true_depth,
        linestyle="--",
        linewidth=1
    )

    axes[0].set_title(
        "1. Known reflector"
    )

    axes[0].set_xlabel(
        "Horizontal position (km)"
    )

    axes[0].set_ylabel(
        "Depth (km)"
    )

    cb0 = fig.colorbar(
        im0,
        ax=axes[0]
    )

    cb0.set_label(
        "Velocity change (km/s)"
    )

    # ======================================================
    # 2. Raw RTM image
    # ======================================================

    vmax_i = np.max(
        np.abs(image_win)
    )

    im1 = axes[1].imshow(
        image_win,
        extent=extent,
        origin="upper",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax_i,
        vmax=vmax_i
    )

    axes[1].axhline(
        true_depth,
        linestyle="--",
        linewidth=1
    )

    axes[1].set_title(
        "2. Raw signed RTM image"
    )

    axes[1].set_xlabel(
        "Horizontal position (km)"
    )

    cb1 = fig.colorbar(
        im1,
        ax=axes[1]
    )

    cb1.set_label(
        "Image (arbitrary units)"
    )

    # ======================================================
    # 3. Depth profile
    # ======================================================

    axes[2].plot(
        profile,
        z
    )

    axes[2].axhline(
        true_depth,
        linestyle="--",
        label=f"Known: {true_depth:.1f} km"
    )

    axes[2].axhline(
        peak_depth,
        linestyle=":",
        label=f"Peak: {peak_depth:.1f} km"
    )

    axes[2].set_xlabel(
        "Row L2 norm"
    )

    axes[2].set_title(
        "3. Depth profile"
    )

    axes[2].legend()

    axes[2].invert_yaxis()

    fig.suptitle(
        "RTM locates the reflector; image amplitudes are not velocity"
    )

    fig.tight_layout()

    out = Path(
        "artifacts/adjoint/image.png"
    )

    fig.savefig(
        out,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", out)


if __name__ == "__main__":
    main()
