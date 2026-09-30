import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    traces = np.load("artifacts/forward/traces.npy")

    with open("inputs/reflector.json", "r") as f:
        exp = json.load(f)

    receivers = np.asarray(exp["receivers"])
    shots = np.asarray(exp["shots"])

    dt = exp["dt"]
    dx = exp["dx"]

    length_km = exp["length_unit_m"] / 1000.0
    time_s = exp["time_unit_s"]

    # Receiver x position in km
    receiver_x = receivers[:, 0] * dx * length_km

    # Receiver samples are taken AFTER each update:
    # sample 0 -> time dt
    times = (
        np.arange(1, exp["steps"] + 1)
        * dt
        * time_s
    )

    vmax = np.max(np.abs(traces))
    vmin = -vmax

    fig, axes = plt.subplots(
        1,
        traces.shape[0],
        figsize=(12, 5),
        sharey=True
    )

    for s, ax in enumerate(axes):

        im = ax.imshow(
            traces[s],
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            extent=[
                receiver_x[0],
                receiver_x[-1],
                times[-1],
                times[0],
            ],
            vmin=vmin,
            vmax=vmax,
            cmap="RdBu_r",
        )

        source_x = (
            shots[s, 0]
            * dx
            * length_km
        )

        ax.set_title(
            f"Shot {s}; source x = {source_x:.1f} km"
        )

        ax.set_xlabel(
            "Receiver position (km)"
        )

    axes[0].set_ylabel("Time (s)")

    cbar = fig.colorbar(
        im,
        ax=axes,
        shrink=0.85
    )

    cbar.set_label(
        "Pressure (arbitrary units)"
    )

    fig.suptitle(
        "Forward receiver shot gathers"
    )

    out = Path(
        "artifacts/forward/gathers.png"
    )

    fig.savefig(
        out,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", out)

    print()
    print("Verification values")

    print(
        "All traces L2 =",
        np.linalg.norm(traces)
    )

    for s in range(traces.shape[0]):

        a = np.abs(traces[s])

        step, receiver = np.unravel_index(
            np.argmax(a),
            a.shape
        )

        print(
            f"shot {s}: "
            f"max={a[step, receiver]:.8f}, "
            f"trace index={step}, "
            f"receiver index={receiver}"
        )


if __name__ == "__main__":
    main()
