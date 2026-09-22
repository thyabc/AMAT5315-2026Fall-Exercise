from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/convergence"

JSON_OUT = ROOT / "evidence/convergence.json"
PNG_OUT = ROOT / "evidence/convergence.png"


def final_frame(folder):
    path = ART / folder / "fields.jsonl"

    with open(path) as f:
        frames = [
            json.loads(line)
            for line in f
        ]

    return frames[-1]


def omega(frame):
    return np.asarray(
        frame["omega"],
        dtype=float,
    )


def relative_error(a, b):
    return (
        np.linalg.norm(a - b)
        / np.linalg.norm(b)
    )


# ------------------------------------------------------------
# Fine reference
# ------------------------------------------------------------
reference = final_frame("reference")
w_ref = omega(reference)

runs = [
    (0.02, "dt0.02"),
    (0.0125, "dt0.0125"),
    (0.01, "dt0.01"),
]

dts = []
errors = []
fields = {}

for dt, folder in runs:

    frame = final_frame(folder)

    assert abs(frame["t"] - 2.0) < 1e-12

    w = omega(frame)

    fields[dt] = w

    err = relative_error(
        w,
        w_ref,
    )

    dts.append(dt)
    errors.append(err)

    print(
        f"dt={dt:<7} "
        f"error={err:.8e}"
    )


dts = np.asarray(dts)
errors = np.asarray(errors)

# ------------------------------------------------------------
# Fit temporal convergence order
# ------------------------------------------------------------
slope, intercept = np.polyfit(
    np.log(dts),
    np.log(errors),
    1,
)

print()
print("fitted slope =", slope)


# ------------------------------------------------------------
# Richardson estimate
#
# h = 0.01
# 2h = 0.02
#
# e_h ~= ||w_2h - w_h||
#         -------------------
#         (2^4 - 1) ||w_h||
# ------------------------------------------------------------
h = 0.01

w_h = fields[0.01]
w_2h = fields[0.02]

richardson_h = (
    np.linalg.norm(w_2h - w_h)
    /
    (
        (2**4 - 1)
        * np.linalg.norm(w_h)
    )
)

candidate = 0.0125

predicted_candidate = (
    richardson_h
    * (candidate / h)**4
)

measured = {
    dt: err
    for dt, err
    in zip(dts, errors)
}

threshold = 5.0e-6

valid = [
    dt
    for dt in dts
    if (
        richardson_h
        * (dt / h)**4
    ) < threshold
]

chosen = max(valid)

chosen_predicted = (
    richardson_h
    * (chosen / h)**4
)

chosen_measured = measured[chosen]

print()
print(
    "Richardson error at dt=0.01 =",
    richardson_h,
)

print(
    "predicted error at dt=0.0125 =",
    predicted_candidate,
)

print()
print("chosen dt =", chosen)
print(
    "chosen predicted error =",
    chosen_predicted,
)
print(
    "chosen measured error  =",
    chosen_measured,
)


# ------------------------------------------------------------
# Save convergence.json
# ------------------------------------------------------------
result = {
    "reference_dt": 0.0025,

    "runs": [
        {
            "dt": float(dt),
            "relative_omega_error": float(err),
        }
        for dt, err in zip(dts, errors)
    ],

    "slope": float(slope),

    "richardson": {
        "order": 4,
        "base_dt": 0.01,
        "estimated_error_at_base_dt":
            float(richardson_h),
        "predicted_error_at_0.0125":
            float(predicted_candidate),
    },

    "selection": {
        "threshold": threshold,
        "dt": float(chosen),
        "predicted_error":
            float(chosen_predicted),
        "measured_error":
            float(chosen_measured),
    },
}

JSON_OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(JSON_OUT, "w") as f:
    json.dump(
        result,
        f,
        indent=2,
    )


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
order = np.argsort(dts)

x = dts[order]
y = errors[order]

fig, ax = plt.subplots(
    figsize=(7.2, 5.1),
    constrained_layout=True,
)

ax.loglog(
    x,
    y,
    marker="o",
    linewidth=2,
    label=f"measured, slope {slope:.3f}",
)

# fourth-order guide
guide = (
    y[0]
    * (x / x[0])**4
    * 3.0
)

ax.loglog(
    x,
    guide,
    linestyle="--",
    label="slope 4 guide",
)

# threshold
ax.axhline(
    threshold,
    linestyle=":",
    label="5e-6 target",
)

# mark chosen step
ax.scatter(
    [chosen],
    [chosen_measured],
    s=90,
    marker="*",
    zorder=5,
    label=(
        f"chosen dt={chosen}"
    ),
)

ax.set_xlabel("time step dt")

ax.set_ylabel(
    "relative omega error at t = 2"
)

ax.set_title(
    "Random-flow time-step refinement"
)

ax.grid(
    True,
    which="both",
    alpha=0.25,
)

ax.legend()

fig.savefig(
    PNG_OUT,
    dpi=180,
)

plt.close(fig)

print()
print(f"Wrote {JSON_OUT}")
print(f"Wrote {PNG_OUT}")
