import json
from itertools import zip_longest
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


BUDGETS = [1, 3, 5, 10]
NSTEPS = 240


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


full_image = np.load(
    "artifacts/adjoint/image.npy"
)

full_result = load_json(
    "artifacts/adjoint/result.json"
)

full_stats = full_result["statistics"]

forward_calls = []
peak_bytes = []

print("Checkpoint verification")
print("=" * 70)

for budget in BUDGETS:
    root = Path(
        f"artifacts/checkpoint-{budget}"
    )

    image = np.load(
        root / "image.npy"
    )

    result = load_json(
        root / "result.json"
    )

    stats = result["statistics"]

    rel = (
        np.linalg.norm(
            image - full_image
        )
        /
        np.linalg.norm(
            full_image
        )
    )

    peak = stats[
        "peak_saved_states"
    ]

    per_shot_forward = stats[
        "per_shot"
    ][0][
        "scheduler_forward_calls"
    ]

    forward_calls.append(
        per_shot_forward
    )

    peak_bytes.append(
        stats["peak_saved_bytes"]
    )

    grad_errors = 0
    invalid_restores = 0
    budget_overruns = 0
    saved_count_errors = 0

    for shot in range(3):
        actions = load_json(
            root /
            f"actions-{shot}.json"
        )

        grads = [
            a["step"]
            for a in actions
            if a["action"] == "grad"
        ]

        expected = list(
            range(
                NSTEPS - 1,
                -1,
                -1
            )
        )

        grad_errors += sum(
            a != b
            for a, b in zip_longest(
                grads,
                expected
            )
        )

        saved = {0}

        for action in actions:
            kind = action["action"]
            step = action["step"]

            if kind == "restore":
                if step not in saved:
                    invalid_restores += 1

            elif kind == "store":
                saved.add(step)

            elif kind == "fetch":
                if (
                    step == 0
                    or step not in saved
                ):
                    invalid_restores += 1
                else:
                    saved.remove(step)

            if len(saved) > budget + 1:
                budget_overruns += 1

            if (
                action["saved_states"]
                != len(saved)
            ):
                saved_count_errors += 1

    print(
        f"budget {budget:2d}: "
        f"relative L2 error = {rel:.3e}, "
        f"peak states = {peak}, "
        f"forward/shot = {per_shot_forward}"
    )

    print(
        f"           grad errors = {grad_errors}, "
        f"invalid restores = {invalid_restores}, "
        f"budget overruns = {budget_overruns}, "
        f"count errors = {saved_count_errors}"
    )


# ==========================================================
# Plot 1: budget-5 action schedule
# ==========================================================

actions = load_json(
    "artifacts/checkpoint-5/actions-0.json"
)

fig, ax = plt.subplots(
    figsize=(11, 6)
)

kinds = [
    "store",
    "restore",
    "call",
    "grad",
    "fetch",
]

for kind in kinds:
    xs = [
        i
        for i, a in enumerate(actions)
        if a["action"] == kind
    ]

    ys = [
        a["step"]
        for a in actions
        if a["action"] == kind
    ]

    ax.scatter(
        xs,
        ys,
        s=10,
        label=kind.capitalize()
    )

ax.set_xlabel(
    "Operation index"
)

ax.set_ylabel(
    "Time step"
)

ax.set_title(
    "Treeverse schedule; reflector shot 0, budget 5"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    "artifacts/checkpoint-actions.png",
    dpi=200
)

plt.close(fig)


# ==========================================================
# Plot 2: storage / recomputation
# ==========================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

axes[0].plot(
    BUDGETS,
    forward_calls,
    marker="o",
    label="Treeverse"
)

axes[0].axhline(
    full_stats[
        "scheduler_forward_calls"
    ] // 3,
    linestyle="--",
    label="Full history"
)

axes[0].set_yscale("log")

axes[0].set_xlabel(
    "Additional checkpoint slots"
)

axes[0].set_ylabel(
    "Forward steps per shot"
)

axes[0].set_title(
    "Recomputation cost"
)

axes[0].legend()


axes[1].plot(
    BUDGETS,
    peak_bytes,
    marker="o",
    label="Treeverse"
)

axes[1].axhline(
    full_stats[
        "peak_saved_bytes"
    ],
    linestyle="--",
    label="Full history"
)

axes[1].set_xlabel(
    "Additional checkpoint slots"
)

axes[1].set_ylabel(
    "Peak saved-state bytes"
)

axes[1].set_title(
    "Saved-state storage"
)

axes[1].legend()

fig.tight_layout()

fig.savefig(
    "artifacts/checkpoint-work.png",
    dpi=200
)

plt.close(fig)

print()
print(
    "Saved artifacts/checkpoint-actions.png"
)
print(
    "Saved artifacts/checkpoint-work.png"
)
