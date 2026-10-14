from pathlib import Path
import argparse
import json
import time

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from part1_datasets import load_mnist

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
EVIDENCE = ROOT / "evidence"
EVIDENCE.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)

STEPS = [1, 10, 100, 1000]
HALF = 14 * 28
SEED = 2026


def load_model(k):
    path = ARTIFACTS / f"part2_rbm_cd{k}_params.npz"
    with np.load(path) as file:
        params = {
            name: jnp.asarray(file[name])
            for name in ("W", "a", "b")
        }
    assert params["W"].shape == (784, 128)
    return params


def make_wrong_indices(labels):
    """Choose a different digit class for each wrong upper half."""
    n = len(labels)
    indices = []
    for i in range(n):
        j = (i + 1) % n
        while labels[j] == labels[i]:
            j = (j + 1) % n
            if j == i:
                raise ValueError("Need at least two digit classes")
        indices.append(j)
    return np.asarray(indices)


@jax.jit
def sample_checkpoints(params, starts, observed_top, key):
    """Clamp observed pixels after every Gibbs sweep."""
    def one_step(carry, _):
        v, rng = carry
        rng, subkey = jax.random.split(rng)
        key_h, key_v = jax.random.split(subkey)

        ph = jax.nn.sigmoid(
            v @ params["W"] + params["b"]
        )
        h = jax.random.bernoulli(
            key_h, ph
        ).astype(jnp.float32)

        pv = jax.nn.sigmoid(
            h @ params["W"].T + params["a"]
        )
        new_v = jax.random.bernoulli(
            key_v, pv
        ).astype(jnp.float32)

        new_v = new_v.at[:, :HALF].set(observed_top)
        return (new_v, rng), None

    v = starts
    outputs = []

    # Cumulative steps: 1, 10, 100, 1000
    for length in (1, 9, 90, 900):
        (v, key), _ = jax.lax.scan(
            one_step,
            (v, key),
            None,
            length=length
        )
        outputs.append(v)

    return jnp.stack(outputs)


def calculate_overlap(states, true_bottom):
    """
    states: [steps, digits, chains, 784]
    returns mean overlap and SEM across independent digits.
    """
    generated = states[..., HALF:] > 0.5
    truth = true_bottom[None, :, None, :] > 0.5

    intersection = np.sum(generated & truth, axis=-1)
    count_a = np.sum(generated, axis=-1)
    count_b = np.sum(truth, axis=-1)

    overlap = (
        2.0 * intersection
        / np.maximum(count_a + count_b, 1)
    )

    per_digit = overlap.mean(axis=2)
    means = per_digit.mean(axis=1)

    sems = (
        per_digit.std(axis=1, ddof=1)
        / np.sqrt(per_digit.shape[1])
    )

    return means, sems


def plot_gallery(results, images, labels, suffix):
    n_show = min(4, len(images))
    n_extra = min(3, results[1].shape[3] - 1)
    n_cols = 6 + n_extra

    fig, axes = plt.subplots(
        n_show * 2, n_cols,
        figsize=(1.6 * n_cols, 2.1 * n_show * 2),
        squeeze=False
    )

    titles = [
        "Truth", "Observed",
        "1", "10", "100", "1000"
    ] + [f"Chain {c+2}\n1000" for c in range(n_extra)]

    for i in range(n_show):
        for model_row, k in enumerate((1, 10)):
            row = 2 * i + model_row
            axes_row = axes[row]

            # Complete held-out image.
            gallery = [images[i].reshape(28, 28)]

            # Upper half observed, lower half deliberately hidden.
            observed = images[i].copy()
            observed[HALF:] = 0.35
            gallery.append(observed.reshape(28, 28))

            # First chain at each checkpoint.
            for t in range(4):
                gallery.append(
                    results[k][t, 0, i, 0].reshape(28, 28)
                )

            # Additional chains at 1000 steps.
            for chain in range(1, n_extra + 1):
                gallery.append(
                    results[k][3, 0, i, chain].reshape(28, 28)
                )

            for col, picture in enumerate(gallery):
                ax = axes_row[col]
                ax.imshow(
                    picture,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest"
                )
                ax.set_xticks([])
                ax.set_yticks([])

                if row == 0:
                    ax.set_title(titles[col], fontsize=9)

                if col == 0:
                    ax.set_ylabel(
                        f"Digit {int(labels[i])}\nCD-{k}",
                        fontsize=9
                    )

    fig.suptitle(
        "Clamped Gibbs completion: upper half fixed",
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    path = EVIDENCE / f"part2_completion_gallery{suffix}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path, flush=True)


def plot_overlap(metrics, suffix):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for k in (1, 10):
        for condition in ("true", "wrong"):
            values = metrics[str(k)][condition]
            ax.errorbar(
                STEPS,
                values["mean"],
                yerr=values["sem"],
                marker="o",
                linestyle="-" if condition == "true" else "--",
                capsize=3,
                label=f"CD-{k}, {condition} upper half"
            )

    ax.set_xscale("log")
    ax.set_xticks(STEPS)
    ax.set_xticklabels([str(x) for x in STEPS])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Clamped Gibbs sweeps")
    ax.set_ylabel("Ink overlap with true lower half")
    ax.set_title("Image completion: true vs wrong observation")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = EVIDENCE / f"part2_completion_overlap{suffix}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    n_digits = 8 if args.preview else 64
    n_chains = 4 if args.preview else 8
    suffix = "_preview" if args.preview else ""

    images_raw, labels_raw = load_mnist("test")
    images = (
        (images_raw[:n_digits] >= 128)
        .astype(np.float32)
        .reshape(n_digits, 784)
    )
    labels = labels_raw[:n_digits]

    wrong_indices = make_wrong_indices(labels)

    true_top = images[:, :HALF]
    wrong_top = true_top[wrong_indices]
    true_bottom = images[:, HALF:]

    rng = np.random.default_rng(SEED)
    random_bottom = rng.integers(
        0, 2,
        size=(n_digits, n_chains, HALF)
    ).astype(np.float32)

    true_top_chains = np.broadcast_to(
        true_top[:, None, :],
        (n_digits, n_chains, HALF)
    )
    wrong_top_chains = np.broadcast_to(
        wrong_top[:, None, :],
        (n_digits, n_chains, HALF)
    )

    starts_true = np.concatenate(
        (true_top_chains, random_bottom), axis=-1
    )
    starts_wrong = np.concatenate(
        (wrong_top_chains, random_bottom), axis=-1
    )

    # First condition is true; second is wrong.
    starts = np.stack(
        (starts_true, starts_wrong), axis=0
    )

    fixed_top = np.stack(
        (true_top_chains, wrong_top_chains), axis=0
    )

    flat_starts = jnp.asarray(starts.reshape(-1, 784))
    flat_top = jnp.asarray(fixed_top.reshape(-1, HALF))

    metrics = {}
    results = {}

    print("Number of digits:", n_digits, flush=True)
    print("Chains per digit:", n_chains, flush=True)
    print("Sampling steps:", STEPS, flush=True)
    print("Conditions: true upper / wrong upper", flush=True)

    for k in (1, 10):
        print(f"\nRunning CD-{k} completion...", flush=True)
        started = time.perf_counter()

        params = load_model(k)

        samples = np.asarray(
            sample_checkpoints(
                params,
                flat_starts,
                flat_top,
                jax.random.PRNGKey(SEED + 100)
            )
        )

        samples = samples.reshape(
            4, 2, n_digits, n_chains, 784
        )

        # Verify that the observed part never changed.
        expected_top = fixed_top[None, ...]
        np.testing.assert_array_equal(
            samples[..., :HALF],
            np.broadcast_to(
                expected_top,
                samples[..., :HALF].shape
            )
        )

        results[k] = samples.astype(np.uint8)
        metrics[str(k)] = {}

        for condition_index, condition in enumerate(
            ("true", "wrong")
        ):
            mean, sem = calculate_overlap(
                samples[:, condition_index],
                true_bottom
            )
            metrics[str(k)][condition] = {
                "mean": mean.tolist(),
                "sem": sem.tolist()
            }

        elapsed = time.perf_counter() - started

        true_final = metrics[str(k)]["true"]["mean"][-1]
        wrong_final = metrics[str(k)]["wrong"]["mean"][-1]
        difference = true_final - wrong_final

        print(f"CD-{k} runtime: {elapsed:.1f} s", flush=True)
        print(f"True overlap at 1000:  {true_final:.4f}", flush=True)
        print(f"Wrong overlap at 1000: {wrong_final:.4f}", flush=True)
        print(f"Difference:             {difference:.4f}", flush=True)

        if not args.preview:
            print(
                "Criterion (difference >= 0.1):",
                "PASS" if difference >= 0.1 else "NOT MET",
                flush=True
            )

    plot_gallery(results, images, labels, suffix)
    plot_overlap(metrics, suffix)

    summary = {
        "n_digits": n_digits,
        "n_chains": n_chains,
        "steps": STEPS,
        "seed": SEED,
        "wrong_upper_indices": wrong_indices.tolist(),
        "metrics": metrics
    }

    with open(
        ARTIFACTS / f"part2_completion_summary{suffix}.json",
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    np.savez_compressed(
        ARTIFACTS / f"part2_completion_samples{suffix}.npz",
        cd1=results[1],
        cd10=results[10],
        truth=images,
        labels=labels,
        steps=np.array(STEPS)
    )

    print("\nCompletion experiment finished.", flush=True)
    print("Observed upper halves remained fixed.", flush=True)


if __name__ == "__main__":
    main()
