from pathlib import Path
import json

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

STEPS = [0, 1, 10, 100, 1000]
N_CHAINS = 3
RANDOM_SEED = 2026
GIBBS_SEED = 2027


def load_params(k):
    path = ARTIFACTS / f"part2_rbm_cd{k}_params.npz"

    with np.load(path) as data:
        params = {
            name: jnp.asarray(data[name])
            for name in ["W", "a", "b"]
        }

    assert params["W"].shape == (784, 128)
    assert params["a"].shape == (784,)
    assert params["b"].shape == (128,)

    for value in params.values():
        assert bool(jnp.all(jnp.isfinite(value)))

    print(f"Loaded CD-{k} model: {path.name}")
    return params


def gibbs_step(params, v, key):
    key_h, key_v = jax.random.split(key)

    p_h = jax.nn.sigmoid(
        v @ params["W"] + params["b"]
    )

    h = jax.random.bernoulli(
        key_h, p_h
    ).astype(jnp.float32)

    p_v = jax.nn.sigmoid(
        h @ params["W"].T + params["a"]
    )

    return jax.random.bernoulli(
        key_v, p_v
    ).astype(jnp.float32)


@jax.jit
def sample_trajectory(params, starts, key):
    """Save states after 0, 1, 10, 100, 1000 Gibbs steps."""

    def body(carry, _):
        v, rng = carry
        rng, subkey = jax.random.split(rng)
        v = gibbs_step(params, v, subkey)
        return (v, rng), None

    v = starts
    outputs = [v]

    # Incremental lengths reach the requested total steps.
    for length in (1, 9, 90, 900):
        (v, key), _ = jax.lax.scan(
            body, (v, key), None, length=length
        )
        outputs.append(v)

    return jnp.stack(outputs, axis=0)


def plot_generation(results):
    fig, axes = plt.subplots(
        2 * N_CHAINS, len(STEPS),
        figsize=(12, 10)
    )

    models = [1, 10]

    for model_index, k in enumerate(models):
        data = results[k]

        for chain in range(N_CHAINS):
            row = model_index * N_CHAINS + chain

            for col, step in enumerate(STEPS):
                ax = axes[row, col]

                image = data[col, chain].reshape(28, 28)

                ax.imshow(
                    image,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest"
                )

                ax.set_xticks([])
                ax.set_yticks([])

                if row == 0:
                    ax.set_title(
                        "Random start" if step == 0
                        else f"{step} Gibbs steps"
                    )

                if col == 0:
                    ax.set_ylabel(
                        f"CD-{k}\nChain {chain+1}",
                        fontsize=10
                    )

    fig.suptitle(
        "Fresh generation from shared random starts",
        fontsize=15
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output = EVIDENCE / "part2_fresh_generation.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output)


def plot_reconstruction(test_x, digit_labels, params):
    reconstructed = {}

    for k in (1, 10):
        key = jax.random.PRNGKey(3100)
        reconstructed[k] = np.asarray(
            gibbs_step(params[k], jnp.asarray(test_x), key)
        )

    fig, axes = plt.subplots(
        3, 10, figsize=(15, 5)
    )

    rows = [
        ("Held-out originals", test_x),
        ("CD-1 reconstruction", reconstructed[1]),
        ("CD-10 reconstruction", reconstructed[10])
    ]

    for row, (description, images) in enumerate(rows):
        for col in range(10):
            ax = axes[row, col]

            ax.imshow(
                images[col].reshape(28, 28),
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest"
            )

            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(
                    f"Digit {digit_labels[col]}",
                    fontsize=10
                )

            if col == 0:
                ax.set_ylabel(description, fontsize=9)

    fig.suptitle(
        "One-step reconstruction of held-out digits",
        fontsize=15
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output = EVIDENCE / "part2_reconstruction.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output)

    for k in (1, 10):
        error = np.mean(
            np.abs(reconstructed[k] - test_x)
        )
        print(f"Displayed CD-{k} reconstruction error: {error:.4f}")


def main():
    params = {
        1: load_params(1),
        10: load_params(10)
    }

    # Choose one held-out test image per digit class.
    test_images, test_labels = load_mnist("test")

    indices = [
        np.flatnonzero(test_labels == digit)[0]
        for digit in range(10)
    ]

    test_x = (
        (test_images[indices] >= 128)
        .astype(np.float32)
        .reshape(10, 784)
    )

    print("\nGenerating one-step reconstructions...")
    plot_reconstruction(test_x, np.arange(10), params)

    # Each model receives the same initial random images.
    rng = np.random.default_rng(RANDOM_SEED)
    starts = rng.integers(
        0, 2, size=(N_CHAINS, 784)
    ).astype(np.float32)

    results = {}

    for k in (1, 10):
        print(f"\nSampling CD-{k} model...")

        key = jax.random.PRNGKey(GIBBS_SEED)

        results[k] = np.asarray(
            sample_trajectory(
                params[k],
                jnp.asarray(starts),
                key
            )
        )

        assert results[k].shape == (
            len(STEPS), N_CHAINS, 784
        )

        np.testing.assert_array_equal(
            results[k][0], starts
        )

        print(f"CD-{k}: 1000 Gibbs steps completed.")

    plot_generation(results)

    np.savez_compressed(
        ARTIFACTS / "part2_generation_samples.npz",
        steps=np.array(STEPS),
        starts=starts,
        cd1=results[1],
        cd10=results[10]
    )

    settings = {
        "sampling_steps": STEPS,
        "number_of_chains": N_CHAINS,
        "random_start_seed": RANDOM_SEED,
        "gibbs_seed": GIBBS_SEED,
        "training_models": ["CD-1", "CD-10"],
        "model_parameters_frozen": True
    }

    with open(
        ARTIFACTS / "part2_generation_settings.json",
        "w", encoding="utf-8"
    ) as file:
        json.dump(settings, file, indent=2)

    print("\nPart 2 generation verification PASSED.")
    print("Both RBMs used identical random starts.")
    print("Model parameters were kept fixed.")


if __name__ == "__main__":
    main()
