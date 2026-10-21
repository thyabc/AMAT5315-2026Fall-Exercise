import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from part1_diffusion import (
    T, ALPHAS, ALPHA_BARS, MEANS, MEANS_NP, SIGMA,
    model_apply, nearest_cluster_metrics, sample_mixture_numpy,
    ROOT, EVIDENCE, ARTIFACTS, CHECKPOINTS
)

N = 2000
SEED = 20261022
NUM_STEPS = 1000


def load_params():
    path = CHECKPOINTS / "part1_stage_step_25000.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing trained checkpoint: {path}\n"
            "Complete Part 1 training first."
        )

    data = np.load(path)
    params = []
    i = 0

    while f"w{i}" in data:
        params.append((
            jnp.asarray(data[f"w{i}"]),
            jnp.asarray(data[f"b{i}"])
        ))
        i += 1

    if not params:
        raise ValueError("No model parameters found in checkpoint.")

    return params


def make_pairs(num_steps):
    levels = np.rint(
        np.linspace(T, 0, num_steps + 1)
    ).astype(np.int32)

    if len(np.unique(levels)) != len(levels):
        raise ValueError("Too many steps for distinct noise levels.")

    return jnp.asarray(
        np.stack([levels[:-1], levels[1:]], axis=1)
    )


@jax.jit
def exact_conditional_noise(x, t):
    # p(x_t | component k) is Gaussian with variance shared by
    # all four components. Posterior weights identify each component.
    ab = ALPHA_BARS[t - 1]
    variance = ab * (SIGMA ** 2) + (1.0 - ab)

    residual = (
        x[:, None, :]
        - jnp.sqrt(ab) * MEANS[None, :, :]
    )

    log_weights = (
        -0.5 * jnp.sum(residual ** 2, axis=-1) / variance
    )
    weights = jax.nn.softmax(log_weights, axis=-1)

    eps_per_component = (
        jnp.sqrt(1.0 - ab) / variance * residual
    )

    return jnp.sum(
        weights[:, :, None] * eps_per_component,
        axis=1
    )


@jax.jit
def ddim_learned(params, x_start, pairs):
    def step(x, pair):
        t, t_next = pair

        t_batch = jnp.full(
            (x.shape[0],), t, dtype=jnp.int32
        )
        eps = model_apply(params, x, t_batch)

        ab_t = ALPHA_BARS[t - 1]
        ab_next = jax.lax.cond(
            t_next > 0,
            lambda _: ALPHA_BARS[t_next - 1],
            lambda _: jnp.array(1.0, dtype=jnp.float32),
            operand=None
        )

        x0_hat = (
            x - jnp.sqrt(1.0 - ab_t) * eps
        ) / jnp.sqrt(ab_t)

        x_next = (
            jnp.sqrt(ab_next) * x0_hat
            + jnp.sqrt(1.0 - ab_next) * eps
        )

        return x_next, x_next[:10]

    final, path = jax.lax.scan(step, x_start, pairs)

    path = jnp.concatenate(
        [x_start[:10][None, :, :], path], axis=0
    )
    return final, path


@jax.jit
def ddim_exact(x_start, pairs):
    def step(x, pair):
        t, t_next = pair
        eps = exact_conditional_noise(x, t)

        ab_t = ALPHA_BARS[t - 1]
        ab_next = jax.lax.cond(
            t_next > 0,
            lambda _: ALPHA_BARS[t_next - 1],
            lambda _: jnp.array(1.0, dtype=jnp.float32),
            operand=None
        )

        x0_hat = (
            x - jnp.sqrt(1.0 - ab_t) * eps
        ) / jnp.sqrt(ab_t)

        x_next = (
            jnp.sqrt(ab_next) * x0_hat
            + jnp.sqrt(1.0 - ab_next) * eps
        )

        return x_next, x_next[:10]

    final, path = jax.lax.scan(step, x_start, pairs)

    path = jnp.concatenate(
        [x_start[:10][None, :, :], path], axis=0
    )
    return final, path


@jax.jit
def ddpm_stochastic(params, x_start, pairs, noise_seq):
    def step(x, inputs):
        pair, z = inputs
        t, t_next = pair

        t_batch = jnp.full(
            (x.shape[0],), t, dtype=jnp.int32
        )
        eps = model_apply(params, x, t_batch)

        ab_t = ALPHA_BARS[t - 1]
        ab_next = jax.lax.cond(
            t_next > 0,
            lambda _: ALPHA_BARS[t_next - 1],
            lambda _: jnp.array(1.0, dtype=jnp.float32),
            operand=None
        )

        a = ab_t / ab_next

        mean = (
            x - (1.0 - a)
            / jnp.sqrt(1.0 - ab_t) * eps
        ) / jnp.sqrt(a)

        x_next = jax.lax.cond(
            t_next > 0,
            lambda _: mean + jnp.sqrt(1.0 - a) * z,
            lambda _: mean,
            operand=None
        )

        return x_next, x_next[:10]

    final, path = jax.lax.scan(
        step, x_start, (pairs, noise_seq)
    )

    path = jnp.concatenate(
        [x_start[:10][None, :, :], path], axis=0
    )
    return final, path


def save_trajectory_plot(paths):
    names = [
        "DDIM learned",
        "DDIM exact conditional mean",
        "DDPM noise draw A",
        "DDPM noise draw B"
    ]

    fig, axes = plt.subplots(
        2, 2, figsize=(11, 10), constrained_layout=True
    )

    for ax, name, path in zip(axes.flat, names, paths):
        for i in range(path.shape[1]):
            ax.plot(
                path[:, i, 0], path[:, i, 1],
                linewidth=0.8, alpha=0.65
            )

        ax.scatter(
            MEANS_NP[:, 0], MEANS_NP[:, 1],
            marker="x", s=75, linewidth=2
        )
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        ax.set_title(name)

    fig.suptitle("Part 2: Reverse trajectories from shared initial noise")
    fig.savefig(EVIDENCE / "part2_trajectories.png", dpi=160)
    plt.close(fig)


def save_sample_plot(samples):
    names = [
        "Exact target",
        "DDIM learned",
        "DDIM exact predictor",
        "DDPM draw A",
        "DDPM draw B"
    ]

    fig, axes = plt.subplots(
        2, 3, figsize=(13, 8), constrained_layout=True
    )

    for ax, name, points in zip(axes.flat, names, samples):
        ax.scatter(points[:, 0], points[:, 1], s=5, alpha=0.45)
        ax.scatter(
            MEANS_NP[:, 0], MEANS_NP[:, 1],
            marker="x", s=65, linewidth=2
        )
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        ax.set_title(name)

    axes.flat[-1].axis("off")
    fig.suptitle("Part 2: Sample distribution comparison")
    fig.savefig(EVIDENCE / "part2_samples.png", dpi=160)
    plt.close(fig)


def get_metrics(points):
    populations, centers, spreads = nearest_cluster_metrics(points)

    return {
        "populations": populations.tolist(),
        "centers": centers.tolist(),
        "per_coordinate_spreads": spreads.tolist(),
        "max_population_error_from_0.25": float(
            np.max(np.abs(populations - 0.25))
        ),
        "cluster_population_check_pass": bool(
            np.all(np.abs(populations - 0.25) <= 0.015)
        ),
        "spread_check_pass": bool(
            np.all((spreads >= 0.29) & (spreads <= 0.32))
        )
    }


def main():
    start_time = time.time()
    print("Loading Part 1 trained model...", flush=True)
    params = load_params()
    pairs = make_pairs(NUM_STEPS)

    key = jax.random.PRNGKey(SEED)
    k_start, k_a, k_b, k_exact = jax.random.split(key, 4)

    x_start = jax.random.normal(k_start, (N, 2))
    noise_a = jax.random.normal(k_a, (NUM_STEPS, N, 2))
    noise_b = jax.random.normal(k_b, (NUM_STEPS, N, 2))

    print("Running learned DDIM twice from identical noise...", flush=True)
    ddim_a, path_ddim = ddim_learned(params, x_start, pairs)
    ddim_b, _ = ddim_learned(params, x_start, pairs)

    print("Running DDIM with exact conditional-mean noise...", flush=True)
    exact_ddim, path_exact = ddim_exact(x_start, pairs)

    print("Running DDPM with two independent reverse-noise sequences...", flush=True)
    ddpm_a, path_a = ddpm_stochastic(
        params, x_start, pairs, noise_a
    )
    ddpm_b, path_b = ddpm_stochastic(
        params, x_start, pairs, noise_b
    )

    # Wait for JAX computations to finish before recording results.
    outputs = [ddim_a, ddim_b, exact_ddim, ddpm_a, ddpm_b]
    outputs = [np.asarray(jax.block_until_ready(x)) for x in outputs]

    ddim_a_np, ddim_b_np, exact_ddim_np, ddpm_a_np, ddpm_b_np = outputs

    same_start_max_diff = float(
        np.max(np.abs(ddim_a_np - ddim_b_np))
    )

    rng = np.random.default_rng(SEED)
    exact_target = sample_mixture_numpy(N, rng)

    samples = [
        exact_target,
        ddim_a_np,
        exact_ddim_np,
        ddpm_a_np,
        ddpm_b_np
    ]

    names = [
        "exact_target",
        "ddim_learned",
        "ddim_exact_predictor",
        "ddpm_noise_a",
        "ddpm_noise_b"
    ]

    results = {
        "seed": SEED,
        "sample_count": N,
        "steps": NUM_STEPS,
        "ddim_max_difference_same_start": same_start_max_diff,
        "ddim_determinism_check_pass": same_start_max_diff == 0.0,
        "runs": {
            name: get_metrics(points)
            for name, points in zip(names, samples)
        }
    }

    labels_a = np.linalg.norm(
        ddpm_a_np[:, None, :] - MEANS_NP[None, :, :],
        axis=-1
    ).argmin(axis=1)
    labels_b = np.linalg.norm(
        ddpm_b_np[:, None, :] - MEANS_NP[None, :, :],
        axis=-1
    ).argmin(axis=1)

    results["ddpm_same_cluster_fraction_between_runs"] = float(
        np.mean(labels_a == labels_b)
    )

    (ARTIFACTS / "part2_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    save_trajectory_plot([
        np.asarray(path_ddim),
        np.asarray(path_exact),
        np.asarray(path_a),
        np.asarray(path_b)
    ])
    save_sample_plot(samples)

    np.savez_compressed(
        ARTIFACTS / "part2_results.npz",
        exact_target=exact_target,
        ddim_learned=ddim_a_np,
        ddim_exact_predictor=exact_ddim_np,
        ddpm_a=ddpm_a_np,
        ddpm_b=ddpm_b_np
    )

    print("\n===== Part 2 results =====")
    print("DDIM max difference from identical start:", same_start_max_diff)
    print("DDIM determinism check:", results["ddim_determinism_check_pass"])
    print(
        "DDPM same-cluster fraction between runs:",
        results["ddpm_same_cluster_fraction_between_runs"]
    )

    for name in names:
        r = results["runs"][name]
        print(f"\n{name}:")
        print("  populations:", np.round(r["populations"], 4))
        print("  spreads:\n", np.round(r["per_coordinate_spreads"], 4))
        print("  population check:", r["cluster_population_check_pass"])
        print("  spread check:", r["spread_check_pass"])

    print(f"\nFigures saved in: {EVIDENCE}")
    print(f"Metrics saved in: {ARTIFACTS / 'part2_metrics.json'}")
    print(f"Elapsed: {(time.time() - start_time) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
