import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

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
    T, MEANS, MEANS_NP, SIGMA, BATCH_SIZE, BASE_LR,
    HIDDEN, UPDATES, init_params, model_apply,
    sample_mixture_numpy, nearest_cluster_metrics,
    ROOT, EVIDENCE, ARTIFACTS, CHECKPOINTS
)
from part2_ddim import (
    load_params, make_pairs, ddim_learned, ddpm_stochastic
)

SEED = 20261023
FLOW_UPDATES = 100000
FLOW_BATCH = 512
FLOW_LR = 2e-3
SAMPLE_COUNT = 4000
STEP_COUNTS = [1, 2, 5, 10, 20, 50, 100, 1000]
FLOW_STEPS = [1, 2, 5, 10, 20, 50, 100]

for folder in [EVIDENCE, ARTIFACTS, CHECKPOINTS]:
    folder.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 1. Exact marginal velocity of the four-component Gaussian mixture
# ------------------------------------------------------------

@jax.jit
def exact_velocity(x, s):
    """
    v(x,s) = E[z - x0 | xs=x]
    xs = (1-s)*x0 + s*z

    Component k:
      mean(xs|k) = (1-s)*mu_k
      var(xs|k)  = (1-s)^2*SIGMA^2 + s^2
    """
    s = jnp.asarray(s, dtype=jnp.float32).reshape(-1)

    q = (1.0 - s) ** 2 * SIGMA**2 + s**2
    component_mean = (
        (1.0 - s)[:, None, None] * MEANS[None, :, :]
    )
    residual = x[:, None, :] - component_mean

    logits = -0.5 * jnp.sum(residual**2, axis=-1) / q[:, None]
    weights = jax.nn.softmax(logits, axis=-1)

    coefficient = (
        s - (1.0 - s) * SIGMA**2
    ) / q

    component_velocity = (
        -MEANS[None, :, :]
        + coefficient[:, None, None] * residual
    )

    return jnp.sum(
        weights[:, :, None] * component_velocity, axis=1
    )


# ------------------------------------------------------------
# 2. Train a fresh velocity network with flow matching
# ------------------------------------------------------------

@jax.jit
def flow_train_step(params, m, v, step, key):
    k_component, k_data, k_noise, k_time = jax.random.split(key, 4)

    labels = jax.random.randint(
        k_component, (FLOW_BATCH,), 0, 4
    )

    x0 = (
        MEANS[labels]
        + SIGMA * jax.random.normal(k_data, (FLOW_BATCH, 2))
    )

    z = jax.random.normal(k_noise, (FLOW_BATCH, 2))
    s = jax.random.uniform(
        k_time, (FLOW_BATCH,), minval=0.0, maxval=1.0
    )

    xs = (1.0 - s[:, None]) * x0 + s[:, None] * z
    target = z - x0

    def loss_fn(p):
        prediction = model_apply(p, xs, s * T)
        return jnp.mean(
            0.5 * jnp.sum((prediction - target) ** 2, axis=-1)
        )

    loss, grads = jax.value_and_grad(loss_fn)(params)

    b1, b2 = 0.9, 0.999
    m = jax.tree_util.tree_map(
        lambda old, g: b1 * old + (1.0 - b1) * g,
        m, grads
    )
    v = jax.tree_util.tree_map(
        lambda old, g: b2 * old + (1.0 - b2) * (g * g),
        v, grads
    )

    step1 = step + 1.0
    mhat = jax.tree_util.tree_map(
        lambda a: a / (1.0 - b1**step1), m
    )
    vhat = jax.tree_util.tree_map(
        lambda a: a / (1.0 - b2**step1), v
    )

    lr = FLOW_LR * jnp.maximum(
        0.0, 1.0 - step / (FLOW_UPDATES - 1)
    )

    params = jax.tree_util.tree_map(
        lambda p, mm, vv: p - lr * mm / (jnp.sqrt(vv) + 1e-8),
        params, mhat, vhat
    )

    return params, m, v, loss


def save_params(params, step):
    data = {}
    for i, (w, b) in enumerate(params):
        data[f"w{i}"] = np.asarray(w)
        data[f"b{i}"] = np.asarray(b)

    path = CHECKPOINTS / f"part3_velocity_step_{step:05d}.npz"
    np.savez_compressed(path, **data)


def train_velocity():
    print("\n========== Part 3A: Train velocity network ==========",
          flush=True)

    key = jax.random.PRNGKey(SEED)
    params = init_params(key)
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)

    stage_steps = {10, 100, 500, 1000, 2500, 5000, 10000, 17500, 25000}
    save_params(params, 0)
    losses = []

    for i in range(FLOW_UPDATES):
        step_key = jax.random.fold_in(key, i + 1)

        params, m, v, loss = flow_train_step(
            params, m, v,
            jnp.asarray(float(i), dtype=jnp.float32),
            step_key
        )

        losses.append(float(loss))
        current = i + 1

        if current in stage_steps:
            save_params(params, current)

        if current % 1000 == 0:
            print(
                f"  update {current}/{FLOW_UPDATES}"
                f" | loss={losses[-1]:.6f}",
                flush=True
            )

    save_params(params, FLOW_UPDATES)

    np.savetxt(
        ARTIFACTS / "part3_flow_loss_history.csv",
        np.asarray(losses),
        delimiter=",",
        header="flow_matching_loss",
        comments=""
    )

    with open(ARTIFACTS / "part3_flow_settings.json", "w",
              encoding="utf-8") as f:
        json.dump({
            "seed": SEED,
            "training_updates": FLOW_UPDATES,
            "batch_size": FLOW_BATCH,
            "learning_rate": FLOW_LR,
            "learning_rate_schedule": "linear decay to zero",
            "architecture": "same as Part 1 noise predictor",
            "initialization": "fresh random initialization",
            "path": "x_s=(1-s)*x0+s*z",
            "target_velocity": "z-x0",
            "final_loss": losses[-1]
        }, f, indent=2)

    return params, losses


# ------------------------------------------------------------
# 3. Validate the learned velocity field
# ------------------------------------------------------------

def validate_velocity(params):
    print("\n========== Part 3B: Velocity field validation ==========",
          flush=True)

    axis = np.linspace(-4.5, 4.5, 25, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)

    grid_j = jnp.asarray(grid)
    records = {}
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5), constrained_layout=True
    )

    for ax, s in zip(axes, [0.1, 0.5, 0.9]):
        s_batch = jnp.full((len(grid),), s, dtype=jnp.float32)

        pred = np.asarray(
            model_apply(params, grid_j, s_batch * T)
        )
        exact = np.asarray(exact_velocity(grid_j, s_batch))

        diff_rms = float(
            np.sqrt(np.mean(np.sum((pred - exact)**2, axis=1)))
        )
        exact_rms = float(
            np.sqrt(np.mean(np.sum(exact**2, axis=1)))
        )
        relative = diff_rms / max(exact_rms, 1e-12)

        records[str(s)] = {
            "difference_rms": diff_rms,
            "exact_velocity_rms": exact_rms,
            "relative_rms_error": relative,
            "pass_below_0.1": bool(relative < 0.1)
        }

        print(
            f"s={s:.1f} | relative RMS error={relative:.5f}"
            f" | {'PASS' if relative < 0.1 else 'CHECK'}",
            flush=True
        )

        # Clip long arrows for readability.
        exact_norm = np.linalg.norm(exact, axis=1, keepdims=True)
        pred_norm = np.linalg.norm(pred, axis=1, keepdims=True)
        exact_plot = exact * np.minimum(1.0, 3.5 / (exact_norm + 1e-8))
        pred_plot = pred * np.minimum(1.0, 3.5 / (pred_norm + 1e-8))

        ax.quiver(
            grid[:, 0], grid[:, 1],
            exact_plot[:, 0], exact_plot[:, 1],
            color="black", alpha=0.55, angles="xy",
            scale_units="xy", scale=3.5
        )
        ax.quiver(
            grid[:, 0], grid[:, 1],
            pred_plot[:, 0], pred_plot[:, 1],
            color="red", alpha=0.55, angles="xy",
            scale_units="xy", scale=3.5
        )
        ax.scatter(
            MEANS_NP[:, 0], MEANS_NP[:, 1],
            marker="x", s=60
        )
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"s={s:.1f}, relative RMS={relative:.3f}")
        ax.grid(alpha=0.2)

    fig.suptitle("Exact (black) vs learned (red) marginal velocity")
    fig.savefig(EVIDENCE / "part3_velocity_fields.png", dpi=160)
    plt.close(fig)

    with open(ARTIFACTS / "part3_velocity_metrics.json", "w",
              encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    return records


# ------------------------------------------------------------
# 4. Euler sampling from s=1 to s=0
# ------------------------------------------------------------

@jax.jit
def flow_euler_sample(params, x_start, levels):
    def step(x, inputs):
        s, ds = inputs
        s_batch = jnp.full(
            (x.shape[0],), s * T, dtype=jnp.float32
        )
        velocity = model_apply(params, x, s_batch)
        x_next = x - ds * velocity
        return x_next, None

    s_values = levels[:-1]
    ds_values = levels[:-1] - levels[1:]
    final, _ = jax.lax.scan(
        step, x_start, (s_values, ds_values)
    )
    return final


# ------------------------------------------------------------
# 5. Energy distance, using exact pairwise averages in chunks
# ------------------------------------------------------------

def mean_pair_distance(a, b, exclude_diagonal=False, chunk=256):
    """
    Average Euclidean distance over all pairs.
    Chunked matrix multiplication avoids allocating an N x N x 2 array.
    For a=b, the zero diagonal is excluded from the denominator.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    b_sq = np.sum(b * b, axis=1)[None, :]
    total = 0.0

    for start in range(0, len(a), chunk):
        aa = a[start:start + chunk]
        a_sq = np.sum(aa * aa, axis=1)[:, None]
        d2 = a_sq + b_sq - 2.0 * (aa @ b.T)
        np.maximum(d2, 0.0, out=d2)
        np.sqrt(d2, out=d2)
        total += float(np.sum(d2, dtype=np.float64))

    if exclude_diagonal:
        n = len(a)
        return total / (n * (n - 1))

    return total / (len(a) * len(b))


def energy_distance(a, b, self_a=None, self_b=None):
    if self_a is None:
        self_a = mean_pair_distance(a, a, exclude_diagonal=True)
    if self_b is None:
        self_b = mean_pair_distance(b, b, exclude_diagonal=True)

    cross = mean_pair_distance(a, b)
    return float(2.0 * cross - self_a - self_b)


def evaluate_samples(samples, exact, exact_self):
    populations, centers, spreads = nearest_cluster_metrics(samples)

    self_samples = mean_pair_distance(
        samples, samples, exclude_diagonal=True
    )
    cross = mean_pair_distance(samples, exact)
    ed = float(2.0 * cross - self_samples - exact_self)

    return {
        "energy_distance": ed,
        "cluster_fractions": populations.tolist(),
        "centers": centers.tolist(),
        "per_coordinate_widths": spreads.tolist(),
        "max_cluster_fraction_error": float(
            np.max(np.abs(populations - 0.25))
        ),
        "max_width_error_from_0.30": float(
            np.nanmax(np.abs(spreads - 0.30))
        ),
        "all_widths_within_0.03": bool(
            np.all(np.abs(spreads - 0.30) <= 0.03)
        )
    }


def estimate_null_scale(rng):
    print("\n========== Part 3C: Estimate energy-distance null scale ==========",
          flush=True)

    distances = []

    for i in range(20):
        a = sample_mixture_numpy(SAMPLE_COUNT, rng)
        b = sample_mixture_numpy(SAMPLE_COUNT, rng)

        d = energy_distance(a, b)
        distances.append(d)

        print(
            f"  exact-vs-exact pair {i+1}/20: D={d:.6f}",
            flush=True
        )

    return distances, float(np.quantile(distances, 0.95))


# ------------------------------------------------------------
# 6. Compare DDPM, DDIM and flow matching at different step counts
# ------------------------------------------------------------

def main():
    started = time.time()

    print("Backend:", jax.default_backend())
    print("Evaluation samples per run:", SAMPLE_COUNT)

    # Train a new velocity network, not a reused noise predictor.
    flow_params, flow_losses = train_velocity()
    velocity_metrics = validate_velocity(flow_params)

    print("\nLoading the Part 1 noise predictor for DDPM/DDIM...",
          flush=True)
    noise_params = load_params()

    rng = np.random.default_rng(SEED + 50)
    exact = sample_mixture_numpy(SAMPLE_COUNT, rng)
    exact_self = mean_pair_distance(
        exact, exact, exclude_diagonal=True
    )

    null_distances, null95 = estimate_null_scale(rng)

    # Use the same initial Gaussian noise for each sampler and step count.
    x_key = jax.random.PRNGKey(SEED + 100)
    x_start = jax.random.normal(x_key, (SAMPLE_COUNT, 2))

    results = {
        "seed": SEED,
        "sample_count_per_run": SAMPLE_COUNT,
        "step_counts_ddpm_ddim": STEP_COUNTS,
        "step_counts_flow": FLOW_STEPS,
        "flow_training_updates": FLOW_UPDATES,
        "flow_final_loss": float(flow_losses[-1]),
        "velocity_field_relative_rms": velocity_metrics,
        "energy_distance_null_samples": null_distances,
        "energy_distance_null_95_quantile": null95,
        "runs": {}
    }

    all_samples = {"exact_target": exact}
    with open(ARTIFACTS / "part3_metrics.json", "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    def record_run(name, samples, steps):
        samples = np.asarray(samples, dtype=np.float32)
        metrics = evaluate_samples(samples, exact, exact_self)
        metrics["steps"] = int(steps)
        results["runs"][name] = metrics
        all_samples[name] = samples

        print(
            f"  {name:12s} | ED={metrics['energy_distance']:.6f}"
            f" | max fraction error={metrics['max_cluster_fraction_error']:.4f}"
            f" | max width error={metrics['max_width_error_from_0.30']:.4f}",
            flush=True
        )

        # Save incremental progress in case a later run is interrupted.
        with open(ARTIFACTS / "part3_metrics.json", "w",
                  encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    for k in STEP_COUNTS:
        print(f"\n========== Sampling with {k} steps ==========",
              flush=True)
        pairs = make_pairs(k)

        print("  Running DDIM...", flush=True)
        ddim_final, _ = ddim_learned(
            noise_params, x_start, pairs
        )
        record_run(f"ddim_{k}", np.asarray(ddim_final), k)

        print("  Running DDPM...", flush=True)
        noise_key = jax.random.fold_in(
            jax.random.PRNGKey(SEED + 200), k
        )
        reverse_noise = jax.random.normal(
            noise_key, (k, SAMPLE_COUNT, 2)
        )
        ddpm_final, _ = ddpm_stochastic(
            noise_params, x_start, pairs, reverse_noise
        )
        record_run(f"ddpm_{k}", np.asarray(ddpm_final), k)

        if k in FLOW_STEPS:
            print("  Running flow-matching Euler sampler...", flush=True)
            levels = jnp.asarray(
                np.linspace(1.0, 0.0, k + 1, dtype=np.float32)
            )
            flow_final = flow_euler_sample(
                flow_params, x_start, levels
            )
            record_run(f"flow_{k}", np.asarray(flow_final), k)

    # Save generated samples for later inspection and reproducibility.
    np.savez_compressed(
        ARTIFACTS / "part3_generated_samples.npz",
        **all_samples
    )

    # Plot all three metrics versus step count.
    fig, axes = plt.subplots(
        1, 3, figsize=(17, 5), constrained_layout=True
    )

    for sampler, counts, marker in [
        ("ddpm", STEP_COUNTS, "o"),
        ("ddim", STEP_COUNTS, "s"),
        ("flow", FLOW_STEPS, "^")
    ]:
        rows = [
            (k, results["runs"][f"{sampler}_{k}"])
            for k in counts
        ]

        xs = [r[0] for r in rows]
        ys_ed = [max(r[1]["energy_distance"], 1e-4) for r in rows]
        ys_fraction = [
            r[1]["max_cluster_fraction_error"] for r in rows
        ]
        ys_width = [
            r[1]["max_width_error_from_0.30"] for r in rows
        ]

        axes[0].plot(xs, ys_ed, marker=marker, label=sampler.upper())
        axes[1].plot(
            xs, ys_fraction, marker=marker, label=sampler.upper()
        )
        axes[2].plot(xs, ys_width, marker=marker, label=sampler.upper())

    axes[0].axhline(
        max(null95, 1e-4), linestyle="--",
        label=f"Exact-vs-exact 95% = {null95:.4f}"
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Energy distance")
    axes[0].set_ylabel("Energy distance (log scale)")

    two_se = 2.0 * np.sqrt(0.25 * 0.75 / SAMPLE_COUNT)
    axes[1].axhline(
        two_se, linestyle="--",
        label=f"2 binomial SE = {two_se:.4f}"
    )
    axes[1].set_title("Largest cluster-fraction error")
    axes[1].set_ylabel("Maximum absolute error from 0.25")

    axes[2].axhline(0.03, linestyle="--", label="Width tolerance 0.03")
    axes[2].set_title("Largest width error")
    axes[2].set_ylabel("Maximum absolute error from 0.30")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Number of sampling steps")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    for ax in axes:
        ax.axvline(20, color="gray", linestyle=":", alpha=0.7)

    fig.suptitle("Part 3: DDPM vs DDIM vs Flow Matching")
    fig.savefig(EVIDENCE / "part3_sampling_comparison.png", dpi=160)
    plt.close(fig)

    # Record the required experiment configuration.
    with open(ARTIFACTS / "part3_settings.json", "w",
              encoding="utf-8") as f:
        json.dump({
            "seed": SEED,
            "sample_count_per_run": SAMPLE_COUNT,
            "step_counts_ddpm_ddim": STEP_COUNTS,
            "step_counts_flow": FLOW_STEPS,
            "flow_updates": FLOW_UPDATES,
            "flow_batch_size": FLOW_BATCH,
            "flow_learning_rate": FLOW_LR,
            "flow_lr_schedule": "linear decay to zero",
            "energy_distance": "all-pairs empirical estimate",
            "null_pairs": 20,
            "null_quantile": 0.95,
            "null_95_quantile": null95
        }, f, indent=2)

    print("\n========== Part 3 complete ==========")
    print("Velocity metrics:", ARTIFACTS / "part3_velocity_metrics.json")
    print("Sampling metrics:", ARTIFACTS / "part3_metrics.json")
    print("Comparison plot:", EVIDENCE / "part3_sampling_comparison.png")
    print("Velocity field plot:", EVIDENCE / "part3_velocity_fields.png")
    print(f"Total elapsed: {(time.time()-started)/60:.1f} minutes")


if __name__ == "__main__":
    main()
