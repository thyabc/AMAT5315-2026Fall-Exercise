import os
import json
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
ARTIFACTS = ROOT / "artifacts"
CHECKPOINTS = ARTIFACTS / "checkpoints"

for folder in (EVIDENCE, ARTIFACTS, CHECKPOINTS):
    folder.mkdir(parents=True, exist_ok=True)

SEED = 20261021
T = 1000
UPDATES = 25000
BATCH_SIZE = 512
N_EVAL = 2000
BASE_LR = 2e-3
HIDDEN = 128
SIGMA = 0.3

MEANS_NP = np.array([
    [3., 0.], [0., 3.], [-3., 0.], [0., -3.]
], dtype=np.float32)

BETAS_NP = np.linspace(0.0001, 0.02, T, dtype=np.float32)
ALPHAS_NP = 1.0 - BETAS_NP
ALPHA_BARS_NP = np.cumprod(ALPHAS_NP)

MEANS = jnp.asarray(MEANS_NP)
BETAS = jnp.asarray(BETAS_NP)
ALPHAS = jnp.asarray(ALPHAS_NP)
ALPHA_BARS = jnp.asarray(ALPHA_BARS_NP)


def sample_mixture_numpy(n, rng):
    labels = rng.integers(0, 4, size=n)
    return (
        MEANS_NP[labels]
        + rng.normal(0, SIGMA, size=(n, 2)).astype(np.float32)
    )


def time_features(t):
    t = jnp.asarray(t, dtype=jnp.float32).reshape(-1) / T
    freqs = 2.0 ** jnp.arange(8, dtype=jnp.float32)
    phase = 2.0 * jnp.pi * t[:, None] * freqs[None, :]
    return jnp.concatenate([
        t[:, None],
        jnp.sin(phase),
        jnp.cos(phase)
    ], axis=-1)


def model_apply(params, x, t):
    h = jnp.concatenate([x, time_features(t)], axis=-1)

    for i, (w, b) in enumerate(params):
        h = h @ w + b
        if i < len(params) - 1:
            h = jax.nn.silu(h)

    return h


def init_params(key):
    dims = [19, HIDDEN, HIDDEN, HIDDEN, 2]
    keys = jax.random.split(key, len(dims) - 1)
    params = []

    for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
        scale = np.sqrt(2.0 / (din + dout))
        w = jax.random.normal(
            keys[i], (din, dout), dtype=jnp.float32
        ) * scale
        b = jnp.zeros((dout,), dtype=jnp.float32)
        params.append((w, b))

    return params


def loss_fn(params, xt, t, eps):
    pred = model_apply(params, xt, t)
    return jnp.mean(0.5 * jnp.sum((pred - eps) ** 2, axis=-1))


@jax.jit
def train_step(params, m, v, step, key):
    k_comp, k_data, k_t, k_eps = jax.random.split(key, 4)

    labels = jax.random.randint(k_comp, (BATCH_SIZE,), 0, 4)
    x0 = MEANS[labels] + SIGMA * jax.random.normal(
        k_data, (BATCH_SIZE, 2)
    )

    t = jax.random.randint(k_t, (BATCH_SIZE,), 1, T + 1)
    eps = jax.random.normal(k_eps, (BATCH_SIZE, 2))

    ab = ALPHA_BARS[t - 1]
    xt = (
        jnp.sqrt(ab[:, None]) * x0
        + jnp.sqrt(1.0 - ab[:, None]) * eps
    )

    loss, grads = jax.value_and_grad(loss_fn)(
        params, xt, t, eps
    )

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
        lambda x: x / (1.0 - b1 ** step1), m
    )
    vhat = jax.tree_util.tree_map(
        lambda x: x / (1.0 - b2 ** step1), v
    )

    lr = BASE_LR * jnp.maximum(
        0.0, 1.0 - step / (UPDATES - 1)
    )

    params = jax.tree_util.tree_map(
        lambda p, mm, vv: p - lr * mm / (jnp.sqrt(vv) + 1e-8),
        params, mhat, vhat
    )

    return params, m, v, loss


@jax.jit
def ddpm_trajectory(params, xT, noise_seq):
    def one_step(x, inputs):
        idx, z = inputs
        t = idx + 1

        t_batch = jnp.full(
            (x.shape[0],), t, dtype=jnp.int32
        )
        eps_hat = model_apply(params, x, t_batch)

        beta = BETAS[idx]
        alpha = ALPHAS[idx]
        alpha_bar = ALPHA_BARS[idx]

        mean = (
            x - beta / jnp.sqrt(1.0 - alpha_bar) * eps_hat
        ) / jnp.sqrt(alpha)

        x_prev = jax.lax.cond(
            idx > 0,
            lambda _: mean + jnp.sqrt(beta) * z,
            lambda _: mean,
            operand=None
        )

        return x_prev, x_prev

    indices = jnp.arange(T - 1, -1, -1, dtype=jnp.int32)
    _, states = jax.lax.scan(
        one_step, xT, (indices, noise_seq)
    )

    return jnp.concatenate([xT[None, ...], states], axis=0)


def params_as_numpy(params):
    return jax.tree_util.tree_map(lambda a: np.asarray(a), params)


def save_checkpoint(params, step):
    p = params_as_numpy(params)
    data = {}

    for i, (w, b) in enumerate(p):
        data[f"w{i}"] = w
        data[f"b{i}"] = b

    path = CHECKPOINTS / f"part1_stage_step_{step:05d}.npz"
    np.savez_compressed(path, **data)
    return p


def fig_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def plot_scatter_grid(states, levels, path):
    fig, axes = plt.subplots(
        2, 3, figsize=(12, 7), constrained_layout=True
    )

    for ax, t in zip(axes.flat, levels):
        p = states[t]
        ax.scatter(p[:, 0], p[:, 1], s=4, alpha=0.45)
        ax.set_title(f"t = {t}")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)

    fig.suptitle("Part 1 - Forward diffusion")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_scatter_gif(states, levels, path, title):
    frames = []

    for t in levels:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        p = states[t]
        ax.scatter(p[:, 0], p[:, 1], s=5, alpha=0.45)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        ax.set_title(f"{title}: t = {t}")
        frames.append(fig_rgb(fig))
        plt.close(fig)

    imageio.mimsave(path, frames, duration=0.7, loop=0)


def make_training_gif(stage_records, trajectories, losses, path):
    frames = []
    last_fig = None

    for (step, _), traj in zip(stage_records, trajectories):
        fig, (ax_path, ax_loss) = plt.subplots(
            1, 2, figsize=(11, 5), constrained_layout=True
        )

        for i in range(traj.shape[1]):
            ax_path.plot(
                traj[:, i, 0], traj[:, i, 1],
                linewidth=0.7, alpha=0.6
            )
            ax_path.scatter(
                traj[-1, i, 0], traj[-1, i, 1], s=18
            )

        ax_path.scatter(
            MEANS_NP[:, 0], MEANS_NP[:, 1],
            marker="x", s=80, linewidth=2
        )
        ax_path.set_xlim(-5, 5)
        ax_path.set_ylim(-5, 5)
        ax_path.set_aspect("equal", adjustable="box")
        ax_path.grid(alpha=0.2)
        ax_path.set_title(f"DDPM paths - update {step}")

        if step > 0:
            ax_loss.plot(
                np.arange(1, step + 1),
                np.asarray(losses[:step]),
                linewidth=0.8
            )
        else:
            ax_loss.text(
                0.5, 0.5, "Initialization",
                ha="center", va="center",
                transform=ax_loss.transAxes
            )

        ax_loss.set_xlabel("Training update")
        ax_loss.set_ylabel("Noise-prediction loss")
        ax_loss.set_title("Training loss")
        ax_loss.grid(alpha=0.2)

        frames.append(fig_rgb(fig))

        if last_fig is not None:
            plt.close(last_fig)
        last_fig = fig

    if last_fig is not None:
        last_fig.savefig(
            EVIDENCE / "training_stages_final.png", dpi=160
        )
        plt.close(last_fig)

    imageio.mimsave(path, frames, duration=1.0, loop=0)


def nearest_cluster_metrics(points):
    distances = np.linalg.norm(
        points[:, None, :] - MEANS_NP[None, :, :], axis=-1
    )
    labels = distances.argmin(axis=1)
    populations = np.bincount(labels, minlength=4) / len(points)

    centers, spreads = [], []

    for k in range(4):
        group = points[labels == k]

        if len(group):
            centers.append(group.mean(axis=0).tolist())
            spreads.append(group.std(axis=0).tolist())
        else:
            centers.append([float("nan"), float("nan")])
            spreads.append([float("nan"), float("nan")])

    return populations, np.asarray(centers), np.asarray(spreads)


def main():
    start_time = time.time()
    rng = np.random.default_rng(SEED)

    settings = {
        "part": "Week 7 Part 1 - Gaussian mixture diffusion",
        "seed": SEED,
        "diffusion_steps": T,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "mixture_centers": MEANS_NP.tolist(),
        "component_covariance": "0.09 * I",
        "component_std": SIGMA,
        "training_updates": UPDATES,
        "batch_size": BATCH_SIZE,
        "base_learning_rate": BASE_LR,
        "learning_rate_schedule": "linear decay to zero",
        "hidden_layers": 3,
        "hidden_width": HIDDEN,
        "evaluation_samples": N_EVAL,
        "sampler": "DDPM; beta variance; no clipping; no final noise",
        "backend": jax.default_backend()
    }

    (ARTIFACTS / "part1_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )

    # 1. Forward diffusion
    print("\n[1/4] Forward diffusion...", flush=True)

    x0 = sample_mixture_numpy(N_EVAL, rng)
    levels = [0, 50, 200, 500, 900, 1000]
    forward_states = {0: x0.copy()}
    x = x0.copy()

    for t in range(1, T + 1):
        i = t - 1
        x = (
            np.sqrt(ALPHAS_NP[i]) * x
            + np.sqrt(BETAS_NP[i])
            * rng.normal(size=x.shape).astype(np.float32)
        )

        if t in levels:
            forward_states[t] = x.copy()

    end_mean = forward_states[T].mean(axis=0)
    end_var = forward_states[T].var(axis=0)

    print(f"Forward t=1000 mean: {end_mean}")
    print(f"Forward t=1000 variance: {end_var}")

    plot_scatter_grid(
        forward_states, levels,
        EVIDENCE / "forward_diffusion.png"
    )
    save_scatter_gif(
        forward_states, levels,
        EVIDENCE / "forward_diffusion.gif",
        "Forward diffusion"
    )

    (ARTIFACTS / "forward_statistics.json").write_text(
        json.dumps({
            "t": T,
            "mean": end_mean.tolist(),
            "variance": end_var.tolist()
        }, indent=2),
        encoding="utf-8"
    )

    # 2. Train noise predictor
    print("\n[2/4] Training noise predictor...", flush=True)

    key = jax.random.PRNGKey(SEED)
    params = init_params(key)
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)

    stage_steps = [
        0, 10, 100, 500, 1000,
        2500, 5000, 10000, 17500, 25000
    ]

    stage_records = [(0, save_checkpoint(params, 0))]
    loss_history = []

    for step in range(UPDATES):
        step_key = jax.random.fold_in(key, step + 1)

        params, m, v, loss = train_step(
            params, m, v,
            jnp.asarray(float(step), dtype=jnp.float32),
            step_key
        )

        loss_history.append(float(loss))
        current_step = step + 1

        if current_step in stage_steps:
            stage_records.append(
                (current_step, save_checkpoint(params, current_step))
            )

        if current_step % 1000 == 0:
            print(
                f"  update {current_step}/{UPDATES}"
                f" | loss={loss_history[-1]:.5f}",
                flush=True
            )

    np.savetxt(
        ARTIFACTS / "part1_loss_history.csv",
        np.asarray(loss_history),
        delimiter=",",
        header="noise_prediction_loss",
        comments=""
    )

    print(
        f"Training complete; saved {len(stage_records)} model stages.",
        flush=True
    )

    # 3. Training-stage trajectories, using shared noise
    print("\n[3/4] Creating training-stage movie...", flush=True)

    key1, key2 = jax.random.split(
        jax.random.PRNGKey(SEED + 1)
    )
    path_starts = jax.random.normal(key1, (10, 2))
    shared_noise = jax.random.normal(key2, (T, 10, 2))

    stage_trajectories = []

    for stage_step, stage_params in stage_records:
        trajectory = ddpm_trajectory(
            stage_params, path_starts, shared_noise
        )
        stage_trajectories.append(np.asarray(trajectory))

    make_training_gif(
        stage_records,
        stage_trajectories,
        loss_history,
        EVIDENCE / "training_stages.gif"
    )

    np.savez_compressed(
        ARTIFACTS / "training_stage_paths.npz",
        stage_steps=np.asarray([s for s, _ in stage_records]),
        trajectories=np.asarray(stage_trajectories, dtype=np.float32)
    )

    # 4. Generate samples and evaluate
    print("\n[4/4] Generating 2000 samples with DDPM...", flush=True)

    key1, key2 = jax.random.split(
        jax.random.PRNGKey(SEED + 2)
    )
    xT = jax.random.normal(key1, (N_EVAL, 2))
    noise_seq = jax.random.normal(key2, (T, N_EVAL, 2))

    trajectory = np.asarray(
        ddpm_trajectory(params, xT, noise_seq),
        dtype=np.float32
    )

    generated = trajectory[-1]
    exact = sample_mixture_numpy(N_EVAL, rng)

    populations, centers, spreads = nearest_cluster_metrics(generated)

    print("\nGenerated sample metrics:")
    print("cluster | population | center_x | center_y | std_x | std_y")

    for k in range(4):
        print(
            f"{k+1:>7} | {populations[k]:.4f}"
            f" | {centers[k,0]: .3f} | {centers[k,1]: .3f}"
            f" | {spreads[k,0]:.3f} | {spreads[k,1]:.3f}"
        )

    pop_ok = bool(np.all(np.abs(populations - 0.25) <= 0.04))
    spread_ok = bool(np.all(np.abs(spreads - 0.30) <= 0.03))
    center_ok = bool(np.all(np.abs(centers - MEANS_NP) <= 0.05))

    metrics = {
        "sample_count": N_EVAL,
        "populations": populations.tolist(),
        "centers": centers.tolist(),
        "per_coordinate_spreads": spreads.tolist(),
        "max_population_error_from_0.25": float(
            np.max(np.abs(populations - 0.25))
        ),
        "max_spread_error_from_0.30": float(
            np.nanmax(np.abs(spreads - 0.30))
        ),
        "max_center_coordinate_error": float(
            np.nanmax(np.abs(centers - MEANS_NP))
        ),
        "population_check_pass": pop_ok,
        "spread_check_pass": spread_ok,
        "center_check_pass": center_ok,
        "forward_mean": end_mean.tolist(),
        "forward_variance": end_var.tolist(),
        "final_training_loss": float(loss_history[-1])
    }

    (ARTIFACTS / "part1_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Compare generated samples with exact target samples
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 5), constrained_layout=True
    )
    axes[0].scatter(exact[:, 0], exact[:, 1], s=5, alpha=0.45)
    axes[0].set_title("Exact target samples")
    axes[1].scatter(
        generated[:, 0], generated[:, 1], s=5, alpha=0.45
    )
    axes[1].set_title("Generated DDPM samples")

    for ax in axes:
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)

    fig.savefig(EVIDENCE / "generated_vs_target.png", dpi=160)
    plt.close(fig)

    # Reverse-process animation; trajectory[0] is x_T
    frames = []

    for idx in range(0, T + 1, 25):
        t = T - idx
        pts = trajectory[idx]

        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.45)
        ax.scatter(
            MEANS_NP[:, 0], MEANS_NP[:, 1],
            marker="x", s=80, linewidth=2
        )
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        ax.set_title(f"DDPM reverse process - t={t}")

        frames.append(fig_rgb(fig))
        plt.close(fig)

    imageio.mimsave(
        EVIDENCE / "reverse_diffusion.gif",
        frames, duration=0.18, loop=0
    )

    np.savez_compressed(
        ARTIFACTS / "part1_generated_samples.npz",
        generated=generated,
        exact=exact,
        reverse_states=trajectory[::25]
    )

    print("\nVerification checks:")
    print(f"Population tolerance: {'PASS' if pop_ok else 'CHECK'}")
    print(f"Spread tolerance:     {'PASS' if spread_ok else 'CHECK'}")
    print(f"Center tolerance:     {'PASS' if center_ok else 'CHECK'}")
    print(
        "Forward mean tolerance:",
        bool(np.all(np.abs(end_mean) < 0.1))
    )
    print(
        "Forward variance tolerance:",
        bool(np.all(np.abs(end_var - 1.0) < 0.1))
    )

    print(f"\nFinished in {(time.time()-start_time)/60:.1f} minutes.")
    print(f"Evidence:  {EVIDENCE}")
    print(f"Artifacts: {ARTIFACTS}")


if __name__ == "__main__":
    main()
