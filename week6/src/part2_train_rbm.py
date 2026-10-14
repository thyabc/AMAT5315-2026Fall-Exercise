from pathlib import Path
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
EVIDENCE = ROOT / "evidence"
ARTIFACTS = ROOT / "artifacts"
EVIDENCE.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)

VISIBLE_DIM = 28 * 28
HIDDEN_DIM = 128
BATCH_SIZE = 100
EPOCHS = 20
LEARNING_RATE = 0.1
TRAIN_SEED = 123


def binarize_mnist(images):
    x = (images >= 128).astype(np.float32)
    return x.reshape(len(x), -1)


def init_params(key, n_visible=VISIBLE_DIM, n_hidden=HIDDEN_DIM):
    w_key, = jax.random.split(key, 1)
    params = {
        "W": 0.01 * jax.random.normal(w_key, (n_visible, n_hidden)),
        "a": jnp.zeros((n_visible,), dtype=jnp.float32),
        "b": jnp.zeros((n_hidden,), dtype=jnp.float32),
    }
    return params


def hidden_prob(params, v):
    return jax.nn.sigmoid(jnp.matmul(v, params["W"]) + params["b"])


def visible_prob(params, h):
    return jax.nn.sigmoid(jnp.matmul(h, params["W"].T) + params["a"])


def sample_bernoulli(key, probs):
    return jax.random.bernoulli(key, probs).astype(jnp.float32)


def gibbs_step(params, v, key):
    key_h, key_v = jax.random.split(key)
    p_h = hidden_prob(params, v)
    h = sample_bernoulli(key_h, p_h)
    p_v = visible_prob(params, h)
    v_new = sample_bernoulli(key_v, p_v)
    return v_new, h


def free_energy(params, v):
    vbias = jnp.dot(v, params["a"])
    hidden_term = jnp.sum(
        jax.nn.softplus(params["b"] + jnp.matmul(v, params["W"]))
    )
    return -vbias - hidden_term


def free_energy_batch(params, v_batch):
    vbias = jnp.matmul(v_batch, params["a"])
    hidden_term = jnp.sum(
        jax.nn.softplus(params["b"] + jnp.matmul(v_batch, params["W"])),
        axis=1
    )
    return -vbias - hidden_term


def make_cd_loss(k):
    def loss_fn(params, v0, key):
        vk = v0
        keys = jax.random.split(key, k)
        for i in range(k):
            vk, _ = gibbs_step(params, vk, keys[i])
        vk = jax.lax.stop_gradient(vk)
        return jnp.mean(free_energy_batch(params, v0) - free_energy_batch(params, vk))
    return loss_fn


def make_update_step(k):
    loss_fn = make_cd_loss(k)

    @jax.jit
    def update_step(params, batch, key, lr):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch, key)
        new_params = jax.tree_util.tree_map(
            lambda p, g: p - lr * g, params, grads
        )
        return new_params, loss

    return update_step, loss_fn


@jax.jit
def one_step_reconstruct(params, v, key):
    key_h, key_v = jax.random.split(key)
    p_h = hidden_prob(params, v)
    h = sample_bernoulli(key_h, p_h)
    p_v = visible_prob(params, h)
    v1 = sample_bernoulli(key_v, p_v)
    return v1


def reconstruction_error(params, data, seed=0, batch_size=500):
    errs = []
    n = len(data)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        batch = jnp.asarray(data[start:stop])
        key = jax.random.PRNGKey(seed + start)
        recon = one_step_reconstruct(params, batch, key)
        err = jnp.mean(jnp.abs(recon - batch))
        errs.append(float(err))
    return float(np.mean(errs))


def surrogate_on_batch(params, batch, k, seed=0):
    loss_fn = make_cd_loss(k)
    key = jax.random.PRNGKey(seed)
    val = loss_fn(params, jnp.asarray(batch), key)
    return float(val)


def train_rbm(train_x, test_x, init_params_dict, k, epoch_perms):
    update_step, loss_fn = make_update_step(k)

    params = jax.tree_util.tree_map(lambda x: jnp.array(x), init_params_dict)

    history = {
        "epoch": [0],
        "cd_surrogate": [],
        "reconstruction_error": [],
    }

    first_batch = train_x[epoch_perms[0][:BATCH_SIZE]]
    init_sur = surrogate_on_batch(params, first_batch, k, seed=10 + k)
    init_rec = reconstruction_error(params, test_x, seed=20 + k)

    history["cd_surrogate"].append(init_sur)
    history["reconstruction_error"].append(init_rec)

    print(f"\n=== Training CD-{k} ===")
    print(f"Initial CD-{k} surrogate: {init_sur:.6f}")
    print(f"Initial reconstruction error: {init_rec:.6f}")

    rng = jax.random.PRNGKey(1000 + k)
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        perm = epoch_perms[epoch - 1]
        losses = []

        for start in range(0, len(train_x), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            batch = jnp.asarray(train_x[idx])
            rng, subkey = jax.random.split(rng)
            params, loss = update_step(params, batch, subkey, LEARNING_RATE)
            losses.append(float(loss))

        epoch_loss = float(np.mean(losses))
        rec_err = reconstruction_error(params, test_x, seed=2000 + epoch + k)

        history["epoch"].append(epoch)
        history["cd_surrogate"].append(epoch_loss)
        history["reconstruction_error"].append(rec_err)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"CD-{k} surrogate = {epoch_loss:.6f} | "
            f"recon error = {rec_err:.6f}"
        )

    elapsed = time.time() - start_time
    print(f"CD-{k} training finished in {elapsed:.1f} seconds.")

    return params, history, elapsed


def plot_training_curves(hist_cd1, hist_cd10):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(hist_cd1["epoch"], hist_cd1["cd_surrogate"], marker="o", label="CD-1")
    axes[0].plot(hist_cd10["epoch"], hist_cd10["cd_surrogate"], marker="o", label="CD-10")
    axes[0].set_title("CD surrogate (not likelihood)")
    axes[0].set_xlabel("Training epoch")
    axes[0].set_ylabel("Mean free-energy difference")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(hist_cd1["epoch"], hist_cd1["reconstruction_error"], marker="o", label="CD-1")
    axes[1].plot(hist_cd10["epoch"], hist_cd10["reconstruction_error"], marker="o", label="CD-10")
    axes[1].set_title("Held-out one-step reconstruction error")
    axes[1].set_xlabel("Training epoch")
    axes[1].set_ylabel("Fraction of changed pixels")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle("RBM training diagnostics on binarized MNIST", fontsize=15)
    fig.tight_layout()
    out = EVIDENCE / "part2_training_curves.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def save_params(path, params):
    np.savez(
        path,
        W=np.asarray(params["W"]),
        a=np.asarray(params["a"]),
        b=np.asarray(params["b"]),
    )


def main():
    train_images, train_labels = load_mnist("train")
    test_images, test_labels = load_mnist("test")

    train_x = binarize_mnist(train_images)
    test_x = binarize_mnist(test_images)

    assert train_x.shape == (60000, 784)
    assert test_x.shape == (10000, 784)
    assert set(np.unique(train_x)).issubset({0.0, 1.0})
    assert set(np.unique(test_x)).issubset({0.0, 1.0})

    print("Train set:", train_x.shape)
    print("Test set :", test_x.shape)
    print("Visible units:", VISIBLE_DIM)
    print("Hidden units :", HIDDEN_DIM)
    print("Batch size   :", BATCH_SIZE)
    print("Epochs       :", EPOCHS)
    print("Learning rate:", LEARNING_RATE)

    rng_np = np.random.default_rng(TRAIN_SEED)
    epoch_perms = [rng_np.permutation(len(train_x)) for _ in range(EPOCHS)]

    init_key = jax.random.PRNGKey(42)
    shared_init = init_params(init_key)

    params_cd1, hist_cd1, time_cd1 = train_rbm(
        train_x, test_x, shared_init, 1, epoch_perms
    )

    params_cd10, hist_cd10, time_cd10 = train_rbm(
        train_x, test_x, shared_init, 10, epoch_perms
    )

    plot_training_curves(hist_cd1, hist_cd10)

    save_params(ARTIFACTS / "part2_rbm_cd1_params.npz", params_cd1)
    save_params(ARTIFACTS / "part2_rbm_cd10_params.npz", params_cd10)

    summary = {
        "settings": {
            "visible_units": VISIBLE_DIM,
            "hidden_units": HIDDEN_DIM,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "train_seed": TRAIN_SEED,
            "jax_init_seed": 42,
        },
        "cd1": {
            "time_sec": time_cd1,
            "history": hist_cd1,
        },
        "cd10": {
            "time_sec": time_cd10,
            "history": hist_cd10,
        },
    }

    with open(ARTIFACTS / "part2_training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n===== Final check =====")
    print(f"CD-1 final reconstruction error : {hist_cd1['reconstruction_error'][-1]:.6f}")
    print(f"CD-10 final reconstruction error: {hist_cd10['reconstruction_error'][-1]:.6f}")
    print("Saved: evidence/part2_training_curves.png")
    print("Saved: artifacts/part2_rbm_cd1_params.npz")
    print("Saved: artifacts/part2_rbm_cd10_params.npz")
    print("Saved: artifacts/part2_training_summary.json")
    print("\nPart 2 training step completed.")

if __name__ == "__main__":
    main()
