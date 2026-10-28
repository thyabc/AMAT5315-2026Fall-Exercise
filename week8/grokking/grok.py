"""Grokking answer key: modular addition c = (a + b) mod p on a CPU (JAX).

Two models share one training loop:
  * ``transformer``: one-layer, attention-plus-MLP transformer without LayerNorm on the
    tokens ``a b =`` (Nanda et al. 2023, arXiv 2301.05217). Only the ``=`` position is
    read out, so the forward pass computes queries, the MLP and the unembedding at that
    position only (keys and values still see all three tokens).
  * ``mlp``: the two-layer network of Gromov 2023 (arXiv 2301.02679): one-hot (a, b)
    -> hidden width N with a quadratic activation -> p logits.

Training is full batch AdamW. Every ``log_every`` steps the loop records train/test
loss and accuracy and the parameter norm, and it keeps three snapshots: initialization,
memorization (first logged step with train accuracy >= 0.99) and the final weights.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

HI = jax.lax.Precision.HIGHEST


# ----------------------------------------------------------------------------- data
def make_split(p, frac, seed, labels="true"):
    """All p*p pairs, a seeded permutation, the first round(frac*p*p) for training.

    ``labels="random"`` replaces (a + b) mod p by a fixed random function of the pair
    (negative control): nothing generalizes because there is nothing to find.
    Returns (a, b, c) arrays and the train/test index arrays into them.
    """
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a, b = a.ravel(), b.ravel()
    rng = np.random.default_rng(seed)
    c = (a + b) % p if labels == "true" else rng.integers(0, p, size=a.size)
    perm = rng.permutation(a.size)
    n_train = int(round(frac * a.size))
    return (a, b, c), np.sort(perm[:n_train]), np.sort(perm[n_train:])


# --------------------------------------------------------------------------- models
def init_transformer(key, p, d_model=128, n_heads=4, d_mlp=512):
    d_head = d_model // n_heads
    ks = jax.random.split(key, 8)
    n = lambda k, shape, fan: jax.random.normal(k, shape) / np.sqrt(fan)
    return {
        "W_E": n(ks[0], (p + 1, d_model), d_model),     # rows 0..p-1 residues, row p is '='
        "W_pos": n(ks[1], (3, d_model), d_model),
        "W_Q": n(ks[2], (n_heads, d_model, d_head), d_model),
        "W_K": n(ks[3], (n_heads, d_model, d_head), d_model),
        "W_V": n(ks[4], (n_heads, d_model, d_head), d_model),
        "W_O": n(ks[5], (n_heads, d_head, d_model), d_model),
        "W_in": n(ks[6], (d_model, d_mlp), d_model), "b_in": jnp.zeros(d_mlp),
        "W_out": n(ks[7], (d_mlp, d_model), d_mlp), "b_out": jnp.zeros(d_model),
        "W_U": n(jax.random.fold_in(key, 99), (d_model, p), d_model),
    }


def transformer_logits(params, a, b, W_E=None):
    """Logits at the '=' position. ``W_E`` overrides the residue embedding (ablations)."""
    p = params["W_U"].shape[1]
    E = params["W_E"] if W_E is None else W_E
    eq = jnp.full_like(a, p)
    x = jnp.stack([E[a], E[b], E[eq]], axis=1) + params["W_pos"][None]      # (B, 3, d)
    d_head = params["W_Q"].shape[-1]
    q = jnp.einsum("bd,hde->bhe", x[:, 2], params["W_Q"], precision=HI)
    k = jnp.einsum("btd,hde->bhte", x, params["W_K"], precision=HI)
    v = jnp.einsum("btd,hde->bhte", x, params["W_V"], precision=HI)
    att = jax.nn.softmax(jnp.einsum("bhe,bhte->bht", q, k, precision=HI) / np.sqrt(d_head), -1)
    z = jnp.einsum("bht,bhte->bhe", att, v, precision=HI)
    r = x[:, 2] + jnp.einsum("bhe,hed->bd", z, params["W_O"], precision=HI)
    h = jax.nn.relu(jnp.dot(r, params["W_in"], precision=HI) + params["b_in"])
    r = r + jnp.dot(h, params["W_out"], precision=HI) + params["b_out"]
    return jnp.dot(r, params["W_U"], precision=HI)


def init_mlp(key, p, width=512):
    ks = jax.random.split(key, 3)
    return {"W_a": jax.random.normal(ks[0], (p, width)),          # one-hot a -> hidden
            "W_b": jax.random.normal(ks[1], (p, width)),          # one-hot b -> hidden
            "W_U": jax.random.normal(ks[2], (width, p)) / width}  # Gromov's 1/N readout


def mlp_logits(params, a, b, W_E=None):
    """``W_E`` overrides (W_a, W_b) jointly as a (2, p, width) array (ablations)."""
    Wa, Wb = (params["W_a"], params["W_b"]) if W_E is None else (W_E[0], W_E[1])
    h = Wa[a] + Wb[b]
    return jnp.dot(h * h, params["W_U"], precision=HI)


MODELS = {"transformer": (init_transformer, transformer_logits),
          "mlp": (init_mlp, mlp_logits)}


def residue_embedding(model, params):
    """The matrix whose rows are indexed by a residue: (p, d) or (2, p, width)."""
    if model == "transformer":
        p = params["W_U"].shape[1]
        return np.asarray(params["W_E"])[:p]
    return np.stack([np.asarray(params["W_a"]), np.asarray(params["W_b"])])


def with_embedding(model, params, E):
    """Logit function keyword for an ablated residue embedding."""
    if model == "transformer":
        full = jnp.concatenate([jnp.asarray(E), params["W_E"][E.shape[0]:]], 0)
        return full
    return jnp.asarray(E)


# ------------------------------------------------------------------------- metrics
def loss_acc(logits, c):
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), -1)
    loss = -jnp.mean(jnp.take_along_axis(logp, c[:, None], 1))
    acc = jnp.mean(jnp.argmax(logits, -1) == c)
    return loss, acc


def param_norm(params):
    return float(np.sqrt(sum(float(jnp.sum(v * v)) for v in params.values())))


# ------------------------------------------------------------------------ training
def train(model="transformer", p=113, frac=0.3, wd=1.0, lr=1e-3, steps=10000, seed=0,
          data_seed=None, labels="true", log_every=50, stop_after_grok=None, arch=None,
          betas=(0.9, 0.98), warmup=10, verbose=False):
    """Full-batch AdamW. Returns a dict with the curves and three parameter snapshots.

    ``stop_after_grok``: stop this many steps after test accuracy first reaches 0.99
    (None: run all steps). The grokking step is the first logged step with test
    accuracy >= 0.99; the memorization step the first with train accuracy >= 0.99.
    """
    init, logits_fn = MODELS[model]
    (a, b, c), tr, te = make_split(p, frac, seed if data_seed is None else data_seed, labels)
    A, B, C = jnp.asarray(a), jnp.asarray(b), jnp.asarray(c)
    a_tr, b_tr, c_tr = A[tr], B[tr], C[tr]
    a_te, b_te, c_te = A[te], B[te], C[te]
    params = init(jax.random.PRNGKey(seed), p, **(arch or {}))
    sched = optax.join_schedules([optax.linear_schedule(0.0, lr, warmup), optax.constant_schedule(lr)], [warmup])
    opt = optax.adamw(sched, b1=betas[0], b2=betas[1], weight_decay=wd)
    state = opt.init(params)

    def loss_fn(prm):
        return loss_acc(logits_fn(prm, a_tr, b_tr), c_tr)[0]

    @jax.jit
    def chunk(prm, st):
        def body(_, carry):
            prm, st = carry
            g = jax.grad(loss_fn)(prm)
            upd, st = opt.update(g, st, prm)
            return optax.apply_updates(prm, upd), st
        return jax.lax.fori_loop(0, log_every, body, (prm, st))

    @jax.jit
    def evaluate(prm):
        trl, tra = loss_acc(logits_fn(prm, a_tr, b_tr), c_tr)
        if te.size:
            tel, tea = loss_acc(logits_fn(prm, a_te, b_te), c_te)
        else:
            tel, tea = jnp.nan, jnp.nan
        return trl, tra, tel, tea

    hist = {k: [] for k in ["step", "train_loss", "train_acc", "test_loss", "test_acc", "norm"]}

    def log(step, prm):
        trl, tra, tel, tea = (float(x) for x in evaluate(prm))
        for k, v in zip(hist, [step, trl, tra, tel, tea, param_norm(prm)]):
            hist[k].append(v)
        if verbose and step % (log_every * 20) == 0:
            print(f"  step {step:6d} train {trl:.3e}/{tra:.3f} test {tel:.3e}/{tea:.3f} |W| {hist['norm'][-1]:.1f}", flush=True)
        return tra, tea

    snaps = {"init": jax.tree.map(np.asarray, params)}
    t0 = time.time()
    log(0, params)
    mem_step = grok_step = None
    step = 0
    while step < steps:
        params, state = chunk(params, state)
        step += log_every
        tra, tea = log(step, params)
        if mem_step is None and tra >= 0.99:
            mem_step = step
            snaps["memorized"] = jax.tree.map(np.asarray, params)
        if grok_step is None and tea >= 0.99:
            grok_step = step
        if stop_after_grok is not None and grok_step is not None and step >= grok_step + stop_after_grok:
            break
    seconds = time.time() - t0
    snaps["final"] = jax.tree.map(np.asarray, params)
    return {"config": dict(model=model, p=p, frac=frac, wd=wd, lr=lr, steps=steps, seed=seed,
                           data_seed=seed if data_seed is None else data_seed, labels=labels,
                           log_every=log_every, arch=arch or {}, betas=list(betas), warmup=warmup),
            "history": hist, "memorization_step": mem_step, "grokking_step": grok_step,
            "steps_run": step, "seconds": seconds, "snapshots": snaps,
            "split": {"train": tr, "test": te, "labels": c}}


def accuracy(model, params, p, idx_pairs, labels, E=None):
    """Accuracy of saved parameters on given pairs, optionally with an ablated embedding."""
    _, logits_fn = MODELS[model]
    prm = jax.tree.map(jnp.asarray, params)
    kw = {} if E is None else {"W_E": with_embedding(model, prm, E)}
    a, b = idx_pairs
    logits = logits_fn(prm, jnp.asarray(a), jnp.asarray(b), **kw)
    return float(jnp.mean(jnp.argmax(logits, -1) == jnp.asarray(labels)))


def loss_on(model, params, a, b, c, E=None):
    _, logits_fn = MODELS[model]
    prm = jax.tree.map(jnp.asarray, params)
    kw = {} if E is None else {"W_E": with_embedding(model, prm, E)}
    return float(loss_acc(logits_fn(prm, jnp.asarray(a), jnp.asarray(b), **kw), jnp.asarray(c))[0])
