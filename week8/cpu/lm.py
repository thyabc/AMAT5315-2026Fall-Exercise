"""CPU answer key core: data, parameter counts, training and blockwise scoring.

Reuses the instructor speedrun decoder (`speedrun/model.py`), the pinned corpus
(`speedrun/prepare.py`) and the canonical export (`speedrun/train.py`) so every
checkpoint written here is readable by the independent NumPy evaluator in
`week8/checker/evaluator.py`.
"""
import json
import os
import sys
import time
from functools import lru_cache, partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
sys.path.insert(0, str(WEEK / 'speedrun'))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from model import initialize, forward  # noqa: E402
from prepare import prepare  # noqa: E402

BLOCK = 128
CHUNK = 25  # optimizer steps per compiled call
VOCAB = 65
DATA_DIR = HERE / 'work' / 'data'


@lru_cache(maxsize=1)
def corpus():
    """Training ids, validation ids and corpus metadata (pinned SHA-256, 90/10 split)."""
    training, validation, meta = prepare(DATA_DIR)
    return training.astype(np.int64), validation.astype(np.int64), meta


def parameter_counts(config):
    """Total and non-embedding parameter counts of the canonical decoder.

    Non-embedding N (Kaplan et al. 2020) counts the transformer blocks only:
    it excludes the token and position embeddings, the final norm and the
    output (unembedding) layer.
    """
    d, layers = config['n_embd'], config['n_layer']
    per_block = 2 * 2 * d + (d * 3 * d + 3 * d) + (d * d + d) + (d * 4 * d + 4 * d) + (4 * d * d + d)
    non_embedding = layers * per_block
    total = non_embedding + VOCAB * d + BLOCK * d + 2 * d + d * VOCAB + VOCAB
    return dict(non_embedding=non_embedding, total=total)


def make_config(n_layer, n_embd, n_head, **training):
    config = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=BLOCK,
                  vocab_size=VOCAB, dropout=0.0)
    config.update(training)
    return config


def schedule(config):
    return optax.warmup_cosine_decay_schedule(
        0.0, config['learning_rate'], config['warmup_steps'], config['max_steps'],
        config['min_lr'])


def optimizer(config):
    return optax.chain(optax.clip_by_global_norm(1.0),
                       optax.adamw(schedule(config), b1=0.9, b2=0.99,
                                   weight_decay=config['weight_decay']))


def blocks(ids):
    """Contract blocks: inputs v[s:s+n], targets v[s+1:s+n+1], s = 0, 128, ..."""
    count = int(np.ceil((len(ids) - 1) / BLOCK))
    x = np.zeros((count, BLOCK), np.int32)
    y = np.zeros_like(x)
    mask = np.zeros((count, BLOCK), bool)
    for i, start in enumerate(range(0, len(ids) - 1, BLOCK)):
        n = min(BLOCK, len(ids) - start - 1)
        x[i, :n] = ids[start:start + n]
        y[i, :n] = ids[start + 1:start + n + 1]
        mask[i, :n] = True
    return x, y, mask


@partial(jax.jit, static_argnames=('layers', 'heads', 'width', 'causal'))
def _token_nll(params, inputs, targets, layers, heads, width, causal=True):
    config = dict(n_layer=layers, n_head=heads, n_embd=width)
    logp = jax.nn.log_softmax(forward(params, inputs, config, causal=causal), axis=-1)
    return -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]


def score(params, config, ids, causal=True, max_blocks=None, batch=64):
    """Token-weighted loss over contract blocks. Returns per-block sums and counts too."""
    x, y, mask = blocks(ids)
    if max_blocks is not None:
        x, y, mask = x[:max_blocks], y[:max_blocks], mask[:max_blocks]
    block_nll = []
    for start in range(0, len(x), batch):
        take = min(batch, len(x) - start)
        xb = np.zeros((batch, BLOCK), np.int32)
        yb = np.zeros_like(xb)
        xb[:take], yb[:take] = x[start:start + take], y[start:start + take]
        nll = np.asarray(_token_nll(params, xb, yb, config['n_layer'], config['n_head'],
                                    config['n_embd'], causal), dtype=np.float64)[:take]
        block_nll.extend((nll * mask[start:start + take]).sum(axis=1).tolist())
    counts = mask.sum(axis=1).tolist()
    total = float(sum(block_nll))
    tokens = int(sum(counts))
    return dict(summed_nll=total, token_count=tokens, loss=total / tokens,
                ppl=float(np.exp(total / tokens)), bpc=total / tokens / np.log(2),
                block_nll=block_nll, block_tokens=counts)


def train(config, *, causal=True, log_every=100, curve_blocks=96, verbose=True):
    """Train one model from `config`; return params, log rows and timing.

    The curve is evaluated on the first `curve_blocks` validation blocks (a fixed
    subset, for speed); the final number is always the full contract score.
    """
    training, validation, _ = corpus()
    training = jnp.asarray(training, jnp.int32)
    tx = optimizer(config)
    key = jax.random.PRNGKey(config['seed'])
    key, init_key = jax.random.split(key)
    params = initialize(config, init_key)
    state = tx.init(params)
    batch = config['batch_size']
    lr = schedule(config)

    def one_step(carry, _):
        params, state, key = carry
        key, batch_key = jax.random.split(key)
        starts = jax.random.randint(batch_key, (batch,), 0, len(training) - BLOCK)
        x = training[starts[:, None] + jnp.arange(BLOCK)]
        y = training[starts[:, None] + jnp.arange(1, BLOCK + 1)]

        def objective(p):
            logits = forward(p, x, config, causal=causal)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        loss, grad = jax.value_and_grad(objective)(params)
        changes, state = tx.update(grad, state, params)
        return (optax.apply_updates(params, changes), state, key), loss

    if log_every % CHUNK or config['max_steps'] % log_every:
        raise ValueError(f'log_every must be a multiple of {CHUNK} and divide max_steps')
    # Scanning CHUNK updates inside one compiled call removes per-step dispatch.
    update = jax.jit(lambda carry: jax.lax.scan(one_step, carry, None, length=CHUNK))

    rows, window = [], []
    started = time.perf_counter()
    compile_seconds = None
    train_seconds = 0.0
    carry = (params, state, key)
    for step in range(CHUNK, config['max_steps'] + 1, CHUNK):
        before = time.perf_counter()
        carry, losses = update(carry)
        losses = np.asarray(losses)  # synchronizes before timing
        elapsed = time.perf_counter() - before
        if compile_seconds is None:
            compile_seconds = elapsed
        else:
            train_seconds += elapsed
        window.extend(losses.tolist())
        if step % log_every == 0:
            params = carry[0]
            val = score(params, config, validation, causal=causal, max_blocks=curve_blocks)['loss']
            rows.append(dict(step=step, tokens=step * batch * BLOCK,
                             train_loss=float(np.mean(window)), val_loss_subset=val,
                             lr=float(lr(step - 1)),
                             wall_seconds=time.perf_counter() - started))
            window = []
            if verbose:
                print(f"  step {step}: train={rows[-1]['train_loss']:.4f} "
                      f"val[{curve_blocks}]={val:.4f} wall={rows[-1]['wall_seconds']:.0f}s",
                      flush=True)
    params = carry[0]
    final = score(params, config, validation, causal=causal)
    steps_timed = config['max_steps'] - CHUNK
    timing = dict(wall_seconds=time.perf_counter() - started, first_step_seconds=compile_seconds,
                  steps_per_second=steps_timed / train_seconds if train_seconds else None,
                  tokens_per_second=steps_timed * batch * BLOCK / train_seconds if train_seconds else None,
                  cpu_count=os.cpu_count(), device=str(jax.devices()[0]), jax_version=jax.__version__)
    return params, rows, final, timing


def export(directory, params, config, step):
    """Canonical model.npz + model.json (the checker's export schema)."""
    from train import export_model  # speedrun/train.py
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _, _, meta = corpus()
    export_model(directory, params, config, meta, step)
    return json.loads((directory / 'model.json').read_text())


def load(directory):
    directory = Path(directory)
    meta = json.loads((directory / 'model.json').read_text())
    with np.load(directory / 'model.npz', allow_pickle=False) as archive:
        params = {k: jnp.asarray(archive[k]) for k in archive.files}
    return params, meta


def causality_probe(params, config, causal=True, prefix=64, pairs=8, seed=0):
    """Max |change| of logits at positions < prefix when later characters change."""
    _, validation, _ = corpus()
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(pairs):
        start = int(rng.integers(0, len(validation) - BLOCK))
        a = validation[start:start + BLOCK].astype(np.int32)[None]
        b = a.copy()
        b[0, prefix:] = rng.integers(0, VOCAB, BLOCK - prefix)
        la = np.asarray(forward(params, jnp.asarray(a), config, causal=causal))
        lb = np.asarray(forward(params, jnp.asarray(b), config, causal=causal))
        worst = max(worst, float(np.max(np.abs(la[0, :prefix] - lb[0, :prefix]))))
    return dict(prefix_length=prefix, pairs=pairs, max_prefix_logit_change=worst)


def generate(params, config, vocabulary, prompt, temperature, length, seed):
    lookup = {c: i for i, c in enumerate(vocabulary)}
    ids = [lookup[c] for c in prompt]

    @jax.jit
    def predict(window, index):
        return forward(params, window, config)[0, index]
    key = jax.random.PRNGKey(seed)
    for _ in range(length):
        context = ids[-BLOCK:]
        padded = np.zeros((1, BLOCK), np.int32)
        padded[0, :len(context)] = context
        logits = predict(padded, len(context) - 1)
        key, sub = jax.random.split(key)
        ids.append(int(jax.random.categorical(sub, logits / temperature)))
    return ''.join(vocabulary[i] for i in ids)


MEASUREMENTS = WEEK / 'data' / 'cpu-measurements.json'
EVIDENCE = WEEK / 'data' / 'cpu-evidence.npz'


def record(section, data):
    """Merge one section into week8/data/cpu-measurements.json."""
    store = json.loads(MEASUREMENTS.read_text()) if MEASUREMENTS.exists() else {}
    store[section] = data
    MEASUREMENTS.write_text(json.dumps(store, indent=2) + '\n')


def save_arrays(**arrays):
    """Merge arrays into week8/data/cpu-evidence.npz (small figure inputs)."""
    existing = {}
    if EVIDENCE.exists():
        with np.load(EVIDENCE, allow_pickle=False) as archive:
            existing = {k: archive[k] for k in archive.files}
    existing.update({k: np.asarray(v) for k, v in arrays.items()})
    np.savez_compressed(EVIDENCE, **existing)


def independent_score(directory):
    """Rescore exported weights with the checker's float64 NumPy evaluator (separate process)."""
    import subprocess
    out = Path(directory) / 'independent-eval.json'
    env = dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
    subprocess.run([sys.executable, str(WEEK / 'checker' / 'evaluator.py'),
                    '--artifacts', str(directory), '--out', str(out)],
                   check=True, cwd=WEEK / 'checker', env=env, capture_output=True)
    report = json.loads(out.read_text())
    return {k: report[k] for k in ('loss', 'ppl', 'token_count', 'sum_nll', 'checkpoint_sha256')}
