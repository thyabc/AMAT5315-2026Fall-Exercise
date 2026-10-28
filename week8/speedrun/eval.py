"""Full held-out blockwise score, including the final short block."""
import argparse
import hashlib
from functools import partial
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from model import forward
from prepare import prepare


def metrics(nll, count):
    loss = float(nll) / int(count)
    return dict(token_count=int(count), summed_nll=float(nll), loss=loss,
                ppl=float(np.exp(loss)), bpc=loss/np.log(2))


def load_model(directory):
    directory = Path(directory)
    meta = json.loads((directory / 'model.json').read_text())
    digest = hashlib.sha256((directory / 'model.npz').read_bytes()).hexdigest()
    if digest != meta['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA-256 mismatch')
    with np.load(directory / 'model.npz', allow_pickle=False) as archive:
        params = {k: jnp.asarray(archive[k]) for k in archive.files}
    return params, meta


@partial(jax.jit, static_argnames=('layers','heads','width'))
def score(params, inputs, targets, layers, heads, width):
    config=dict(n_layer=layers,n_head=heads,n_embd=width)
    logp = jax.nn.log_softmax(forward(params, inputs, config), axis=-1)
    return -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]


def evaluate(params, config, validation, batch_size=32):
    # Pad inputs for one compiled shape, but count only actual target characters.
    blocks = int(np.ceil((len(validation)-1)/128))
    x = np.zeros((blocks, 128), np.int32)
    y = np.zeros_like(x)
    mask = np.zeros_like(x, dtype=bool)
    for i, start in enumerate(range(0, len(validation)-1, 128)):
        count = min(128, len(validation)-start-1)
        x[i, :count] = validation[start:start+count]
        y[i, :count] = validation[start+1:start+count+1]
        mask[i, :count] = True
    nlls = []
    for start in range(0, blocks, batch_size):
        take = min(batch_size, blocks-start)
        xb = np.zeros((batch_size, 128), np.int32)
        yb = np.zeros_like(xb)
        xb[:take], yb[:take] = x[start:start+take], y[start:start+take]
        losses = np.asarray(score(params, xb, yb, config["n_layer"], config["n_head"], config["n_embd"]))[:take]
        nlls.extend(np.sum(losses * mask[start:start+take], axis=1, dtype=np.float64).tolist())
    counts = mask.sum(axis=1).tolist()
    return dict(**metrics(sum(nlls), sum(counts)), sum_nll=sum(nlls), block_nll=nlls, block_counts=counts, block_tokens=counts,
                evaluation_policy='contiguous-128-reset-context-final-short-block', block_size=128)


def save_evaluation(directory, data='data'):
    _, validation, corpus = prepare(data)
    params, meta = load_model(directory)
    result = evaluate(params, meta['architecture'], validation)
    result.update(corpus_sha256=corpus['corpus_sha256'], checkpoint_sha256=meta['checkpoint_sha256'],
                  vocabulary=corpus['vocabulary'], train_tokens=corpus['train_tokens'], val_tokens=corpus['val_tokens'])
    (Path(directory) / 'eval.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', default='artifacts')
    parser.add_argument('--data-dir', default='data')
    args = parser.parse_args()
    result = save_evaluation(args.artifacts, args.data_dir)
    print({k: result[k] for k in ['loss', 'ppl', 'token_count']})
