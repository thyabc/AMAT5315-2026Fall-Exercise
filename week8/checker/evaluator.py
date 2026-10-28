"""Independent numeric decoder. Never imports or executes student model code."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import erf, logsumexp

from _vendor.data import CORPUS_SHA256, _load_corpus_bytes

BLOCK = 128
PARAM_LIMIT = 10_000_000


def corpus():
    text = _load_corpus_bytes().decode('utf-8')
    vocabulary = sorted(set(text))
    lookup = {c: i for i, c in enumerate(vocabulary)}
    ids = np.fromiter((lookup[c] for c in text), dtype=np.int64)
    split = int(0.9 * len(ids))
    return ids[:split], ids[split:], vocabulary


def numeric_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name] for name in archive.files}
    if not result or any(a.dtype.kind not in 'fiu' or not np.all(np.isfinite(a))
                         for a in result.values()):
        raise ValueError(f'{path}: arrays must be finite real numeric values')
    return result


def shapes(config):
    d, layers = config['n_embd'], config['n_layer']
    expected = {'token_embedding': (65, d), 'position_embedding': (BLOCK, d),
                'final_norm.scale': (d,), 'final_norm.bias': (d,),
                'output.weight': (d, 65), 'output.bias': (65,)}
    for i in range(layers):
        prefix = f'blocks.{i}.'
        for norm in ('norm1', 'norm2'):
            expected[prefix + norm + '.scale'] = (d,)
            expected[prefix + norm + '.bias'] = (d,)
        for name, width_in, width_out in (
            ('attention.qkv', d, 3*d), ('attention.output', d, d),
            ('mlp.up', d, 4*d), ('mlp.down', 4*d, d)):
            expected[prefix + name + '.weight'] = (width_in, width_out)
            expected[prefix + name + '.bias'] = (width_out,)
    return expected


def load_model(directory, vocabulary):
    directory = Path(directory)
    meta = json.loads((directory / 'model.json').read_text())
    config = meta['architecture']
    if meta['schema_version'] != 1 or meta['corpus_sha256'] != CORPUS_SHA256:
        raise ValueError('model metadata schema/corpus mismatch')
    if meta['vocabulary'] != vocabulary:
        raise ValueError('vocabulary must be the sorted pinned corpus characters')
    for name, low, high in [('n_embd', 4, 512), ('n_layer', 1, 16), ('n_head', 1, 16)]:
        if type(config[name]) is not int or not low <= config[name] <= high:
            raise ValueError(f'invalid architecture {name}')
    if (config['vocab_size'] != 65 or config['block_size'] != BLOCK
            or config['n_embd'] % config['n_head'] or config['gelu'] != 'exact'
            or config['layer_norm_epsilon'] != 1e-5 or config['dropout'] != 0):
        raise ValueError('architecture differs from canonical decoder contract')
    if type(meta['step']) is not int or meta['step'] < 0:
        raise ValueError('checkpoint step must be a nonnegative integer')
    digest = hashlib.sha256((directory / 'model.npz').read_bytes()).hexdigest()
    if meta['checkpoint_sha256'] != digest:
        raise ValueError('checkpoint SHA-256 mismatch')
    expected = shapes(config)
    if sum(np.prod(shape) for shape in expected.values()) >= PARAM_LIMIT:
        raise ValueError(f'parameter count must be below {PARAM_LIMIT}')
    arrays = numeric_npz(directory / 'model.npz')
    if arrays.keys() != expected.keys():
        raise ValueError('model parameter names differ from canonical export')
    for name, shape in expected.items():
        if arrays[name].shape != shape or arrays[name].dtype.kind != 'f':
            raise ValueError(f'{name}: expected float shape {shape}')
    return {k: v.astype(np.float64) for k, v in arrays.items()}, config, meta


def forward(params, config, ids, causal=True):
    """Float64 NumPy forward; exact GELU and inclusive lower-triangular mask."""
    batch, length = ids.shape
    width, heads = config['n_embd'], config['n_head']
    if not 1 <= length <= BLOCK:
        raise ValueError('input length outside 1..128')
    x = params['token_embedding'][ids] + params['position_embedding'][:length]

    def norm(x, prefix):
        centered = x - x.mean(axis=-1, keepdims=True)
        return (centered / np.sqrt(np.mean(centered**2, axis=-1, keepdims=True) + 1e-5)
                * params[prefix + '.scale'] + params[prefix + '.bias'])

    def linear(x, prefix):
        return x @ params[prefix + '.weight'] + params[prefix + '.bias']

    attention = None
    for i in range(config['n_layer']):
        prefix = f'blocks.{i}.'
        qkv = linear(norm(x, prefix + 'norm1'), prefix + 'attention.qkv')
        q, k, v = (a.reshape(batch, length, heads, width // heads).transpose(0, 2, 1, 3)
                   for a in np.split(qkv, 3, axis=-1))
        scores = q @ k.swapaxes(-1, -2) / np.sqrt(width // heads)
        if causal:
            scores = np.where(np.tri(length, dtype=bool), scores, -np.inf)
        attention = np.exp(scores - logsumexp(scores, axis=-1, keepdims=True))
        attended = (attention @ v).transpose(0, 2, 1, 3).reshape(batch, length, width)
        x = x + linear(attended, prefix + 'attention.output')
        hidden = linear(norm(x, prefix + 'norm2'), prefix + 'mlp.up')
        hidden = 0.5 * hidden * (1 + erf(hidden / np.sqrt(2)))
        x = x + linear(hidden, prefix + 'mlp.down')
    return linear(norm(x, 'final_norm'), 'output'), attention


def evaluate(params, config, ids, target_shift=1):
    """Each validation character after the first is scored once, including the tail."""
    starts = list(range(0, len(ids) - 1, BLOCK))
    block_nll, block_tokens = [], []
    for start in range(0, len(starts), 8):
        batch_starts = starts[start:start + 8]
        counts = [min(BLOCK, len(ids) - 1 - s) for s in batch_starts]
        x = np.zeros((len(counts), max(counts)), dtype=np.int64)
        for b, (s, n) in enumerate(zip(batch_starts, counts)):
            x[b, :n] = ids[s:s+n]
        logits, _ = forward(params, config, x)
        log_probs = logits - logsumexp(logits, axis=-1, keepdims=True)
        for b, (s, n) in enumerate(zip(batch_starts, counts)):
            targets = ids[s+target_shift:s+target_shift+n]
            block_nll.append(float(-log_probs[b, np.arange(n), targets].sum()))
            block_tokens.append(n)
    total = sum(block_tokens)
    loss = sum(block_nll) / total
    return dict(sum_nll=sum(block_nll), token_count=total, loss=loss,
                ppl=float(np.exp(loss)), block_nll=block_nll, block_tokens=block_tokens)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', default='artifacts', help='directory containing model.npz and model.json')
    parser.add_argument('--out', type=Path, help='optional path for the independently computed JSON report')
    args = parser.parse_args()
    try:
        train, validation, vocabulary = corpus()
        params, config, meta = load_model(args.artifacts, vocabulary)
        report = evaluate(params, config, validation)
        report.update(corpus_sha256=CORPUS_SHA256, checkpoint_sha256=meta['checkpoint_sha256'],
                      vocabulary=vocabulary, train_tokens=len(train), val_tokens=len(validation),
                      evaluation_policy='contiguous-128-reset-context-final-short-block')
        if args.out:
            args.out.write_text(json.dumps(report, indent=2)+'\n')
        print(f"loss={report['loss']:.9f}, PPL={report['ppl']:.9f}, tokens={report['token_count']}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'FAIL: schema: {exc}\n')
