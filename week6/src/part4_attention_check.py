"""Week 6 Part 4: verify modern Hopfield retrieval equals JAX softmax attention."""
from pathlib import Path
import json
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence'
ARTIFACTS = ROOT / 'artifacts'
BETAS = (0.003, 0.03, 0.3)
SEED = 2026


def compare_with_attention(patterns, queries, beta):
    """patterns (P,784) stores digit spin rows; queries (Q,784)."""
    X = jnp.asarray(patterns, dtype=jnp.float32)
    Q = jnp.asarray(queries, dtype=jnp.float32)

    scores = beta * (Q @ X.T)
    weights = jax.nn.softmax(scores, axis=-1)
    hopfield_output = weights @ X

    # Explicitly set scale=beta. JAX's default scale is 1/sqrt(784),
    # which is NOT the inverse temperature in this experiment.
    def one_query(q):
        output = jax.nn.dot_product_attention(
            q[None, None, None, :],
            X[None, :, None, :],
            X[None, :, None, :],
            scale=beta,
            implementation='xla',
        )
        return output[0, 0, 0, :]

    attention_output = jax.jit(jax.vmap(one_query))(Q)
    max_difference = float(
        jnp.max(jnp.abs(hopfield_output - attention_output))
    )
    return np.asarray(weights), max_difference


def main():
    from part1_datasets import load_mnist

    EVIDENCE.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)
    images, labels = load_mnist('train')
    # P=100; query each of these 100 stored digits once, as in the sheet.
    patterns = 2.0 * (images[:100].reshape(100, 784) >= 128).astype(np.float32) - 1.0
    queries = patterns.copy()
    rng = np.random.default_rng(SEED)
    queries[:, 14 * 28:] = rng.choice([-1.0, 1.0], size=(100, 14 * 28)).astype(np.float32)

    results = {}
    first_query_weights = {}
    for beta in BETAS:
        weights, max_diff = compare_with_attention(patterns, queries, beta)
        first_query_weights[str(beta)] = weights[0]
        nearest = np.argmax(queries @ patterns.T, axis=1)
        success_nearest = float(np.mean(nearest == np.arange(100)))
        results[str(beta)] = {
            'max_abs_output_difference': max_diff,
            'mean_target_weight': float(np.mean(np.diag(weights))),
            'first_query_target_weight': float(weights[0, 0]),
            'fraction_target_is_nearest': success_nearest,
        }
        print(f'beta={beta:g} | max |Hopfield-attention|={max_diff:.8g} '
              f'| target weight (first query)={weights[0, 0]:.4f}', flush=True)
        if not max_diff < 1e-5:
            raise AssertionError(f'JAX attention mismatch at beta={beta}: {max_diff}')

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, beta in zip(axs, BETAS):
        w = first_query_weights[str(beta)]
        ax.bar(np.arange(len(w)), w, width=1.0)
        ax.set_title(f'beta={beta:g}\ntarget weight={w[0]:.3f}')
        ax.set_xlabel('Stored pattern index')
        ax.set_xlim(-1, 100)
        ax.grid(axis='y', alpha=0.2)
    axs[0].set_ylabel('Softmax weight')
    fig.suptitle('Modern Hopfield: attention weights for the first corrupted query')
    fig.tight_layout()
    figure = EVIDENCE / 'part4_attention_weights.png'
    fig.savefig(figure, dpi=200, bbox_inches='tight')
    plt.close(fig)

    summary = {
        'patterns': 100, 'queries': 100, 'visible_spins': 784,
        'seed': SEED, 'betas': list(BETAS), 'comparison': results,
        'attention_scale': 'beta (explicit, not 1/sqrt(d))',
    }
    (ARTIFACTS / 'part4_attention_check.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8'
    )
    print('All attention comparisons PASSED (< 1e-5).')
    print('Saved:', figure)
    print('Saved:', ARTIFACTS / 'part4_attention_check.json')


if __name__ == '__main__':
    main()
