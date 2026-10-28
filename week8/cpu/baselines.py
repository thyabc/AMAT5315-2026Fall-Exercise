"""Part 1: uniform, training-only add-one unigram and add-one bigram baselines.

Pure NumPy. Every validation character except the first is scored exactly once.
A bigram's context is the previous character, which every target has (the block
input v[s] supplies it), so block resets do not change its score.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm import corpus, record, save_arrays, VOCAB  # noqa: E402


def metrics(summed_nll, count):
    loss = summed_nll / count
    return dict(token_count=int(count), summed_nll=float(summed_nll), loss=loss,
                ppl=math.exp(loss), bpc=loss / math.log(2))


def unigram_probabilities(training):
    counts = np.bincount(training, minlength=VOCAB).astype(np.float64)
    return counts, (counts + 1) / (counts.sum() + VOCAB)


def bigram_probabilities(training):
    counts = np.zeros((VOCAB, VOCAB))
    np.add.at(counts, (training[:-1], training[1:]), 1)
    return counts, (counts + 1) / (counts.sum(axis=1, keepdims=True) + VOCAB)


def baselines():
    training, validation, meta = corpus()
    targets = validation[1:]
    contexts = validation[:-1]
    n = len(targets)
    uni_counts, uni = unigram_probabilities(training)
    bi_counts, bi = bigram_probabilities(training)
    result = dict(
        corpus_sha256=meta['corpus_sha256'], vocabulary=meta['vocabulary'],
        train_tokens=int(len(training)), val_tokens=int(len(validation)),
        scored_tokens=int(n),
        uniform=metrics(n * math.log(VOCAB), n),
        unigram=metrics(float(-np.log(uni[targets]).sum()), n),
        bigram=metrics(float(-np.log(bi[contexts, targets]).sum()), n),
        two_target_example=metrics(-math.log(0.5) - math.log(0.25), 2),
        unigram_training_loss=float(-np.log(uni[training[1:]]).mean()),
        bigram_training_loss=float(-np.log(bi[training[:-1], training[1:]]).mean()),
        unseen_bigrams_in_validation=int(np.sum(bi_counts[contexts, targets] == 0)),
    )
    return result, uni, bi


if __name__ == '__main__':
    result, uni, bi = baselines()
    record('part1', {k: v for k, v in result.items() if k != 'vocabulary'})
    save_arrays(part1_unigram=uni, part1_bigram=bi)
    for name in ('uniform', 'unigram', 'bigram'):
        print(f"{name:8s} loss={result[name]['loss']:.6f} PPL={result[name]['ppl']:.6f} "
              f"BPC={result[name]['bpc']:.4f} tokens={result[name]['token_count']}")
