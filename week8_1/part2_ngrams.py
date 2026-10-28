"""AMAT5315 Week 8 Part 2: Markov chains and n-gram count language models.

Install/run inside week8_1 via:
  uv run --no-project --with numpy --with matplotlib --with scipy --with sentencepiece --with huggingface_hub python -u part2_ngrams.py

CPU-only. Token arrays, downloads, and raw reports go to artifacts/ and data/;
evidence figures, this source, and README are suitable for git.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from scipy.stats import chi2

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
ARTIFACTS = BASE / 'artifacts'
EVIDENCE = BASE / 'evidence'
VOCAB_SIZE = 512
TOKEN_BITS = 9  # 512 == 2^9; 6-grams fit inside 54 bits / int64
SMOOTHING_K = [0.0001, 0.001, 0.01, 0.1, 1.0]
SEED = 1730
CORPUS_REPO = 'roneneldan/TinyStories'
CORPUS_REV = 'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64'
CORPUS_FILE = 'TinyStoriesV2-GPT4-valid.txt'
TOKENIZER_REPO = 'karpathy/tinyllamas'
TOKENIZER_REV = '0bd21da'
TOKENIZER_FILE = 'stories260K/tok512.model'


def log(*args):
    print(*args, flush=True)


def download_inputs():
    log('Downloading TinyStories pinned revision and Karpathy tok512.model...')
    DATA.mkdir(parents=True, exist_ok=True)
    corpus = hf_hub_download(repo_id=CORPUS_REPO, repo_type='dataset',
                             filename=CORPUS_FILE, revision=CORPUS_REV,
                             local_dir=str(DATA))
    model = hf_hub_download(repo_id=TOKENIZER_REPO, filename=TOKENIZER_FILE,
                            revision=TOKENIZER_REV, local_dir=str(DATA))
    log('  Corpus:', corpus)
    log('  BPE model:', model)
    return Path(corpus), Path(model)


def prepare_tokens(text_file, model_file):
    sp = spm.SentencePieceProcessor(model_file=str(model_file))
    vocab = sp.get_piece_size()
    eos = sp.eos_id()
    log(f'SentencePiece vocabulary: {vocab}; EOS token ID: {eos}')
    if vocab != VOCAB_SIZE or not (0 <= eos < VOCAB_SIZE):
        raise RuntimeError('Tokenizer does not have the required 512 tokens and valid EOS ID.')

    text = text_file.read_text(encoding='utf-8')
    stories = [s.strip() for s in text.split('<|endoftext|>') if s.strip()]
    n_train = int(len(stories) * 0.9)
    train_stories = stories[:n_train]
    val_stories = stories[n_train:]
    log(f'Story count: {len(stories)}; train: {len(train_stories)}; held-out: {len(val_stories)}')
    if (len(stories), n_train, len(val_stories)) != (27630, 24867, 2763):
        raise RuntimeError('Story count differs from PDF reference (27,630 total / 24,867 train / 2,763 validation).')

    def encode_set(story_set, label):
        # uint16 is sufficient for 512-token vocabulary and uses 2 bytes/token.
        seqs = []
        for idx, s in enumerate(story_set):
            ids = sp.encode(s, out_type=int)
            seqs.append(np.asarray(ids + [eos], dtype=np.uint16))
            if (idx + 1) % 5000 == 0:
                log(f'  Tokenized {label}: {idx + 1:,}/{len(story_set):,} stories')
        return np.concatenate(seqs)

    tokens_train = encode_set(train_stories, 'train')
    tokens_val = encode_set(val_stories, 'validation')
    val_chars = sum(len(s) for s in val_stories)
    chars_per_token = val_chars / len(tokens_val)
    normalized_mismatches = sum(sp.decode(sp.encode(s, out_type=int)) != s for s in val_stories)
    log(f'Tokens: train={len(tokens_train):,}; validation={len(tokens_val):,}')
    log(f'Validation characters per token: {chars_per_token:.5f}')
    log(f'Encode/decode story mismatches: {normalized_mismatches} (PDF reference: 3)')

    np.save(ARTIFACTS / 'part2_train_tokens.npy', tokens_train)
    np.save(ARTIFACTS / 'part2_val_tokens.npy', tokens_val)
    return tokens_train, tokens_val, eos, val_chars, normalized_mismatches


def pack_ngrams(tokens, n, eos):
    """Pack all n-grams ending at each token, with EOS left padding.

    For n<=6 and V=512, the packed uint64 key is exactly collision-free.
    """
    seq = np.asarray(tokens, dtype=np.int64)
    if n > 1:
        seq = np.pad(seq, (n - 1, 0), mode='constant', constant_values=eos)
    m = len(tokens)
    keys = np.zeros(m, dtype=np.int64)
    for offset in range(n):
        keys <<= TOKEN_BITS
        keys |= seq[offset:offset+m]
    return keys


@dataclass
class CountModel:
    n: int
    unique_grams: np.ndarray
    gram_counts: np.ndarray
    unique_contexts: np.ndarray
    ctx_counts: np.ndarray

    @classmethod
    def fit(cls, tokens, n, eos):
        keys = pack_ngrams(tokens, n, eos)
        u, counts = np.unique(keys, return_counts=True)
        del keys
        context = u >> TOKEN_BITS
        unique_contexts, indexes = np.unique(context, return_index=True)
        context_counts = np.add.reduceat(counts, indexes)
        del context, indexes
        return cls(n, u, counts, unique_contexts, context_counts)

    @staticmethod
    def _lookup(sorted_ids, values, keys):
        indexes = np.searchsorted(sorted_ids, keys)
        valid = indexes < len(sorted_ids)
        safe = np.minimum(indexes, len(sorted_ids)-1)
        valid &= sorted_ids[safe] == keys
        out = np.zeros(len(keys), dtype=np.int64)
        out[valid] = values[safe[valid]]
        return out

    def lookup(self, packed_grams):
        c_ngram = self._lookup(self.unique_grams, self.gram_counts, packed_grams)
        c_context = self._lookup(self.unique_contexts, self.ctx_counts, packed_grams >> TOKEN_BITS)
        return c_ngram, c_context


def loss_from_counts(cngram, ccontext, k):
    return float(np.mean(np.log(ccontext + k*VOCAB_SIZE) - np.log(cngram + k)))


def bigram_matrix(model, k):
    if model.n != 2:
        raise ValueError('Requires bigram model')
    P = np.full((VOCAB_SIZE, VOCAB_SIZE), k, dtype=np.float64)
    rows = np.asarray(model.unique_grams >> TOKEN_BITS, dtype=np.int64)
    cols = np.asarray(model.unique_grams & (VOCAB_SIZE-1), dtype=np.int64)
    P[rows, cols] += model.gram_counts
    P /= P.sum(axis=1, keepdims=True)
    return P


def explicit_normalization(P, eos):
    # Enumerate 512^1, 512^2, and 512^3 possible continuations from EOS.
    # For length 3, 512^3 contributions are evaluated in chunks without
    # allocating a large 3D array.
    p0 = P[eos]
    total1 = float(p0.sum())
    total2 = float(np.sum(p0[:, None] * P))
    total3 = 0.0
    for x1 in range(VOCAB_SIZE):
        total3 += float(np.sum((p0[x1] * P[x1, :, None]) * P))
    totals = [total1, total2, total3]
    return {'length_1': total1, 'length_2': total2, 'length_3': total3,
            'max_abs_error': max(abs(x-1.0) for x in totals)}


def sample_chi_square(P, eos, train_tokens, n_draws=250_000):
    rng = np.random.default_rng(SEED)
    cdf = np.cumsum(P, axis=1)
    cdf[:, -1] = 1.0
    draws = np.empty(n_draws, dtype=np.uint16)
    prevs = np.empty(n_draws, dtype=np.uint16)
    previous = int(eos)
    for i in range(n_draws):
        prevs[i] = previous
        nxt = int(np.searchsorted(cdf[previous], rng.random(), side='right'))
        draws[i] = nxt
        previous = nxt

    top_contexts = np.argsort(np.bincount(train_tokens, minlength=VOCAB_SIZE))[-20:][::-1]
    chi2_stat = 0.0
    df = 0
    per_context = []
    for a in top_contexts:
        mask = (prevs == a)
        n = int(mask.sum())
        if n < 1:
            continue
        observed = np.bincount(draws[mask], minlength=VOCAB_SIZE).astype(np.float64)
        expected = n * P[a]
        frequent = expected >= 5.0
        obs = list(observed[frequent])
        exp = list(expected[frequent])
        if np.any(~frequent):
            obs_rare = float(observed[~frequent].sum())
            exp_rare = float(expected[~frequent].sum())
            if exp_rare >= 5.0:
                obs.append(obs_rare)
                exp.append(exp_rare)
            else:
                # Rare group of <5 is merged with the smallest remaining expected bin.
                if exp:
                    j = int(np.argmin(exp))
                    obs[j] += obs_rare
                    exp[j] += exp_rare
                else:
                    obs = [obs_rare]
                    exp = [exp_rare]
        if len(exp) < 2:
            continue
        obs = np.asarray(obs)
        exp = np.asarray(exp)
        stat = float(np.sum((obs-exp)**2 / exp))
        chi2_stat += stat
        df += len(exp) - 1
        per_context.append({'context_id': int(a), 'sample_occurrences': n,
                            'chi2': stat, 'degrees_of_freedom': len(exp)-1})
    p_value = float(chi2.sf(chi2_stat, df)) if df > 0 else float('nan')
    return {'n_generated': n_draws, 'contexts': 20, 'statistic': chi2_stat,
            'degrees_of_freedom': df, 'p_value': p_value,
            'seed': SEED, 'per_context': per_context}


def plot_results(results):
    orders = [r['n'] for r in results]
    bpcs = [r['bits_per_character'] for r in results]
    unseens = [100*r['unseen_fraction'] for r in results]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.6))
    ax1.plot(orders, bpcs, 'o-', lw=1.8)
    for x, y in zip(orders, bpcs):
        ax1.annotate(f'{y:.3f}', xy=(x,y), xytext=(0,9),
                     textcoords='offset points', ha='center', fontsize=8)
    ax1.set(xlabel='n (n-gram order)', ylabel='Held-out bits per character',
            title='Held-out loss')
    ax1.set_xticks(orders)
    ax1.grid(alpha=.25)
    ax2.plot(orders, unseens, 'o-', lw=1.8, color='#8667bd')
    for x, y in zip(orders, unseens):
        ax2.annotate(f'{y:.1f}%', xy=(x,y), xytext=(0,9),
                     textcoords='offset points', ha='center', fontsize=8)
    ax2.set(xlabel='n (n-gram order)', ylabel='Unseen held-out n-grams (%)',
            title='The curse of dimensionality')
    ax2.set_xticks(orders)
    ax2.set_ylim(bottom=-0.5)
    ax2.grid(alpha=.25)
    fig.suptitle('Week 8 Part 2 — n-gram models on TinyStories', fontsize=13)
    fig.tight_layout()
    target = EVIDENCE / 'part2_ngram_loss_unseen.png'
    fig.savefig(target, dpi=190, bbox_inches='tight')
    plt.close(fig)
    return target


def update_readme(summary, chart_path):
    rows = summary['n_grams']
    table = '\n'.join(
        f"| {r['n']} | {r['best_k']} | {r['nats_per_token']:.4f} | {r['perplexity']:.2f} | {r['bits_per_character']:.4f} | {100*r['unseen_fraction']:.2f}% |"
        for r in rows
    )
    intro = (
        '## Part 2: Count models — Markov chains and n-grams\n\n'
        'Run from `week8_1/` (with `HF_ENDPOINT=https://hf-mirror.com` if needed):\n\n'
        '```bash\n'
        'uv run --no-project --with numpy --with matplotlib --with scipy '
        '--with sentencepiece --with huggingface_hub python -u part2_ngrams.py\n'
        '```\n\n'
        f"Stories: {summary['train_stories']} train / {summary['validation_stories']} held-out; "
        f"512 BPE tokens; {summary['characters_per_token']:.4f} characters/token. "
        'The first 90% of training tokens tune add-k smoothing (k in 0.0001, 0.001, '
        '0.01, 0.1, 1); each model is then refit on all training tokens. '
        'The final 10% of whole stories is reserved for held-out evaluation.\n\n'
        '| n | Chosen k | Nats/token | Perplexity | Bits/character | Unseen fraction |\n'
        '|---|---:|---:|---:|---:|---:|\n' + table + '\n\n'
        f"Uniform: {summary['uniform']['nats_per_token']:.5f} nats/token; "
        f"perplexity {summary['uniform']['perplexity']:.0f}; "
        f"{summary['uniform']['bits_per_character']:.4f} bits/character.\n\n"
        f"Bigram exact normalization (lengths 1–3) maximum absolute error: "
        f"{summary['normalization']['max_abs_error']:.3g}. "
        f"250k-token pooled bigram sampler chi-squared p-value: "
        f"{summary['sampling']['p_value']:.5f}.\n\n"
        f"Evidence: `evidence/{chart_path.name}`. Raw runs: `artifacts/part2_summary.json`, "
        '`artifacts/part2_ngram_metrics.csv` and the cached token arrays.\n'
    )
    start, end = '<!-- BEGIN WEEK8 PART2 -->', '<!-- END WEEK8 PART2 -->'
    readme = BASE / 'README.md'
    old = readme.read_text(encoding='utf-8') if readme.exists() else '# Week 8 — Language models and the transformer\n'
    chunk = start + '\n' + intro + end
    if start in old and end in old:
        before, tail = old.split(start, 1)
        _, after = tail.split(end, 1)
        content = before + chunk + after
    else:
        content = old.rstrip() + '\n\n' + chunk + '\n'
    readme.write_text(content, encoding='utf-8')
    return readme


def main():
    for path in (DATA, ARTIFACTS, EVIDENCE):
        path.mkdir(parents=True, exist_ok=True)
    corpus_file, tokenizer_file = download_inputs()
    train, valid, eos, val_chars, mismatches = prepare_tokens(corpus_file, tokenizer_file)
    chars_per_token = val_chars / len(valid)
    uniform_nll = math.log(VOCAB_SIZE)
    summary = {
        'corpus': {'repo': CORPUS_REPO, 'revision': CORPUS_REV, 'filename': CORPUS_FILE},
        'tokenizer': {'repo': TOKENIZER_REPO, 'revision': TOKENIZER_REV, 'filename': TOKENIZER_FILE},
        'vocabulary_size': VOCAB_SIZE,
        'train_stories': 24867, 'validation_stories': 2763,
        'train_tokens': int(len(train)), 'validation_tokens': int(len(valid)),
        'validation_characters': int(val_chars), 'characters_per_token': chars_per_token,
        'decode_mismatches': mismatches,
        'uniform': {'nats_per_token': uniform_nll, 'perplexity': float(VOCAB_SIZE),
                    'bits_per_character': uniform_nll*len(valid)/(math.log(2)*val_chars)},
        'n_grams': [],
        'seed': SEED,
    }
    tune_cut = int(len(train)*0.9)
    train_fit = train[:tune_cut]
    train_tune = train[tune_cut:]
    log(f'Add-k tuning: first {len(train_fit):,} training tokens fit; last {len(train_tune):,} tune.')

    bigram_model = None
    bigram_k = None
    for n in range(1, 7):
        log(f'\n=== n={n}: count/train-tuning models ===')
        fit_model = CountModel.fit(train_fit, n, eos)
        tune_keys = pack_ngrams(train_tune, n, eos)
        counts, contexts = fit_model.lookup(tune_keys)
        del tune_keys, fit_model
        candidates = {str(k): loss_from_counts(counts, contexts, k) for k in SMOOTHING_K}
        best_k = min(SMOOTHING_K, key=lambda k: candidates[str(k)])
        del counts, contexts
        log(f'  Tuning nats/token: {candidates}; chosen k={best_k}')

        final_model = CountModel.fit(train, n, eos)
        val_keys = pack_ngrams(valid, n, eos)
        counts, contexts = final_model.lookup(val_keys)
        unseen = float(np.mean(counts == 0))
        nll = loss_from_counts(counts, contexts, best_k)
        ppl = math.exp(nll)
        bpc = nll*len(valid)/(val_chars*math.log(2))
        result = {'n': n, 'best_k': best_k, 'nats_per_token': nll,
                  'perplexity': ppl, 'bits_per_character': bpc,
                  'unseen_fraction': unseen,
                  'unique_training_ngrams': int(len(final_model.unique_grams)),
                  'tuning_nats_per_token': candidates}
        summary['n_grams'].append(result)
        log(f'  Held-out: nats/token={nll:.4f}; ppl={ppl:.3f}; '
            f'bits/char={bpc:.4f}; unseen={100*unseen:.2f}%')
        del val_keys, counts, contexts
        if n == 2:
            bigram_model = final_model
            bigram_k = best_k
        else:
            del final_model
        gc.collect()

    log('\n=== Bigram normalization and ancestral-sampling checks ===')
    if bigram_model is None:
        raise RuntimeError('Bigram model missing')
    P = bigram_matrix(bigram_model, bigram_k)
    normalization = explicit_normalization(P, eos)
    log('  Probability sums for sequence lengths 1,2,3:', normalization)
    sampling = sample_chi_square(P, eos, train)
    log('  250,000 bigram tokens: pooled chi-square p-value=', f"{sampling['p_value']:.5f}")
    del bigram_model, P
    summary['normalization'] = normalization
    summary['sampling'] = sampling

    chart = plot_results(summary['n_grams'])
    with (ARTIFACTS / 'part2_ngram_metrics.csv').open('w', encoding='utf-8', newline='') as handle:
        columns = ['n', 'best_k', 'nats_per_token', 'perplexity',
                   'bits_per_character', 'unseen_fraction', 'unique_training_ngrams']
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in summary['n_grams']:
            writer.writerow({column: item[column] for column in columns})

    ref = {1:4.839, 2:3.038, 3:2.094}
    bpc_ref = {4:1.310, 5:1.460, 6:1.790}
    accept = {
        'split': summary['train_stories'] == 24867 and summary['validation_stories'] == 2763,
        'characters_per_token': abs(chars_per_token - 2.05) <= 0.06,
        'normalization': normalization['max_abs_error'] < 1e-12,
        'sampling': sampling['p_value'] > .001,
        'nats_baselines': all(abs(summary['n_grams'][n-1]['nats_per_token'] - value) < .03 for n,value in ref.items()),
        'n4_n5_n6_bpc': all(abs(summary['n_grams'][n-1]['bits_per_character'] - value) < .03 for n,value in bpc_ref.items()),
        'lowest_at_n4': int(np.argmin([r['bits_per_character'] for r in summary['n_grams']])) == 3,
        'unseen_fractions': all(abs(100*summary['n_grams'][n-1]['unseen_fraction'] - value) < 2.0
                                for n,value in {3:0.6,4:4.2,5:12.2,6:23.5}.items()),
    }
    summary['checks'] = accept
    summary['overall_pass'] = all(accept.values())
    file_json = ARTIFACTS / 'part2_summary.json'
    file_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    readme = update_readme(summary, chart)

    log('\n=== Acceptance checks ===')
    for name, accepted in accept.items():
        log(f"  {name:25s} {'PASS' if accepted else 'CHECK'}")
    log('Evidence figure:', chart)
    log('Raw summary:', file_json)
    log('README:', readme)
    log('Overall:', 'PASS' if summary['overall_pass'] else 'CHECK — share the output for diagnosis; do not alter values')


if __name__ == '__main__':
    main()
