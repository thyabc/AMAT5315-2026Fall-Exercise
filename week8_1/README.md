# Week 8 — Language models and the transformer

<!-- BEGIN WEEK8 PART1 -->
## Part 1: From tokens to vectors (embeddings)

Run from repository root:

```bash
uv run --no-project --with numpy --with matplotlib --with safetensors --with huggingface_hub --with tokenizers python week8_1/part1_embeddings.py
```

Source: `minishlab/potion-base-8M`, revision `bf8b056`. Model: `(29528, 256)`. Control: shuffled embedding rows, seed `1730`.

Word vectors were L2-normalized, and each pair's offset was compared by cosine with the mean of the other eight pair offsets (leave-one-out).

| Relation | Original mean | Shuffled mean | Acceptance |
|---|---:|---:|---|
| country -> capital | 0.4659 | 0.0295 | PASS |
| male -> female | 0.5485 | -0.0148 | PASS |
| adjective -> comparative | 0.4330 | 0.0088 | PASS |

Acceptance: original means >= 0.3; shuffled means between -0.15 and 0.15.

Evidence: `evidence/part1_embedding_directions.png`

Raw pair results: `artifacts/part1_pair_cosines.csv`; summary/settings: `artifacts/part1_summary.json` (both ignored by git).
<!-- END WEEK8 PART1 -->

<!-- BEGIN WEEK8 PART2 -->
## Part 2: Count models — Markov chains and n-grams

Run from `week8_1/` (with `HF_ENDPOINT=https://hf-mirror.com` if needed):

```bash
uv run --no-project --with numpy --with matplotlib --with scipy --with sentencepiece --with huggingface_hub python -u part2_ngrams.py
```

Stories: 24867 train / 2763 held-out; 512 BPE tokens; 2.0518 characters/token. The first 90% of training tokens tune add-k smoothing (k in 0.0001, 0.001, 0.01, 0.1, 1); each model is then refit on all training tokens. The final 10% of whole stories is reserved for held-out evaluation.

| n | Chosen k | Nats/token | Perplexity | Bits/character | Unseen fraction |
|---|---:|---:|---:|---:|---:|
| 1 | 0.0001 | 4.8394 | 126.39 | 3.4027 | 0.00% |
| 2 | 0.01 | 3.0382 | 20.87 | 2.1362 | 0.03% |
| 3 | 0.01 | 2.0943 | 8.12 | 1.4726 | 0.63% |
| 4 | 0.01 | 1.8628 | 6.44 | 1.3098 | 4.15% |
| 5 | 0.01 | 2.0825 | 8.02 | 1.4643 | 12.22% |
| 6 | 0.001 | 2.5504 | 12.81 | 1.7933 | 23.50% |

Uniform: 6.23832 nats/token; perplexity 512; 4.3863 bits/character.

Bigram exact normalization (lengths 1–3) maximum absolute error: 3.66e-15. 250k-token pooled bigram sampler chi-squared p-value: 0.58035.

Evidence: `evidence/part2_ngram_loss_unseen.png`. Raw runs: `artifacts/part2_summary.json`, `artifacts/part2_ngram_metrics.csv` and the cached token arrays.
<!-- END WEEK8 PART2 -->

<!-- BEGIN WEEK8 PART3 -->
## Part 3 — The MLP: a memory in the weights

Re-run inside `week8_1/` using Python/uv and CPU JAX:
```bash
uv run --no-project --with numpy --with matplotlib --with 'jax[cpu]' --with sentencepiece --with sentencepiece python -u part3_mlp.py --only memory
uv run --no-project --with numpy --with matplotlib --with 'jax[cpu]' --with sentencepiece --with sentencepiece python -u part3_mlp.py --only mlp
```

**Memory stores:** Three random draws for each d=16,32,64 and M=8,...,65536 (powers of two). For M>1024 only a random subset of stored query keys is evaluated against all stored memories, an explicit CPU-saving approximation to the sheet's exhaustive query requirement. Results at smaller M query all keys. Interference R is the norm of the leakage from other memories divided by signal.

| Store | d=64 capacity | Scaling exponent |
|---|---:|---:|
| Outer product | 68.3 | 1.071 |
| Least squares | 128.9 | 1.005 |
| ReLU | 141.0 | 0.907 |
| ReLU^2 | 2814.3 | 1.868 |
| ReLU^3 | 38945.3 | 2.808 |

Evidence: `evidence/part3_memory_interference.png`, `evidence/part3_capacity_scaling.png`.

**One-token MLP:** 512-token vocabulary, embedding 64, tanh hidden layer 512; 4000 AdamW steps, batch size 16384, learning rate peak 0.006, cosine decay and warmup.
Held-out loss: **3.04098** nats/token, Bigram baseline: **3.03819**, difference: +0.00279. Threshold ±0.01: PASS.
Evidence: `evidence/part3_mlp_training_curve.png`.

Raw metrics are in `artifacts/part3_*.json` (not for Git).
<!-- END WEEK8 PART3 -->
