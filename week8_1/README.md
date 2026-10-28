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

## Part 4: Attention — recall from context-built memory

Reproduce from `week8_1/`:

```bash
uv run --no-project --with numpy --with matplotlib python part4_attention.py
```

In d=64 with β=8 and seed=1730, store a 32×32 binary character
(+1 ink, -1 background) among M=64/256/1024/4096 memories.
Compare outer-product, ReLU, ReLU² and softmax stores.
Normalize each recalled image by the target memory's own weight;
R is the L2 norm of interference divided by L2 norm of target.

- Main evidence: `evidence/part4_attention_recall.png` (4×4 dot panels)
- Original picture: `evidence/part4_original_picture.png`
- Raw results: `artifacts/part4_recall_ratios.csv` and `artifacts/part4_summary.json`
- Implementation: `part4_attention.py`

The character in the supplied PDF is not machine-readable (missing glyph);
this experiment uses a reproducible, manually drawn 32×32 亮-like glyph.
The image's shape does not affect the theoretical R comparisons.
The color of each panel is red when R < 1, gray otherwise.

<!-- BEGIN WEEK8 PART5 -->
## Part 5 — Transformer: training and generating

See `part5_transformer.py` for exact model and tests.

```bash
uv run --no-project --with numpy --with matplotlib --with "jax[cpu]" --with sentencepiece --with scipy python -u part5_transformer.py --stage train
uv run --no-project --with numpy --with matplotlib --with "jax[cpu]" --with sentencepiece --with scipy python -u part5_transformer.py --stage verify
```

Validation: **1.20632 bits/character** vs 4-gram **1.30975**. Last 8 context positions: 1.64998 nats/token; trigram 2.09434.

Evidence:
- `evidence/part5_transformer_training.png`
- `evidence/part5_context_loss.png`
- `evidence/part5_transformer_vs_counts.png`
- `evidence/part5_samples_and_tests.txt`

Raw checkpoint and full metrics: `artifacts/part5_*`. Sampling: 100,000 draws per distribution; surprise/entropy: 128 independent sequences of 128 tokens per temperature.

Acceptance checks:
- beats_best_count_model: PASS
- context_last_8_beats_trigram: PASS
- sampler_chi_square: PASS
- greedy_0.0001: PASS
- entropy_temperature_1: PASS
- surprise_temperature_0.7: PASS
<!-- END WEEK8 PART5 -->

<!-- BEGIN WEEK8 PART6 -->
## Part 6 — Scaling laws: how to spend a compute budget

Dataset: Epoch AI's Chinchilla Figure 4 reconstruction, fixed revision `92258837425e1b5f2851d624287f0120583a3d0e`:
`https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla/92258837425e1b5f2851d624287f0120583a3d0e/data/svg_extracted_data.csv`. Data file `data/chinchilla-runs.csv` has 245 rows, and the 5 highest-loss
runs are excluded (240 fitted runs). Keep `data/` and `artifacts/` out of Git.

Reproduce inside the `week8_1/` folder:

```bash
uv run --no-project --with numpy --with scipy --with matplotlib python -u part6_scaling.py --bootstrap 400
```

Fit `L(N,D)=E+A/N^alpha+B/D^beta`, with `D=C/(6N)`, via Huber loss
(`delta=0.001`) applied to log-loss residuals; fit L-BFGS-B from a grid,
then bootstrap 400 times (seed 1730) for standard errors and a 95% interval.

| Parameter | Refit | Bootstrap standard error |
|---|---:|---:|
| E | 1.8172 | 0.02536 |
| A | 477.83 | 118.8 |
| B | 2143.4 | 1129 |
| alpha | 0.34731 | 0.01501 |
| beta | 0.36717 | 0.0197 |

- Figure 19, fit and measured-vs-predicted: `evidence/part6_chinchilla_fit.png`.
- Figure 20, residual comparison: `evidence/part6_chinchilla_residuals.png`.
- Figure 21, optimal allocation: `evidence/part6_chinchilla_allocation.png`.
- The mean log-loss residual for our refit is -0.00066; for printed parameters +0.01674.
- At 5.76e23 FLOP, our `D*/N*` is 17.921 with 95% bootstrap interval [7.51, 37.13].
- Raw numeric results are `artifacts/part6_summary.json` and `artifacts/part6_optimal_allocation.csv`.
<!-- END WEEK8 PART6 -->
