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
