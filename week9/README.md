# AMAT5315 Fall 2026 — Week 9

## Scope

This project implements the Week 9 tensor-network inference exercises:

- ASIA exact inference and validation.
- Random and OMECO Greedy contraction-order planning, replay, and complexity recounting.
- Loopy sum-product belief propagation (BP), including fixed-point residual and marginal-error traces.
- Promedus_14 exact inference with evidence, Greedy and seeded time-only TreeSA plans, and comparison against the published PR/MAR files.
- Promedus_14 loopy BP with the same settings used for ASIA.

## Environment

- WSL Ubuntu on Windows
- Rust / Cargo: `1.96.0`
- `omeco = 0.2.6`
- `tenferro-* = 0.7.1`
- CPU backend: `cpu-faer`
- Python 3 with `matplotlib` for plots

The Rust dependencies are pinned in `Cargo.toml` and `Cargo.lock`.

## ASIA reproduction

The existing ASIA implementation is kept separate from the Promedus implementation.

```bash
cargo +1.96.0 run --release --bin infer
cargo +1.96.0 run --release --bin plan_orders
cargo +1.96.0 run --release --bin replay_orders
cargo +1.96.0 run --release --bin bp_asia
python3 plot_orders.py
python3 plot_bp.py
python3 plot_asia.py
```

Main ASIA evidence:

- `artifacts/prior/run.json`
- `artifacts/AD/run.json`
- `artifacts/ADX/run.json`
- `runs/random/` and `runs/greedy/`
- `artifacts/bp-prior/run.json`
- `artifacts/bp-AD/run.json`
- `artifacts/bp-ADX/run.json`
- `evidence/orders.png`
- `evidence/bp_traces.png`
- `evidence/graphs.svg` (AD tuberculosis-query factor graph)
- `evidence/posteriors.png` (prior, AD and ADX disease probabilities; enumeration vs contraction)
- `evidence/joint.png` (T-L joint, product of marginals and dependence)

## ASIA AD contraction-order comparison

The saved random and Greedy plans both replay to the enumerated evidence mass
`0.004501375` (relative error `1.93e-16`).

| Plan | Work | Largest generated tensor entries |
|---|---:|---:|
| Seeded random (seed 42) | 248 | 32 |
| OMECO Greedy | 70 | 8 |

The exact costs can differ from the reference plan because the random generator
and Greedy tie-breaking may produce different valid trees.

## Measured ASIA results

Exact inference was checked against full enumeration for the prior, AD and ADX
observation cases. State 0 means yes/present/positive.

| Case | Evidence mass | P(T=0) | P(L=0) | P(B=0) | Max marginal error |
|---|---:|---:|---:|---:|---:|
| Prior | 1.000000000 | 0.010400000 | 0.055000000 | 0.450000000 | 2.22e-16 |
| AD (A=0,D=0) | 0.004501375 | 0.087750965 | 0.099525145 | 0.811402072 | 2.22e-16 |
| ADX (A=0,D=0,X=0) | 0.00098822675 | 0.391711720 | 0.444270508 | 0.628821776 | 1.67e-16 |

For the ADX case, the exact T-L joint probability is retained separately from
the product of its marginals. The joint-minus-product matrix is the dependence
term; it must not be replaced by an independence approximation.

The ADX BP run converged in 98 sweeps at residual `7.89e-11`, while its maximum
marginal error remained `0.01061`. This is the intended demonstration that a
small fixed-point residual does not guarantee exact marginals on a loopy graph.

## ASIA visual evidence

Generate the ASIA diagram and probability figures from the saved enumeration and
contraction outputs:

```bash
python3 plot_asia.py
```

The script writes:

- `evidence/graphs.svg`: the eight-factor AD tuberculosis-query network, with A=0
  and D=0 observed and T left open.
- `evidence/posteriors.png`: state-0 disease probabilities for prior, AD and ADX,
  overlaying enumeration and tensor contraction.
- `evidence/joint.png`: the T-L joint posterior, product of marginals, and their
  difference (dependence) for each evidence case.

The ASIA UAI scope order and factor labels are retained in the diagram. State 0
means yes/present/positive, as defined in the learning sheet.

## Promedus_14 data

The model, evidence, published PR and published MAR files are stored in `promedus/`:

- `promedus/Promedus_14.uai`
- `promedus/Promedus_14.uai.evid`
- `promedus/Promedus_14.uai.PR`
- `promedus/Promedus_14.uai.MAR`

Pinned source:

```text
https://raw.githubusercontent.com/mroavi/uai-2014-inference-competition/22005844423f5bca62798548df808cc418fc0d7d/uai2014/PR/Promedus_14.uai
https://raw.githubusercontent.com/mroavi/uai-2014-inference-competition/22005844423f5bca62798548df808cc418fc0d7d/uai2014/PR/Promedus_14.uai.evid
https://raw.githubusercontent.com/mroavi/uai-2014-inference-competition/22005844423f5bca62798548df808cc418fc0d7d/uai2014/PR/Promedus_14.uai.PR
https://raw.githubusercontent.com/mroavi/uai-2014-inference-competition/22005844423f5bca62798548df808cc418fc0d7d/uai2014/MAR/Promedus_14.uai.MAR
```

The model has 414 binary variables and 414 factors. The evidence file contains nine observed variables.

## Promedus_14 reproduction

Run exact inference first, then BP:

```bash
cargo +1.96.0 run --release --bin promedus -- all
```

Or run the stages independently:

```bash
cargo +1.96.0 run --release --bin promedus -- exact
cargo +1.96.0 run --release --bin promedus -- bp
```

Generate the figures after both stages finish:

```bash
python3 plot_promedus.py
```

### Exact inference settings

- Evidence is applied by fixing observed coordinates in every factor table.
- Factors that become scalars are retained as scalar multipliers.
- UAI row-major factor tables are converted to the tensor backend's column-major storage.
- OMECO plans use integer variable labels; this avoids the single-character label limit for a 414-variable model.
- Greedy uses `omeco::GreedyMethod::default()`.
- TreeSA uses `ScoreFunction::time_optimized()`, one trial, 50 iterations per beta level, and the default 300-level beta schedule.
- In omeco 0.2.6, the TreeSA implementation seeds each trial with `SmallRng::seed_from_u64(trial_index + 42)`. With one trial, the recorded seed is 42.
- Every input factor and intermediate tensor is divided by its maximum absolute entry. The logarithm of each removed multiplier is accumulated and restored when computing `log10(Z_E)`.
- Exact one-variable marginals are computed with the requested variable left open, for IDs 59, 102, and 139. Both Greedy and TreeSA query plans are run and compared.

Expected published comparisons from the assignment sheet:

- Published `log10(Z_E)` is approximately `-7.806746304582435`; the PR file is rounded to `-7.80675`.
- State-1 MAR values for IDs 59, 102, and 139 are approximately `0.1975882582`, `0.1497534323`, and `0.6175220184`.
- Compare Greedy and TreeSA exact log-partition results with absolute tolerance `1e-10`.
- Compare published PR with tolerance `1e-4`.
- Compare published MAR with tolerance `1e-5`.

Exact outputs:

- `runs/promedus/greedy/order.json`
- `runs/promedus/greedy/complexity.json`
- `runs/promedus/treesa/order.json`
- `runs/promedus/treesa/complexity.json`
- `runs/promedus/query-59-greedy/` and `runs/promedus/query-59-treesa/`
- `runs/promedus/query-102-greedy/` and `runs/promedus/query-102-treesa/`
- `runs/promedus/query-139-greedy/` and `runs/promedus/query-139-treesa/`
- `runs/promedus/settings.json`
- `artifacts/promedus/exact.json`
- `artifacts/promedus/marginal_59.json`
- `artifacts/promedus/marginal_102.json`
- `artifacts/promedus/marginal_139.json`

### Promedus BP settings

BP uses the ASIA settings:

- Damping: `0.5`
- Undamped residual tolerance: `1e-10`
- Maximum sweeps: `1000`
- Uniform initial messages
- Synchronous sum-product updates
- Evidence applied by slicing observed states into factor tables

BP is judged by both its undamped fixed-point residual and its selected-query marginal error. A small residual is not evidence that the approximate marginals are accurate. If the final residual remains above `1e-10`, report the run as unconverged.

BP output:

- `artifacts/bp-promedus/run.json`

Plots:

- `evidence/promedus_orders.png`
- `evidence/promedus_bp.png`

## Final checks before submission

```bash
cargo +1.96.0 check --all-targets
cargo +1.96.0 run --release --bin infer
cargo +1.96.0 run --release --bin plan_orders
cargo +1.96.0 run --release --bin replay_orders
cargo +1.96.0 run --release --bin bp_asia
cargo +1.96.0 run --release --bin promedus -- all
python3 plot_orders.py
python3 plot_bp.py
python3 plot_asia.py
python3 plot_promedus.py
git status --short
```

Review the generated JSON files and plots before committing. Do not claim BP converged unless the final **undamped** residual is below `1e-10`.

## Git

After verifying all outputs:

```bash
git add Cargo.toml Cargo.lock README.md src plot_asia.py plot_promedus.py \
  promedus runs artifacts evidence
git commit -m "Complete Week 9 tensor inference and Promedus benchmark"
git push
```


## Actual Promedus_14 results

### Measured exact inference

| Method | Work | Largest tensor entries | Planning (s) | Contraction (s) | log10(Z_E) |
|---|---:|---:|---:|---:|---:|
| Greedy | 224845616 | 2097152 | 0.0320 | 0.7366 | -7.806746304582 |
| TreeSA | 10525886 | 131072 | 1.3228 | 0.1454 | -7.806746304582 |

- Published PR: `-7.806750000000`
- PR absolute error: `3.695418e-06`
- Greedy/TreeSA logZ difference: `1.776357e-15`
- Maximum selected MAR error: `4.323192e-07`

### Measured selected marginals

| Variable | Greedy state 0 | Greedy state 1 | TreeSA state 0 | TreeSA state 1 | Reference state 1 |
|---|---:|---:|---:|---:|---:|
| 102 | 0.8502465677 | 0.1497534323 | 0.8502465677 | 0.1497534323 | 0.1497530000 |
| 139 | 0.3824779816 | 0.6175220184 | 0.3824779816 | 0.6175220184 | 0.6175220000 |
| 59 | 0.8024117418 | 0.1975882582 | 0.8024117418 | 0.1975882582 | 0.1975880000 |

### Measured Promedus BP

- Converged: `False`
- Sweeps: `1000`
- Damping: `0.5`
- Undamped residual tolerance: `1e-10`
- Final undamped residual: `2.116264941066e-10`
- Cumulative update work: `22424000`
- Runtime: `0.1592 s`

| Variable | BP state 1 | Exact state 1 | Absolute error |
|---|---:|---:|---:|
| 102 | 0.0002507605 | 0.1497534323 | 1.495027e-01 |
| 139 | 0.9998749358 | 0.6175220184 | 3.823529e-01 |
| 59 | 0.0000856180 | 0.1975882582 | 1.975026e-01 |

BP did not converge within 1000 sweeps. Its final undamped residual remained above the stopping tolerance, and its selected marginals were inaccurate relative to exact inference. This agrees with the qualitative behavior reported in the Week 9 benchmark.

### Evidence figures

- `evidence/promedus_orders.png`
- `evidence/promedus_bp.png`
