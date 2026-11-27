# Week 12: Measured results and presentation notes

All numbers below come from the saved local JSON evidence. Timings are single-run wall-clock measurements on this computer and include search bookkeeping and recording. They are not a statistical hardware benchmark.

## 1. MILP, relaxation and rounding

Let x_i=(1-s_i)/2 and introduce continuous bond variables z_ij in [0,1]. Minimize C-2 sum_i h_i*x_i-2 sum_(i,j) J_ij*z_ij, where C=sum_i h_i+sum_(i,j) J_ij. Each bond obeys z<=x_i+x_j, z<=2-x_i-x_j, z>=x_i-x_j and z>=x_j-x_i. Binary spins force z to their XOR.

Relaxing the spins to continuous [0,1] permits x_i=1/2, so each bond can independently take its favorable z value. This hides inconsistent spin preferences around frustrated cycles. Rounding the spins does not preserve the favorable LP bond values.

On 16 graphs (n=22, m=52, seed=2026):

| Metric | Actual mean |
| --- | ---: |
| LP lower bound | -52.000000 |
| Exact ground energy | -38.250000 |
| Integrality gap | 13.750000 |
| Rounding excess above the optimum | 34.250000 |

The four-spin frustrated square has LP -4, exact energy -2 and nearest-rounded energy +2; the local reference check verifies these values. In the opening workshop maximum problem, the actual search visits the root LP (83/3), the integer x2<=2 child (27), and prunes the x2>=3 child with LP upper bound 26.75.

Figure: [gap.png](artifacts/gap.png).

## 2. The price of Strong Branching

Most-frac chooses the spin closest to 1/2. Strong Branching tests both children of every candidate, using the product of child bound gains. Trials never update the incumbent or seed a cache. A smaller tree can therefore cost many more LP solves.

| Policy | Mean nodes | Mean LP solves | Mean total ms |
| --- | ---: | ---: | ---: |
| most-frac | 50.875 | 50.875 | 19.132 |
| strong | 16.000 | 287.417 | 109.261 |

These values use the original 48-instance Part 2 run files, whose timing evidence was preserved. Figure: [bnb.png](artifacts/bnb.png).

## 3. Imitating Strong without its trial LPs

Training: 276 nodes from 36 graphs. Validation: 84 nodes from 12 disjoint graphs. Instance overlap: 0.
Best saved checkpoint: epoch 85; validation CE without regularization: 1.748563.
Largest six-probe JAX/Rust probability difference: 4.441e-16. Largest test energy error: 2.004e-12.

The shared 12-48(tanh)-1 candidate scorer uses local, size-independent graph and LP features. Training-only mean/std and masked candidate softmax prevent validation statistics and padded candidates from influencing training. Labels are split by graph, not by node. The model changes the branching spin; LP bounds, complete child partitions and feasible incumbents retain responsibility for exactness.

| Policy | Original Part 3 mean nodes | Original mean total ms |
| --- | ---: | ---: |
| most-frac | 50.875 | 19.611 |
| neural | 15.375 | 6.162 |
| strong | 15.625 | 109.182 |

Largest Part 3 saving: instance 4, exact energy -38.000000. Most-frac: 157 nodes, first splits [0, 1]; neural: 23 nodes, first splits [4, 21].

Figure: [neural_bnb.png](artifacts/neural_bnb.png).

## 4. Root caching, time breakdown and size transfer

Root-neural scores all spins once at the root, centers its logits and divides by max(population std,1e-6). It then chooses the candidate maximizing cached_root_score + 3*2*min(x_i,1-x_i). It computes no deeper-node features or network outputs. Neural recomputes features and scores at each branch node. The 22-spin model is reused unchanged on the 30-spin graphs; there is no transfer retraining.

| Graph size / count / seed | Policy | Nodes | LP solves | Total ms | LP ms | Features + inference ms | Speedup vs Strong |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 22 spins / 16 / 3026 | most-frac | 50.875 | 50.875 | 18.206 | 18.056 | 0.0000 | — |
| 22 spins / 16 / 3026 | strong | 15.625 | 287.750 | 105.706 | 105.381 | 0.0000 | — |
| 22 spins / 16 / 3026 | neural | 15.375 | 15.375 | 6.098 | 5.893 | 0.1640 | 17.34x |
| 22 spins / 16 / 3026 | root-neural | 15.500 | 15.500 | 5.876 | 5.809 | 0.0254 | 17.99x |
| 30 spins / 8 / 4026 | most-frac | 177.750 | 177.750 | 85.548 | 84.986 | 0.0000 | — |
| 30 spins / 8 / 4026 | strong | 38.000 | 899.250 | 446.221 | 445.065 | 0.0000 | — |
| 30 spins / 8 / 4026 | neural | 39.250 | 39.250 | 21.482 | 20.818 | 0.5205 | 20.77x |
| 30 spins / 8 / 4026 | root-neural | 52.750 | 52.750 | 27.664 | 27.396 | 0.0357 | 16.13x |

Largest energy error across both sizes and all four policies: 2.004e-12. Every non-Strong rule has LP solves equal to explored nodes. Root scoring is checked to run exactly once per graph, and its candidate choice is independently recomputed from the saved cache at every branch.

The time comparison includes feature and network cost. Model loading, the six probes, separate HiGHS MILP oracles and final JSON writing are outside each search timer. LP time includes constructing, solving and extracting both visited LPs and Strong trial LPs. Root cached-score arithmetic is counted as tree management. The plotted remainder also includes trace recording.

Both learned policies must reduce mean nodes on both sizes. The script also checks at least 2x speedup over Strong, a conservative numerical interpretation of the learning sheet's qualitative timing requirement. Actual ratios are shown above; no reference values are substituted.

Figure: [eval.png](artifacts/eval.png).

## Reproduction, environment and evidence

Run the saved stages in order from week12/:

```bash
bash run_part1.sh
bash run_part2.sh
bash run_part3.sh
bash run_part4.sh
```

Model SHA-256 before and after both evaluations: `4464c0e1638154fa3a5644306d001009a9dac3d59972b18cbdefb2679639cbd6`.

```json
{
  "recorded_at": "2026-11-27T15:49:58.186454+08:00",
  "platform": "Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43",
  "cpu_model": "Intel(R) Core(TM) Ultra X7 358H",
  "rustc": "rustc 1.98.1 (48a229cea 2026-09-01)",
  "cargo": "cargo 1.98.1 (797e8a9bc 2026-08-05)",
  "cmake": "cmake version 4.2.3",
  "gcc": "gcc (Ubuntu 15.2.0-16ubuntu1) 15.2.0",
  "uv": "uv 0.12.21 (x86_64-unknown-linux-gnu)",
  "summary_python": "3.14.4",
  "training_plot_python_requested": "3.12 via uv",
  "training_environment": {
    "jax": "0.12.1",
    "numpy": "2.5.5",
    "platform": "Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43",
    "backend": "cpu",
    "x64": true
  },
  "solver_crates": {
    "highs": "2.4.0",
    "highs-sys": "1.16.0"
  },
  "solver_options": {
    "presolve": "off",
    "threads": 1,
    "mip_rel_gap": 0.0,
    "mip_abs_gap": 0.0
  },
  "model_sha256": "4464c0e1638154fa3a5644306d001009a9dac3d59972b18cbdefb2679639cbd6"
}
```

Keep the four figures, raw run folders, pred files and model.json under artifacts/ on this computer. Source, both design files, reproduction/plot scripts, README.md, this results sheet and the independent review log belong in Git. The artifact directory is ignored.

All formal statistics in this document are produced by the local reproduction runs. DEEPSEEK_REVIEW.md records the actual static-review scope and the response to each material comment.
