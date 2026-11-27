# Week 12: Integer programming and learned branching

## Part 1 implementation

Rust `branchopt` uses `highs = 2.4.0` to solve both the LP relaxation and the
mixed-integer model. Only spin bits are integer; bond-disagreement variables
remain continuous. The constant sum(J)+sum(h) is included when reconstructing
physical energies. Presolve is off, one solver thread is used, and both MIP
optimality-gap stopping tolerances are zero. Only Optimal solver statuses pass.

The graph generator implements the specified SplitMix64 with explicit wrapping
arithmetic, lexicographic edge order, per-instance seeds, and fields in {-1,0,1}.
Raw instances, primal solutions and all computed statistics are saved in JSON.
Reference values are assertions only and are never substituted for measurements.

## Reproduction (run from week12/ in WSL/Linux)

```bash
pred create SpinGlass --graph 0-1,1-2,2-3,3-0 --couplings=1,1,1,-1 -o artifacts/pred/square.json
pred solve artifacts/pred/square.json
pred create SpinGlass --random --num-vertices 100 --edge-prob 0.04 --seed 2026 -o artifacts/pred/r100.json
pred solve artifacts/pred/r100.json
cargo install --path . --force
branchopt --mode gap --spins 22 --edges 52 --count 16 --seed 2026 --out artifacts/gap
uv run --no-project --python 3.12 --with matplotlib==3.10.7 python plot_gap.py --run artifacts/gap/run.json --out artifacts/gap.png
```

The bootstrap script records tool versions in artifacts/gap/environment.txt and
Python plotting dependencies in artifacts/gap/plot_environment.json. Cargo.lock
pins resolved Rust dependencies. pred is version 0.6.0. Its caught panic warning
may precede a valid result; successful CLI results are checked explicitly.

The figure is produced solely from the saved Rust run by plot_gap.py. Expected
checks: square LP -4, integer -2, rounded +2; workshop LP 83/3, integer 27,
nearest-rounding infeasible; 16-instance means LP -52, exact -38.25,
gap 13.75, fractional count 22, rounding excess 34.25, 16 suboptimal roundings.
Actual results and timings must come from the local run.

## Files retained locally

- artifacts/pred/: input JSONs, raw solver logs and measured wall time.
- artifacts/gap/run.json: full computed raw data and summaries.
- artifacts/gap.png: energy, integrality-gap and rounding-excess figure.
- artifacts/setup_backups/: existing files backed up before bootstrap writes.

artifacts/ and target/ are ignored by Git. The bootstrap commits only its
explicit source-file list after all Part 1 checks and plotting succeed, leaving
any pre-existing staged changes untouched by that commit. It does not push.
Parts 2-4 are added in the next stages.

## Environment recorded by this local run

```text
2026-11-27T12:59:53+08:00
Linux 6.18.33.2-microsoft-standard-WSL2 x86_64 GNU/Linux
rustc 1.98.1 (48a229cea 2026-09-01)
cargo 1.98.1 (797e8a9bc 2026-08-05)
cmake version 4.2.3

CMake suite maintained and supported by Kitware (kitware.com/cmake).
gcc (Ubuntu 15.2.0-16ubuntu1) 15.2.0
Copyright (C) 2025 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0
Copyright (C) 2025 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

git version 2.53.0
pred 0.6.0
uv 0.12.21 (x86_64-unknown-linux-gnu)
```

<!-- WEEK12_PART2_BEGIN -->
## Part 2: Exact Branch-and-Bound

`src/bnb.rs` builds a fresh continuous HiGHS LP for each visited box and each
Strong Branching trial. The search begins with no incumbent, visits the nearer
child first (1 first on exactly 0.5), and prunes using the LP lower bound.
Integer incumbents are recomputed from the original Hamiltonian. Only fractional
free spin variables may be split, into the exact partition x_i=0 / x_i=1.

Most-frac minimizes distance from 0.5. Strong Branching maximizes the product of
floored child gains. A bound-pruned or infeasible child has gain 1e6. Candidate
indices are traversed in increasing order; ties within 1e-12 numerical tolerance
(relative to max(1, abs(score)) for strong) retain the earlier index.

Trials use cloned box vectors and independent models. They never update the
incumbent, modify the real stack, or cache solutions for later child visits.
Actual LP solve calls = visited nodes + trial solve calls. The oracle is outside
the search timer and is not an input to search. Total time includes recording
overhead; LP time includes setup, solve and extraction for visited and trial LPs.

The workshop trace runs a real maximum-revenue LP at each visited box. It
requires three visited nodes with LP objectives 83/3, 27 and 26.75 and a final
bound prune. It does not replace the trace with hardcoded output.

Run from week12/:

```bash
cargo test --release --locked -- --test-threads=1
cargo install --path . --force --locked
branchopt --mode bnb --spins 22 --edges 52 --count 48 --policy most-frac --seed 2026 --out artifacts/bnb_most
branchopt --mode bnb --spins 22 --edges 52 --count 48 --policy strong --seed 2026 --out artifacts/bnb_strong
uv run --no-project --python 3.12 --with matplotlib==3.10.7 python plot_bnb.py
```

Raw runs, full traces and measured timing are in artifacts/bnb_most/ and
artifacts/bnb_strong/. The figure artifacts/bnb.png independently sorts all
four curves. Input equality, exact energy and solve accounting are validated.
Node-count and timing reference values are not substituted for measurements.
The two meaningful unit tests compare both policies with exhaustive enumeration
on eight small mixed-sign/field graphs and verify the workshop's actual pruning.
The assignment's two 48-instance runs are performed on the user's computer.

Part 1 gap data and gap.png remain available. The upgraded executable still
supports --mode gap. See DEEPSEEK_REVIEW.md for the independent review scope.
<!-- WEEK12_PART2_END -->

<!-- WEEK12_PART3_BEGIN -->
## Part 3: Imitation learning for branching

Run `bash ./week12_part3.sh` from the existing `week12/` directory after Part 2.
The saved `run_part3.sh` is the same self-contained reproduction script.
The script runs the four small Rust checks, installs the updated executable,
exports Strong labels, trains JAX on the CPU, runs the unseen evaluation, plots
the measured evidence, and makes one local commit after all criteria pass.

Strong labels use 48 graphs with n=22, m=52, seed=2026. The optional
`--export-labels` flag defaults to false. Only branch nodes with at least two
candidates are recorded, with a cap of 80 records per graph; the search still
finishes normally. Instance IDs 0..35 are training graphs and 36..47 are
validation graphs. The labels contain the original parent x/z, four row-duals
per bond, fixed-spin box, incumbent, candidate indices, features, gains and scores.
Features are extracted before any Strong child trials and never use the oracle.
Python independently recomputes the twelve features before training.
The complete feature order and formulas are frozen in branchopt.design.toml.

Label collection writes to `artifacts/bnb_labels/`; its dataset is also copied
to the course's canonical `artifacts/bnb_strong/dataset.json`. This keeps the
existing Part 2 run.json, timings and bnb.png available for comparison. The
script verifies that the exported graph IDs, graph data, node counts and LP
solve counts agree with the earlier Strong run before starting training.

The shared 12->48(tanh)->1 MLP uses JAX 0.12.1, CPU float64. Population mean and
std are fitted to real candidate rows from training graphs only; the std
denominator is floored at 1e-6. Padded rows are finite zeros in standardized
coordinates and have zero target and masked probability. Near-best Strong
scores share a uniform target using tolerance 1e-9*max(1,abs(best)). Xavier-normal
weights and zero biases are initialized with JAX PRNGKey(2026). Adam uses
learning rate 0.004, batch size 64, 120 epochs, and 0.0002 times the sum of
squared weights, excluding biases. A seeded NumPy generator shuffles training
nodes each epoch. CE histories exclude regularization. The saved checkpoint
has the lowest validation CE among epoch 1 and every fifth epoch. Test graphs
never select a checkpoint or influence the training statistics.

`artifacts/model.json` contains the best weights, feature statistics, complete
training/validation CE history, six validation probes, graph provenance and
software environment. Rust uses the same w1[input][hidden] matrix layout in
f64, verifies every probe probability to <1e-4, and also records logit error.
Invalid or non-finite features/model outputs stop the run with an error.

Neural branching scores only the current candidate spins. It changes which
spin is split; it does not supply bounds, prune nodes, seed an incumbent, or
solve trial LPs. Splits still partition the box into x_i=0 and x_i=1. The
existing LP lower-bound pruning and feasible integer incumbents preserve the
exact search, independently of model quality. Every test energy is checked
against a separate HiGHS MILP optimum. Most-frac and neural must each perform
exactly one LP solve per explored node.

Training and validation states come from the Strong policy. Neural search
can encounter different states and incumbents, so validation CE alone does not
guarantee improved search. A separate test on 16 unseen graphs (n=22, m=52,
seed=3026) compares most-frac, strong and neural. This test measures model
performance without changing the selected checkpoint. Actual values are saved
even when a performance criterion fails; no reference values replace results.

Reproduce individual steps from `week12/`:

```bash
branchopt --mode bnb --spins 22 --edges 52 --count 48 --seed 2026 --policy strong --export-labels --out artifacts/bnb_labels
cp artifacts/bnb_labels/dataset.json artifacts/bnb_strong/dataset.json
uv run --no-project --python 3.12 --with jax==0.12.1 python train_branch.py --dataset artifacts/bnb_strong/dataset.json --out artifacts/model.json
branchopt --mode eval --spins 22 --edges 52 --count 16 --seed 3026 --model artifacts/model.json --out artifacts/eval_test
uv run --no-project --python 3.12 --with matplotlib==3.10.7 python plot_neural_bnb.py --require-course-settings
```

The figure `artifacts/neural_bnb.png` shows train/validation CE and independently
sorted node-count curves. The raw test run stores each graph, oracle, exact
search trace, node/LP counts, timings and first splits. The printed summary and
`artifacts/eval_test/part3_checks.json` report actual dataset sizes, zero overlap,
best validation CE, probe error, energy error, policy means and the instance
with the largest node saving. Total search time includes trace overhead; LP
time includes model construction, solve and extraction; feature-plus-inference
time measures feature construction and the neural forward pass. MILP oracles,
model loading, probe checks and JSON file writing are outside search timers.

Pre-delivery checks used small development graphs: eight mixed-field graphs
checked against exhaustive enumeration, the workshop's three actual nodes,
a hand-computed feature example, and adversarial-network searches checked
against exhaustive enumeration. An additional n=6 development pipeline checked
label export, all 120 training epochs, JSON model export, six-probe Rust parity,
exact unseen searches and plotting. These are development checks; the course's
n=22 experiments are executed on the user's machine by the reproduction script.
The independent static review and its dispositions are in DEEPSEEK_REVIEW.md.
<!-- WEEK12_PART3_END -->
