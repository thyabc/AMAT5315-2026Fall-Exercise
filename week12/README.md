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
