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
