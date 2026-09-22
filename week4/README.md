# Week 4 — Continuum Fluid Dynamics

This directory contains the Week 4 implementation of explicit time
integrators, periodic Fourier differentiation, and a two-dimensional
incompressible vorticity solver on the periodic domain
\([0,2\pi)^2\).

The implementation includes:

- Forward Euler, explicit midpoint (RK2), and classical RK4 behind a
  common `Integrator` interface.
- A one-dimensional periodic advection-diffusion test problem using
  Fourier derivatives and centred finite differences.
- A two-dimensional Fourier pseudospectral vorticity-streamfunction
  solver.
- The two-thirds dealiasing rule applied to both the carried vorticity
  and nonlinear products.
- Taylor-Green and seeded random initial fields.
- Stability, sensitivity, order, and time-step refinement experiments.

Generated run data are stored under `artifacts/`.  The committed
verification figures and numerical summary are stored under
`evidence/`.

## Build

From the `week4/` directory:

```bash
cargo build --release --bins
```

Run the Rust tests:

```bash
cargo test
```

The two command-line tools used by the contract runs are:

```text
target/release/field
target/release/fluid
```

## Taylor-Green contract run

Generate and integrate the Taylor-Green field with the contract
parameters:

```bash
mkdir -p artifacts/taylor-green

target/release/field taylor-green --n 64 \
| target/release/fluid \
  --method rk4 \
  --nu 0.1 \
  --dt 0.01 \
  --t-end 1 \
  --every 0.1 \
  --out artifacts/taylor-green \
> artifacts/taylor-green.tsv
```

The expected initial values are

```text
E(0) = 0.250000
Z(0) = 0.500000
```

and at `t = 1` the run gives approximately

```text
E(1) = 0.167580
Z(1) = 0.335160
```

Generate the exact Taylor-Green field at `t = 1`:

```bash
target/release/field taylor-green \
  --n 64 \
  --nu 0.1 \
  --t 1 \
> artifacts/taylor-green/exact-t1.json
```

Generate the Taylor-Green evidence figure:

```bash
python3 scripts/taylor_green.py
```

Writes:

```text
evidence/taylor-green.png
```

## Random-flow contract run

Generate the seeded random initial field and integrate it with RK4:

```bash
mkdir -p artifacts/random

target/release/field random \
  --n 128 \
  --seed 2026 \
  --k-min 2 \
  --k-max 6 \
| target/release/fluid \
  --method rk4 \
  --nu 0.004 \
  --dt 0.01 \
  --t-end 10 \
  --every 0.1 \
  --out artifacts/random \
> artifacts/random.tsv
```

Generate the four-frame random-flow figure:

```bash
python3 scripts/random_flow.py
```

Writes:

```text
evidence/random.png
```

## Part 1: line stability

Generate the one-dimensional stability data:

```bash
cargo run --release --bin line_stability_data
```

Generate the figure:

```bash
python3 scripts/line_stability.py
```

Writes:

```text
evidence/line-stability.png
```

The figure compares the measured RK4 stability region with the
stability boundaries and shows Gaussian-pulse runs at `dt = 0.045`
and `dt = 0.056`.

## Part 1: line accuracy

Generate the pulse-race and time-order data:

```bash
cargo run --release --bin line_accuracy_data
```

Generate the figure:

```bash
python3 scripts/line_accuracy.py
```

Writes:

```text
evidence/line-accuracy.png
```

The fitted temporal orders are approximately first order for Euler,
second order for midpoint, fourth order for classical RK4, and second
order for the deliberately equal-weight RK4 control.

## Part 2: derivative verification

Run the two-dimensional derivative comparison:

```bash
cargo run --release --bin derivative_check
```

The Fourier derivatives of

```text
g(x,y) = sin(3x) cos(2y)
```

are accurate to roundoff, while the centred finite-difference errors
decrease by approximately a factor of four when the grid is refined
from `n = 32` to `n = 64`.

## Part 3: stability runs

Create the scan directory:

```bash
mkdir -p artifacts/scan
```

Taylor-Green below the measured diffusive stability boundary:

```bash
target/release/field taylor-green --n 64 \
| target/release/fluid \
  --method rk4 \
  --nu 0.1 \
  --dt 0.032 \
  --t-end 8 \
  --every 0.5 \
  --out artifacts/scan/tg-0032 \
> artifacts/scan/tg-0032.tsv
```

Taylor-Green above the measured diffusive stability boundary:

```bash
target/release/field taylor-green --n 64 \
| target/release/fluid \
  --method rk4 \
  --nu 0.1 \
  --dt 0.033 \
  --t-end 8 \
  --every 0.5 \
  --out artifacts/scan/tg-0033 \
> artifacts/scan/tg-0033.tsv
```

For the seeded random field, the measured RK4 stability boundary is
bracketed by `dt = 0.030` and `dt = 0.032`.

Stable random RK4 run:

```bash
target/release/field random \
  --n 128 \
  --seed 2026 \
  --k-min 2 \
  --k-max 6 \
| target/release/fluid \
  --method rk4 \
  --nu 0.004 \
  --dt 0.030 \
  --t-end 10 \
  --every 0.5 \
  --out artifacts/scan/random-rk4-0030 \
> artifacts/scan/random-rk4-0030.tsv
```

Unstable random RK4 run:

```bash
target/release/field random \
  --n 128 \
  --seed 2026 \
  --k-min 2 \
  --k-max 6 \
| target/release/fluid \
  --method rk4 \
  --nu 0.004 \
  --dt 0.032 \
  --t-end 10 \
  --every 0.5 \
  --out artifacts/scan/random-rk4-0032 \
> artifacts/scan/random-rk4-0032.tsv
```

Euler random run:

```bash
target/release/field random \
  --n 128 \
  --seed 2026 \
  --k-min 2 \
  --k-max 6 \
| target/release/fluid \
  --method euler \
  --nu 0.004 \
  --dt 0.01 \
  --t-end 10 \
  --every 0.1 \
  --out artifacts/scan/random-euler-001 \
> artifacts/scan/random-euler-001.tsv
```

Generate the stability figure:

```bash
python3 scripts/blowup.py
```

Writes:

```text
evidence/blowup.png
```

## Part 3: sensitivity

The sensitivity study uses RK4 with `dt = 0.01` to `t = 20` and
compares each flow with a copy whose initial vorticity contains a
small Fourier-mode perturbation.

The perturbation tool is built with:

```bash
cargo build --release --bin perturb_field
```

After generating the base and perturbed Taylor-Green and random runs
under

```text
artifacts/sensitivity/
```

generate the figure with:

```bash
python3 scripts/sensitivity.py
```

Writes:

```text
evidence/sensitivity.png
```

The random pair separates strongly, while the Taylor-Green pair
decays toward the six-decimal storage floor.

## Part 4: RK4 order

Run Taylor-Green at

```text
n = 8
nu = 0.5
t_end = 2
dt = 0.4, 0.25, 0.2
```

and store the runs under:

```text
artifacts/order/rk4-dt0.4/
artifacts/order/rk4-dt0.25/
artifacts/order/rk4-dt0.2/
```

Generate the exact field at `t = 2`:

```bash
target/release/field taylor-green \
  --n 8 \
  --nu 0.5 \
  --t 2 \
> artifacts/order/exact-t2.json
```

Generate the order figure:

```bash
python3 scripts/order.py
```

Writes:

```text
evidence/order.png
```

The measured RK4 slope is approximately:

```text
4.10
```

## Part 4: random-flow time-step refinement

Use the same `n = 128`, seed `2026`, `k` band `[2,6]`, and
`nu = 0.004` for every run.

The tested time steps are:

```text
0.0200
0.0125
0.0100
```

The same-grid reference uses:

```text
dt = 0.0025
```

The runs are stored under:

```text
artifacts/convergence/dt0.02/
artifacts/convergence/dt0.0125/
artifacts/convergence/dt0.01/
artifacts/convergence/reference/
```

Analyse the retained final vorticity fields with:

```bash
python3 scripts/convergence.py
```

Writes:

```text
evidence/convergence.json
evidence/convergence.png
```

The measured convergence slope is approximately fourth order.  The
time-step selection recorded in `convergence.json` is produced from
the measured errors and the fourth-order Richardson estimate.

## Required unstable checker run

The checker also reads the following deliberately unstable
Taylor-Green run:

```bash
mkdir -p artifacts/unstable/taylor-green

target/release/field taylor-green --n 64 \
| target/release/fluid \
  --method rk4 \
  --nu 0.1 \
  --dt 0.04 \
  --t-end 4 \
  --every 0.1 \
  --out artifacts/unstable/taylor-green \
> artifacts/unstable/taylor-green.tsv
```

The run is expected to terminate after the energy becomes non-finite.

## Evidence files

The committed `evidence/` directory contains:

```text
line-stability.png
line-accuracy.png
taylor-green.png
blowup.png
sensitivity.png
random.png
order.png
convergence.png
convergence.json
```

The `artifacts/` directory contains intermediate and raw run data and
is intentionally not committed.

## Checker

After generating the contract, order, and unstable runs, run:

```bash
SEED=2026 python3 checker/check .
```

A successful submission ends with:

```text
PASS
```
