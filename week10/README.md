# Week 10 --- Approximate Tensor Contraction for the 2D Ising Model

This repository implements three parts of the Week 10 exercise:

1.  SVD compression of a Gaussian random matrix.
2.  Open-boundary contraction with a boundary MPS.
3.  Periodic tensor renormalization group (TRG) contraction.

The model is the zero-field square-lattice Ising model with spins
(s_i=`\pm1`{=tex}), coupling (K=1/T), and \[
Z(K)=`\sum`{=tex}*{{s}}`\exp`{=tex}`\left`{=tex}(K`\sum`{=tex}*{`\langle `{=tex}i,j`\rangle`{=tex}}s_is_j`\right`{=tex}),`\qquad`{=tex}
g(K)=`\frac{\log Z(K)}{L^2}`{=tex}. \] Specific heat per spin is
estimated from the curvature of (g(K)): \[
c_V(K;h)`\approx `{=tex}K\^2`\frac{g(K+h)-2g(K)+g(K-h)}{h^2}`{=tex}. \]
The exact square-lattice critical temperature used for comparison is \[
T_c\^{`\mathrm{Onsager}`{=tex}}=`\frac{2}{\log(1+\sqrt2)}`{=tex}`\approx2.269185314213`{=tex}.
\]

## Repository contents

-   `src/main.rs`: Part 1, seeded Gaussian matrix generation, SVD,
    reconstructions, independent error checks, and `artifacts/svd.png`.
-   `src/bin/thermo.rs`: command-line entry point for the Ising
    tensor-network calculations.
-   `src/bin/thermo/mps.rs`: MPS representation and canonical SVD
    compression.
-   `src/bin/thermo/boundary.rs`: open-boundary row absorption and exact
    small-system checks.
-   `src/bin/thermo/trg.rs`: periodic TRG, split-factor evidence,
    normalization scales, and exact small-system checks.
-   `thermo.design.toml`: design specification for the `thermo` command.
-   `scripts/plot_trg_coarse.py`: creates a TRG figure from
    already-saved coarse-scan CSV files; it does **not** launch TRG
    calculations.
-   `runs/`: generated numerical evidence (kept locally, not committed).
-   `artifacts/`: generated figures (kept locally, not committed).

## Build and tests

From the `week10/` directory:

``` bash
cargo fmt
cargo test
cargo build --release --bins
```

The latest local terminal run after adding the TRG tests reported **15
passed, 0 failed** for `cargo test --bin thermo`. Run `cargo test` to
execute tests for all binaries as well.

## Part 1 --- SVD compression

Run:

``` bash
cargo run --release
```

The program generates a (512`\times512`{=tex}) matrix with independent
standard-normal entries using `ChaCha8Rng` and seed `2026`. It saves the
original matrix, singular values, and reconstructions at ranks
(`\chi=16`{=tex},64,128,256,512) under `runs/svd/`.

For every retained rank, the program reloads the saved matrix and
reconstruction, computes the relative Frobenius error, and compares it
with the discarded-singular-value prediction. It stops with an error if
the difference is at least (10\^{-8}), or if the full-rank
reconstruction error is at least (10\^{-8}).

The factor-storage count is
(512`\chi`{=tex}+`\chi512`{=tex}=1024`\chi`{=tex}), compared with
(512\^2) entries in the original matrix. The break-even rank is
therefore (`\chi=256`{=tex}). In the plotted result, the 10% error
threshold is reached only at a rank above this break-even point
(approximately (`\chi=395`{=tex})); thus this matrix does not achieve
both 10% relative error and storage reduction at the same time.

Output: - `runs/svd/original_matrix.bin` -
`runs/svd/singular_values.bin` - `runs/svd/reconstruction_chi_*.bin` -
`runs/svd/summary.csv` - `runs/svd/metadata.txt` - `artifacts/svd.png`

## Part 2 --- Boundary MPS on open squares

The boundary-MPS implementation absorbs rows of the open square and
compresses the boundary representation using canonical SVD sweeps.
Missing bonds at open edges are closed with the boundary vector
specified in the learning sheet.

The recorded peak analysis used (L=16,32,64), coarse spacing (h=0.0025),
and refined samples with spacing (h=0.00125) at (`\chi=16`{=tex},32).
The three-size and largest-two extrapolations were performed using
(T\_{`\mathrm{peak}`{=tex}}(L)=T_c+a/L).

  Quantity                                                             Result
  ---------------------------------- ----------------------------------------
  Onsager (T_c)                                                2.269185314213
  (T_c), all three sizes                                       2.267506445602
  (T_c), largest two sizes                                     2.268763780635
  Relative error, all sizes                                       0.07398552%
  Relative error, largest two                                     0.01857643%
  (c_V), 4×4 finite difference                                 0.439393072554
  (c_V), 4×4 exact energy variance                             0.439396912160
  Absolute (c_V) difference            (3.839606051159`\times10`{=tex}\^{-6})

The 4×4 specific-heat check passed. The recorded analysis-script time
was 89.042 s.

Output: - `runs/critical/`: coarse and refined (g(K)) data and
small-system checks. - `artifacts/boundary-critical.png`

## Part 3 --- Periodic TRG

The periodic TRG implementation performs two diagonal SVD splits and
regroups four half-tensors into one coarse-grained tensor. Before each
step, the tensor is divided by its maximum absolute entry and the
corresponding scale contribution is accumulated in (`\log `{=tex}Z). For
a periodic square of power-of-two side length (L), the implementation
uses (2`\log`{=tex}*2 L) steps. The final periodic closure is \[
z=`\sum`{=tex}*{i,j}A\_{ijij}, \] with the two sums independent.

A completed (L=1024), (K=0.44), (`\chi=16`{=tex}) run used 20 TRG steps,
ended with tensor bond dimension 16, and took approximately 145.869 s.
The recorded values were
(`\log `{=tex}Z`\approx9.738376577832154`{=tex}`\times10`{=tex}\^5) and
(g`\approx0.9287239625770716`{=tex}).

### Coarse peak scan --- preliminary result only

The saved coarse peak table contains the following (`\chi=16`{=tex}),
(h=0.0025) estimates:

  ----------------------------------------------------------------------------------------------------------------
              \(L\)   (K\_{`\mathrm{peak}`{=tex}})   (T\_{`\mathrm{peak}`{=tex}})   (c\_{V,`\mathrm{peak}`{=tex}})
  ----------------- ------------------------------ ------------------------------ --------------------------------
                 64                         0.4375                       2.285714                         2.251706

                256                         0.4400                       2.272727                         2.824852

               1024                         0.4400                       2.272727                         2.948088
  ----------------------------------------------------------------------------------------------------------------

A linear fit of these coarse peak temperatures against (1/L) gives: -
All three sizes: (T_c`\approx2.2705627706`{=tex}), relative error
approximately 0.0607%. - Largest two sizes:
(T_c`\approx2.2727272727`{=tex}), relative error approximately 0.1561%.

These are **coarse (`\chi=16`{=tex}) estimates**, not the final refined
result. The (L=256) and (L=1024) grid maxima both fall at (K=0.44), so
the largest-two fit is especially limited by the coarse grid resolution.

The requested fine scan (nine-point stencils at (h=0.00125), for both
(`\chi=16`{=tex}) and (`\chi=32`{=tex})) was intentionally not run.
Consequently, rank dependence, step-size dependence, and the final
(`\chi=32`{=tex}) TRG extrapolation are not available. The coarse result
should be described as preliminary and should not be presented as
satisfying those checks.

To plot the already-saved coarse data without running any new TRG
calculation:

``` bash
python3 scripts/plot_trg_coarse.py \
  --data-dir runs/trg_scan \
  --peak-file coarse_peak_estimates.csv \
  --output artifacts/trg-critical.png
```

The script reads `coarse_L64_chi16.csv`, `coarse_L256_chi16.csv`, and
`coarse_L1024_chi16.csv` when present. Each file should contain columns
`L,K,chi,g,log_Z,solver_seconds`. If the raw (g(K)) files are
unavailable, the script can still plot the saved peak-only table, but it
cannot reconstruct the full (c_V(T)) curves from peak points alone.

## Reproduction and verification commands

``` bash
# SVD experiment
cargo run --release

# One representative open-boundary MPS calculation
cargo run --release --bin thermo -- \
  --method boundary --l 64 --k 0.45 --chi 16 --out runs/boundary64

# One representative periodic TRG calculation
cargo run --release --bin thermo -- \
  --method trg --l 1024 --k 0.44 --chi 16 --out runs/trg1024

# Recreate the TRG figure from existing CSV evidence only
python3 scripts/plot_trg_coarse.py \
  --data-dir runs/trg_scan \
  --peak-file coarse_peak_estimates.csv \
  --output artifacts/trg-critical.png
```

Record the exact environment before submission:

``` bash
uname -a
rustc --version
cargo --version
python3 --version
```

## Git and generated evidence

The learning sheet asks that generated data, tensors, normalization
scales, figures, and archives remain local rather than being committed.
Commit the source, design specification, analysis scripts, README, and
report:

``` bash
git status --short
git add Cargo.toml Cargo.lock src README.md thermo.design.toml scripts .gitignore WEEK10_REPORT.md
git commit -m "Complete Week 10 tensor contraction exercise"
git push
```

Check `git status --short` before committing to ensure `runs/`,
generated figures, binary tensors, and archive files are not staged.
