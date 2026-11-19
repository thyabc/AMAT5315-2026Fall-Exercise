# AMAT5315 Week 11 — Hard Problems, Reductions, and Tensor-Network Sampling

This project implements the computational work for Week 11. It connects
logic-gate gadgets, spin-glass reductions, tensor-network contraction
and sampling, and the N-Queens problem.

The work covered in this repository is organized as follows:

- **Part 2 — Gates:** enumerate the truth-table energies of seven
  logic-gate gadgets and the 16 states of an AND–NOT–SET1 circuit.
- **Part 3 — Factor 15:** contract a SpinGlass reduction of the
  factoring problem, calculate its ground state and partition function,
  sample configurations, decode a ground-state sample, and compare
  greedy and TreeSA contraction costs.
- **Part 4 — N-Queens:** encode the N-Queens constraints directly as a
  sparse 0/1 tensor network, count exact solutions, and sample valid
  boards.

Part 1 is the preparation step used to create the Factor 15 reduction
bundle for Part 3.

------------------------------------------------------------------------

## 1. Environment and settings

The following environment information was recorded from the project
terminal.

| Item                          | Version / setting                     |
|:------------------------------|:--------------------------------------|
| Rust                          | `rustc 1.98.1 (48a229cea 2026-09-01)` |
| Cargo                         | `cargo 1.98.1 (797e8a9bc 2026-08-05)` |
| `pred`                        | `0.6.0`                               |
| `omeco`                       | `0.2.6`                               |
| `tnsample`                    | `0.1.0`                               |
| Random seed                   | `2026`                                |
| Boltzmann inverse temperature | `β = 2.0`                             |
| SpinGlass contraction order   | `treesa`                              |
| Factor 15 samples             | `2000`                                |
| N-Queens samples              | `200` for each of `N = 6` and `N = 8` |

Python plotting scripts are run with **`python3`**. On the WSL
environment used for this project, the `python` command is not
available.

The plotting scripts require Matplotlib. If it is not installed, install
it with:

``` bash
python3 -m pip install matplotlib
```

The `pred` command is provided by `problemreductions-cli` version
`0.6.0`. If needed, install the version used for this exercise with:

``` bash
cargo install problemreductions-cli@0.6.0 --locked
```

## 2. Build the Rust program

Run all commands below from the `week11/` directory:

``` bash
cargo fmt
cargo check
cargo install --path . --force
```

The executable is named `tnsample`. The command-line interface and run
parameters are defined in `tnsample.design.toml`.

------------------------------------------------------------------------

## 3. Part 2 — Logic-gate gadgets

### Goal

Represent bits as spins using `s = 2x - 1`, enumerate the energy of
every truth-table row for the seven logic-gate gadgets, and verify that
the valid rows lie at each gadget’s minimum energy.

The seven gadgets are:

- AND
- OR
- NOT
- COPY
- SET0
- SET1
- XOR

For XOR, the reported truth-table energy is minimized over its helper
spin.

The program also enumerates all 16 assignments of the four-spin circuit:

``` text
AND(a,b,t) + NOT(t,out) + SET1(out)
```

### Run

``` bash
mkdir -p runs/gadgets artifacts

tnsample --mode gadgets --seed 2026 --out runs/gadgets

python3 scripts/plot_gates.py
```

### Expected checks

| Gadget / circuit         | Minimum energy |
|:-------------------------|---------------:|
| AND                      |             -3 |
| OR                       |             -3 |
| NOT                      |             -1 |
| COPY                     |             -1 |
| SET0                     |             -1 |
| SET1                     |             -1 |
| XOR                      |             -4 |
| AND + NOT + SET1 circuit |             -5 |

The three-gate circuit reaches its ground-state energy of **-5** in
exactly **3 of its 16 states**. These are the input assignments `00`,
`01`, and `10`, for which `NOT(AND(a,b)) = 1`.

### Output

- Raw run evidence: `runs/gadgets/run.json`
- Figure: `artifacts/gates.png`

The figure marks valid and invalid truth-table rows and shows the energy
of all 16 circuit states.

------------------------------------------------------------------------

## 4. Part 3 — Factor 15 by tensor-network contraction

### Goal

Use tensor-network contraction to calculate the exact ground-state
energy and degeneracy of the SpinGlass representation of Factor 15,
sample Boltzmann configurations, decode a lowest-energy sample back to
the factors, and compare contraction costs for different factor bit
lengths.

### 4.1 Create the Factor 15 reduction bundle

If the Part 1 files are not already present, create them with:

``` bash
mkdir -p runs/pred

pred create Factoring --target 15 --m 2 --n 3 \
  -o runs/pred/factor_15.json

pred reduce runs/pred/factor_15.json --to SpinGlass \
  -o runs/pred/factor_15.bundle.json
```

The reduction bundle represents the factoring problem as a SpinGlass
instance.

### 4.2 Count and sample the Factor 15 SpinGlass

First check the 10 × 10 ferromagnet, then contract the Factor 15 bundle
and draw 2000 independent samples at `β = 2.0`:

``` bash
tnsample --mode count --l 10 --p-af 0 --beta 2.0 \
  --seed 2026 --out runs/count_p0

tnsample --mode count --input runs/pred/factor_15.bundle.json \
  --beta 2.0 --seed 2026 --out runs/count_f15

tnsample --mode sample --input runs/pred/factor_15.bundle.json \
  --beta 2.0 --samples 2000 --seed 2026 --out runs/sample_f15
```

The sampling run stores the sampled configurations and energies. It also
writes `best_config.txt`, containing the lowest-energy target-space
configuration in comma-separated 0/1 form.

### 4.3 Decode and check the lowest-energy sample

For the lowest-energy configuration observed in this run:

``` bash
pred extract runs/pred/factor_15.bundle.json --config 1,1,1,0,1

pred evaluate runs/pred/factor_15.json --config 1,1,1,0,1
```

The decoded configuration is `[1, 1, 1, 0, 1]`, corresponding to the
factors **3 × 5 = 15**. The evaluation result is `Min(0)`, confirming
that the product matches the target.

### 4.4 Compare contraction costs for k-bit factors

For each bit length `k = 2, ..., 8`, create a factoring instance with
target `(2^k - 1)^2`, reduce it to a SpinGlass, and plan its contraction
tree using both `greedy` and `treesa`:

``` bash
for k in 2 3 4 5 6 7 8; do
  T=$(( ((1 << k) - 1) * ((1 << k) - 1) ))

  pred create Factoring --target "$T" --m "$k" --n "$k" \
    -o "runs/pred/scale_$k.json"

  pred reduce "runs/pred/scale_$k.json" --to SpinGlass \
    -o "runs/pred/scale_$k.bundle.json"

  for ord in greedy treesa; do
    tnsample --mode plan --order "$ord" \
      --input "runs/pred/scale_$k.bundle.json" \
      --seed 2026 --out "runs/plan_k${k}_${ord}"
  done
done
```

The planner reports:

- `variables`: number of spins / variables in the reduced problem.
- `tc`: time complexity, in log₂ scale.
- `sc`: space complexity, in log₂ scale.
- `rwc`: read-write complexity.

### 4.5 Plot the results

``` bash
python3 scripts/plot_factoring.py
```

### Results and interpretation

The 10 × 10 ferromagnet check returns `E_min = -180` with two ground states.
For Factor 15, the exact contraction returns `E_min = -299` and ground-state
degeneracy `g0 = 1`; the recorded partition-function result is
`log Z ≈ 598.3179`. The largest intermediate tensor has 32 entries
(`sc = 5`). The lowest-energy configuration decodes to **3 × 5**, and the
original factoring instance evaluates to `Min(0)`.

For the 2000-sample run:

- Inverse temperature: `β = 2.0`.
- Predicted number of ground-state samples from the partition function:
  approximately **1455**.
- Observed number of samples at `E = -299` in the current run: **1494**.

The observed frequency is close to the theoretical prediction.

The contraction-cost figure compares the `sc` values produced by
`greedy` and `treesa` for `k = 2` to `8`. The TreeSA planner
substantially reduces the largest intermediate tensor compared with the
greedy plan in the current runs. The exact values are retained in each
`runs/plan_k*_*/run.json` file.

The key scaling point is that TreeSA exploits the multiplier network’s
narrow diagonal structure: its contraction width grows approximately
linearly with the factor bit length `k`, while the tensor size and total
contraction cost remain exponential in that width.

### Output

- Factor 15 reduction: `runs/pred/factor_15.json`
- SpinGlass bundle: `runs/pred/factor_15.bundle.json`
- Exact contraction evidence: `runs/count_f15/run.json`
- Sampling evidence: `runs/sample_f15/run.json`
- Best sampled configuration: `runs/sample_f15/best_config.txt`
- Planning evidence: `runs/plan_k*_greedy/run.json` and
  `runs/plan_k*_treesa/run.json`
- Figure: `artifacts/factoring.png`

------------------------------------------------------------------------

## 5. Part 4 — N-Queens tensor network

### Goal

Encode the N-Queens constraints directly as a sparse 0/1 tensor network,
without reducing the puzzle to a SpinGlass.

Each square has one binary variable shared by four carry tensors: its
row, column, and two diagonals. The carry tensors enforce:

- Exactly one queen in each row.
- Exactly one queen in each column.
- At most one queen on each diagonal.

Because each valid board has weight 1, contraction counts all valid
boards exactly, and sampling from the stored contraction tree draws
valid boards uniformly.

### 5.1 Count exact solutions

Run the exact count for board sizes `N = 4` to `10`:

``` bash
for N in 4 5 6 7 8 9 10; do
  tnsample --mode count --n "$N" --seed 2026 \
    --out "runs/count_nq$N"
done
```

### 5.2 Sample valid boards

Draw 200 boards for both `N = 6` and `N = 8`:

``` bash
for N in 6 8; do
  tnsample --mode sample --n "$N" --samples 200 --seed 2026 \
    --out "runs/sample_nq$N"
done
```

### 5.3 Plot and validate

``` bash
python3 scripts/plot_nqueens.py
```

The plotting script checks that:

- The exact counts for `N = 4` to `10` match the known sequence.
- Both sampling runs contain exactly 200 samples.
- Every sampled board is valid.
- All four valid `N = 6` boards appear in the sample histogram.
- The `N = 6` histogram totals 200 samples.

### Results

| Board size (N)          |   4 |   5 |   6 |   7 |   8 |   9 |  10 |
|:------------------------|----:|----:|----:|----:|----:|----:|----:|
| Contracted valid boards |   2 |  10 |   4 |  40 |  92 | 352 | 724 |
| Known valid boards      |   2 |  10 |   4 |  40 |  92 | 352 | 724 |

The contracted counts match the known N-Queens sequence for every tested
board size.

For `N = 6`, the 200 samples visited all four valid boards. The observed
frequencies in the current run were:

| Sampled board (queen columns by row) | Frequency |
|:-------------------------------------|----------:|
| `(1, 3, 5, 0, 2, 4)`                 |        52 |
| `(2, 5, 1, 4, 0, 3)`                 |        47 |
| `(3, 0, 4, 1, 5, 2)`                 |        45 |
| `(4, 2, 0, 5, 3, 1)`                 |        56 |
| **Total**                            |   **200** |

Each board appears close to the uniform expectation of 50 occurrences
(reference range: `50 ± 6`).

For `N = 8`, the 200 samples visited **78 distinct valid boards**. All
sampled boards passed the row, column, and diagonal validity checks. The
expected number of distinct boards is approximately 82.

### Output

- Exact-count evidence: `runs/count_nqN/run.json`
- `N = 6` sampling evidence: `runs/sample_nq6/run.json`
- `N = 8` sampling evidence: `runs/sample_nq8/run.json`
- Figure: `artifacts/nqueens.png`

------------------------------------------------------------------------

## 6. Figures and run evidence

The three figures required for the Week 11 presentation are:

| Figure                    | Content                                                                                  |
|:--------------------------|:-----------------------------------------------------------------------------------------|
| `artifacts/gates.png`     | Energies of the seven logic-gate gadgets and the 16 states of the combined circuit       |
| `artifacts/factoring.png` | Greedy / TreeSA contraction space complexity and the Factor 15 sampling-energy histogram |
| `artifacts/nqueens.png`   | Exact N-Queens counts and the 200 sampled `N = 6` boards                                 |

The corresponding raw run evidence is stored under `runs/`. The JSON
files preserve run parameters, summaries, and the data used by the
plotting scripts.

Keep the `runs/` and `artifacts/` directories locally so the figures and
results can be reproduced. They are generated experiment evidence and
should not be included in the source-code commit unless specifically
requested.

## 7. Reproduce all three figures

After the relevant runs have been completed, regenerate the figures from
the saved run data:

``` bash
python3 scripts/plot_gates.py
python3 scripts/plot_factoring.py
python3 scripts/plot_nqueens.py
```

The scripts read the JSON evidence under `runs/` and write the figures
under `artifacts/`.

## 8. Git submission

The source code, design file, plotting scripts, and README are the files
to commit and push. Keep generated `runs/` and `artifacts/` data out of
Git.

For a README-only update after the main project commit:

``` bash
git add README.md
git commit -m "Complete Week 11 README"
git push
```

For the complete source submission, include the relevant source and
script files as well:

``` bash
git add Cargo.toml Cargo.lock src/main.rs tnsample.design.toml scripts README.md
git commit -m "Complete Week 11 tensor network exercise"
git push
```

------------------------------------------------------------------------

## 9. Presentation checklist

When presenting the work, show the three figures and explain:

1.  **Gates:** how each gadget’s ground-state energy selects its valid
    truth-table rows, and how the shared-wire AND–NOT–SET1 circuit
    reaches its floor only for the three accepted input assignments.
2.  **Factor 15:** how the factoring instance is reduced to a SpinGlass,
    solved by tensor contraction, sampled at `β = 2.0`, and decoded back
    to `3 × 5 = 15`. Compare the observed ground-state sample count with
    the prediction from `2000 p0`.
3.  **Contraction scaling:** how TreeSA reduces the largest intermediate
    tensor compared with greedy, using the multiplier’s narrow diagonal
    structure, while the overall cost still grows exponentially with the
    contraction width.
4.  **N-Queens:** how sparse carry tensors enforce row, column, and
    diagonal constraints, how the exact counts match the known sequence
    for `N = 4` to `10`, and how the sampling runs produce only valid
    boards.
