# AMAT5315 Week 11 — Part 4: N-Queens tensor network

This update adds the Part 4 N-queens carry network to the existing `tnsample`
program. The earlier `gadgets`, SpinGlass `plan`, `count`, and `sample` paths are
retained. N-queens uses 0/1 carry tensors and sparse contractions; it does not
compile the puzzle into a SpinGlass.

## Environment recorded from the project terminal

- Rust: `rustc 1.98.1 (48a229cea 2026-09-01)`
- Cargo: `cargo 1.98.1 (797e8a9bc 2026-08-05)`
- `pred`: `0.6.0`
- `omeco`: `0.2.6`
- Seed: `2026`
- Contraction order: `treesa`
- Sampling: `200` boards for each of N=6 and N=8

## Build and run

Run these commands from the `week11/` project directory:

```bash
cargo fmt
cargo check
cargo install --path . --force

for N in 4 5 6 7 8 9 10; do
  tnsample --mode count --n "$N" --order treesa --seed 2026 --out "runs/count_nq$N"
done

for N in 6 8; do
  tnsample --mode sample --n "$N" --samples 200 --order treesa --seed 2026 \
    --out "runs/sample_nq$N"
done

python scripts/plot_nqueens.py
```

## Expected exact counts

| N | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Valid boards | 2 | 10 | 4 | 40 | 92 | 352 | 724 |

The `count` run evidence is saved to `runs/count_nqN/run.json`. The sampling
evidence is saved to `runs/sample_nqN/run.json`; each sampled board is checked
against the row, column and diagonal rules before it is recorded.

For N=6, there are four valid boards, so each should appear about 50 times in
200 uniform samples (the assignment's reference range is 50 ± 6). For N=8,
record the number of distinct boards visited among the 200 samples; the expected
value is approximately 82 (the reference run observed 80).

The plotting script writes `artifacts/nqueens.png`, showing the exact counts
against the known sequence and the N=6 sample frequencies.
