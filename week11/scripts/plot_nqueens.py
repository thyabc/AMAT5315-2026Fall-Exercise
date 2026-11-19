#!/usr/bin/env python3
"""Plot exact N-queens counts and the N=6 uniform-sampling frequencies."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "artifacts" / "nqueens.png"
KNOWN = {4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352, 10: 724}


def read_json(path: Path):
    if not path.exists():
        raise SystemExit(f"Missing run evidence: {path}. Run the N-queens commands first.")
    return json.loads(path.read_text(encoding="utf-8"))


sizes = list(KNOWN)
contracted = []
for n in sizes:
    data = read_json(RUNS / f"count_nq{n}" / "run.json")
    count = int(data["valid_boards"])
    if count != KNOWN[n]:
        raise SystemExit(f"N={n}: contraction returned {count}, expected {KNOWN[n]}")
    contracted.append(count)

sample = read_json(RUNS / "sample_nq6" / "run.json")
sample8 = read_json(RUNS / "sample_nq8" / "run.json")
histogram = sample.get("histogram", [])
histogram = sorted(histogram, key=lambda item: tuple(item["columns"]))
labels = [str(tuple(item["columns"])) for item in histogram]
frequencies = [int(item["count"]) for item in histogram]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

ax = axes[0]
ax.plot(sizes, contracted, marker="o", linewidth=2, label="sparse tensor contraction")
ax.scatter(sizes, [KNOWN[n] for n in sizes], marker="x", s=70, label="known (OEIS A000170)")
ax.set_yscale("log")
ax.set_xticks(sizes)
ax.set_xlabel("board size N")
ax.set_ylabel("valid boards (log scale)")
ax.set_title("(a) Exact N-queens counts")
ax.grid(True, which="both", alpha=0.25)
ax.legend()

ax = axes[1]
if not frequencies:
    raise SystemExit("sample_nq6/run.json contains no sampling histogram")
ax.bar(range(len(frequencies)), frequencies, width=0.72, label="sampled frequency")
ax.axhspan(44, 56, alpha=0.14, label="expected range 50 ± 6")
ax.axhline(50, linestyle="--", linewidth=1.2, label="uniform expectation = 50")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_xlabel("queen columns by row")
ax.set_ylabel("times drawn (out of 200)")
ax.set_title("(b) 200 sampled 6-queens boards")
ax.grid(True, axis="y", alpha=0.25)
ax.legend(fontsize=8)

if not sample.get("all_samples_valid", False) or not sample8.get("all_samples_valid", False):
    raise SystemExit("Sampling evidence reports an invalid board")
if sample.get("samples_requested") != 200 or sample8.get("samples_requested") != 200:
    raise SystemExit("Both N=6 and N=8 runs must contain exactly 200 samples")
if len(histogram) != 4:
    raise SystemExit(f"N=6 sample visited {len(histogram)} distinct boards; expected all 4")
if sum(frequencies) != 200:
    raise SystemExit(f"N=6 histogram totals {sum(frequencies)}, expected 200")
if any(count < 44 or count > 56 for count in frequencies):
    print(f"Note: N=6 frequencies {frequencies} fall outside the assignment's approximate 50 ± 6 range")

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Saved {OUT}")
print(f"N=6 distinct boards: {len(histogram)}; frequencies: {frequencies}")
print(f"N=8 distinct boards among 200 samples: {sample8['unique_boards']}")
print("All N=6 and N=8 sampled boards passed validity checks.")
