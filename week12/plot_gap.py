#!/usr/bin/env python3
"""Plot actual saved Rust runs; never substitute reference values for data."""
import argparse
import json
from pathlib import Path
import platform
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--run", default="artifacts/gap/run.json")
parser.add_argument("--out", default="artifacts/gap.png")
args = parser.parse_args()
run = json.loads(Path(args.run).read_text())
if run.get("validation_status") != "passed":
    raise SystemExit("Run validation did not pass; inspect the raw evidence before plotting.")
rows = run["instances"]
ids = [r["id"] for r in rows]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
ax[0].plot(ids, [r["lp"]["energy"] for r in rows], "o--", label="LP lower bound", color="#3765ad")
ax[0].plot(ids, [r["milp"]["energy"] for r in rows], "s-", label="Exact ground-state energy", color="#d0762b")
ax[0].set(xlabel="Instance", ylabel="Energy", title="LP relaxation and exact solution")
ax[1].bar([i-0.19 for i in ids], [r["gap"] for r in rows], width=0.38,
          label="Integrality gap", color="#3765ad")
ax[1].bar([i+0.19 for i in ids], [r["rounding_excess"] for r in rows], width=0.38,
          label="Naive rounding excess", color="#d0762b")
ax[1].set(xlabel="Instance", ylabel="Energy difference", title="Gap and rounding error")
for a in ax:
    a.set_xticks(ids)
    a.grid(axis="y", alpha=0.2)
    a.legend(frameon=False)
p = run["parameters"]
fig.suptitle(f"Spin glasses: n={p['spins']}, m={p['edges']}, seed={p['seed']}")
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=180)
print("Saved", out.resolve())
print("\nComputed results:")
print(json.dumps(run["summary"], indent=2))
env = {"python": sys.version, "matplotlib": matplotlib.__version__, "platform": platform.platform()}
Path(args.run).parent.joinpath("plot_environment.json").write_text(json.dumps(env, indent=2))
