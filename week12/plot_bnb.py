#!/usr/bin/env python3
"""Independently sort each curve; use only actual saved B&B runs."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument("--most",default="artifacts/bnb_most/run.json")
parser.add_argument("--strong",default="artifacts/bnb_strong/run.json")
parser.add_argument("--out",default="artifacts/bnb.png")
args=parser.parse_args()
runs=[json.loads(Path(path).read_text()) for path in [args.most,args.strong]]
for run in runs:
    if run.get("validation_status")!="passed":
        raise SystemExit("A saved run failed validation; inspect its raw data before plotting.")
p0,p1=[r["parameters"] for r in runs]
assert all(p0[k]==p1[k] for k in ["spins","edges","count","seed"]),"Benchmark parameters differ"
assert [r["parameters"]["policy"] for r in runs]==["most-frac","strong"]
assert [r["graph"] for r in runs[0]["instances"]]==[r["graph"] for r in runs[1]["instances"]],"Input graphs differ"
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
for run,color,label in zip(runs,["#3765ad","#d0762b"],["Most fractional","Strong branching"]):
    records=run["instances"]
    rank=range(1,len(records)+1)
    axes[0].plot(rank,sorted(r["search"]["nodes"] for r in records),label=label,color=color,lw=2)
    axes[1].plot(rank,sorted(r["search"]["lp_solves"] for r in records),label=label,color=color,lw=2)
for ax,title,ylabel in zip(axes,["Explored nodes","All LP solves, including lookahead"],["Explored nodes","LP solves"]):
    ax.set(xlabel="Independent sorted rank",ylabel=ylabel,title=title)
    ax.grid(axis="y",alpha=0.2)
    ax.legend(frameon=False)
fig.suptitle(f"Exact Branch-and-Bound: n={p0['spins']}, m={p0['edges']}, instances={p0['count']}, seed={p0['seed']}")
out=Path(args.out)
out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out,dpi=180)
print("Saved",out.resolve())
print("\nPolicy       Mean nodes   Mean LP solves   Mean time (ms)   Max energy error")
for run in runs:
    s=run["summary"]
    print(f"{s['policy']:<12} {s['mean_nodes']:10.3f} {s['mean_lp_solves']:16.3f} {s['mean_total_ms']:16.3f} {s['max_energy_error']:18.3e}")
    print("  LP accounting passed:",s["lp_accounting_passed"])
    if s["policy"]=="most-frac":
        print("  LP solves = nodes:",s["lp_solves_equal_nodes"])
    else:
        print("  Mean extra trial LP solves:",s["mean_trial_lp_solves"])
print("\nWorkshop: actual LP solves =",runs[0]["workshop"]["node_count"])
for row in runs[0]["workshop"]["nodes"]:
    print(json.dumps(row))
print("\nPART 2 VALIDATION PASSED")
