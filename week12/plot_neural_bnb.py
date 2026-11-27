#!/usr/bin/env python3
"""Plot measured training/evaluation evidence and report all Part 3 criteria."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument("--model",default="artifacts/model.json")
parser.add_argument("--run",default="artifacts/eval_test/run.json")
parser.add_argument("--out",default="artifacts/neural_bnb.png")
parser.add_argument("--require-course-settings",action="store_true")
args=parser.parse_args()
model=json.loads(Path(args.model).read_text())
run=json.loads(Path(args.run).read_text())
training=model["training_summary"]; summary=run["summary"]
history=model["history"]; records=run["instances"]
train_ids=set(training["train_instance_ids"]); val_ids=set(training["validation_instance_ids"])
source_hashes=set(model["data_provenance"]["graph_hashes"])
test_hashes={hashlib.sha256(json.dumps(r["graph"],sort_keys=True,separators=(",",":")).encode()).hexdigest() for r in records}
allowed=[h for h in history if h["epoch"]==1 or (h["epoch"]>0 and h["epoch"]%5==0)]
checks={
    "exactness_and_lp_accounting":run["validation_status"]=="passed" and summary["max_energy_error"]<=1e-6 and summary["lp_accounting_passed"],
    "train_validation_instance_overlap_zero":not bool(train_ids&val_ids) and not training["instance_overlap"],
    "instance_split_correct":all(0<=i<36 for i in train_ids) and all(36<=i<48 for i in val_ids),
    "unseen_test_graphs":not bool(source_hashes&test_hashes),
    "cpu_float64":model["environment"]["backend"]=="cpu" and model["environment"]["x64"] is True,
    "six_probability_probes":summary["probability_parity"]["probe_count"]==6 and summary["probability_parity"]["max_probability_error"]<1e-4,
    "independent_feature_recomputation":training["feature_check_max_abs_error"]<1e-10,
    "padded_candidates_have_zero_probability":training["maximum_padded_probability"]==0,
    "checkpoint_selected_by_validation":abs(training["best_validation_ce"]-min(h["validation_ce"] for h in allowed))<1e-10,
    "training_ce_reduced":history[-1]["train_ce"]<history[0]["train_ce"],
    "neural_uses_no_trial_lps":all(r["searches"]["neural"]["trial_lp_solves"]==0 and r["searches"]["neural"]["lp_solves"]==r["searches"]["neural"]["nodes"] for r in records),
    "most_frac_lp_solves_equal_nodes":all(r["searches"]["most-frac"]["lp_solves"]==r["searches"]["most-frac"]["nodes"] for r in records),
    "neural_saves_nodes":summary["neural_saves_nodes"],
}
if args.require_course_settings:
    source=model["data_provenance"]["parameters"]; evaluation=run["parameters"]
    checks["course_settings"]=(source["spins"],source["edges"],source["count"],source["seed"])==(22,52,48,2026) and \
        (evaluation["spins"],evaluation["edges"],evaluation["count"],evaluation["seed"])==(22,52,16,3026) and \
        training["epochs"]==120 and training["batch_size"]==64 and training["seed"]==2026 and \
        training["learning_rate"]==4e-3 and training["weight_l2"]==2e-4 and \
        not training["biases_regularized"] and model["environment"]["jax"]=="0.12.1"

plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout="constrained")
axes[0].plot([h["epoch"] for h in history],[h["train_ce"] for h in history],label="Training",color="#3765ad")
axes[0].plot([h["epoch"] for h in history],[h["validation_ce"] for h in history],label="Validation",color="#d0762b")
axes[0].scatter([training["best_epoch"]],[training["best_validation_ce"]],marker="*",s=120,color="#923b23",zorder=4,label="Saved checkpoint")
axes[0].set(xlabel="Epoch",ylabel="Cross-entropy (without L2)",title="Imitating Strong Branching")
for policy,color,label in [("most-frac","#3765ad","Most fractional"),("neural","#329568","Neural"),("strong","#d0762b","Strong branching")]:
    axes[1].plot(range(1,len(records)+1),sorted(r["searches"][policy]["nodes"] for r in records),label=label,color=color,lw=2)
axes[1].set(xlabel="Independent sorted rank",ylabel="Explored nodes",title="Exact search on unseen graphs")
for ax in axes:
    ax.grid(axis="y",alpha=0.2); ax.legend(frameon=False)
p=run["parameters"]
fig.suptitle(f"Learned branching: train/validation split by instance; test n={p['spins']}, m={p['edges']}, seed={p['seed']}")
out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out,dpi=180)

largest=summary["largest_saving"]
largest_brief={"instance_id":largest["instance_id"],"ground_energy":largest["ground_energy"]}
for policy in ["most-frac","neural"]:
    largest_brief[policy]={k:largest[policy][k] for k in ["nodes","total_ms","first_splits"]}
report={"checks":checks,"all_passed":all(checks.values()),"training_summary":training,
    "probability_parity":summary["probability_parity"],"max_energy_error":summary["max_energy_error"],
    "policies":summary["policies"],"largest_saving":largest_brief}
Path(args.run).parent.joinpath("part3_checks.json").write_text(json.dumps(report,indent=2))
print("Saved",out.resolve())
print(f"Train: {training['train_nodes']} nodes / {len(train_ids)} instances; validation: {training['validation_nodes']} nodes / {len(val_ids)} instances; overlap: {len(train_ids&val_ids)}")
print(f"Best checkpoint: epoch {training['best_epoch']}, validation CE={training['best_validation_ce']:.6f}")
print("Max feature error:",training["feature_check_max_abs_error"])
print("Max JAX/Rust probability error:",summary["probability_parity"]["max_probability_error"])
print("Max energy error vs HiGHS:",summary["max_energy_error"])
print("\nPolicy       Mean nodes   Mean LP solves   Mean time (ms)")
for policy in ["most-frac","neural","strong"]:
    s=summary["policies"][policy]
    print(f"{policy:<12} {s['mean_nodes']:10.3f} {s['mean_lp_solves']:16.3f} {s['mean_total_ms']:16.3f}")
print("\nLargest node saving:",json.dumps(largest_brief))
print("\nChecks:",json.dumps(checks,indent=2))
if not all(checks.values()):
    raise SystemExit("Part 3 criteria not met: "+", ".join(k for k,v in checks.items() if not v))
print("\nPART 3 VALIDATION PASSED")
