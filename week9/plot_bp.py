import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

cases = [
    ("prior", "Prior"),
    ("AD", "AD"),
    ("ADX", "ADX"),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
summary = []

for key, label in cases:
    with open(f"artifacts/bp-{key}/run.json") as f:
        d = json.load(f)

    h = d["history"]
    x = [v["sweep"] for v in h]
    residual = [max(v["undamped_residual"], 1e-18) for v in h]
    error = [max(v["max_marginal_error"], 1e-18) for v in h]

    axes[0].semilogy(x, residual, label=label)
    axes[1].semilogy(x, error, label=label)

    summary.append({
        "case": label,
        "converged": d["converged"],
        "sweeps": d["sweeps"],
        "residual": d["final_undamped_residual"],
        "max_marginal_error": d["max_marginal_error"],
        "update_work": d["cumulative_update_work"]
    })

axes[0].axhline(1e-10, linestyle="--", label="Tolerance")
axes[0].set_title("BP Fixed-point Residual")
axes[0].set_xlabel("Sweeps")
axes[0].set_ylabel("Max |F(m)-m|")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].set_title("BP Marginal Error")
axes[1].set_xlabel("Sweeps")
axes[1].set_ylabel("Max |p_BP-p_exact|")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

fig.suptitle("ASIA: Loopy Belief Propagation")
fig.tight_layout()

os.makedirs("evidence", exist_ok=True)
fig.savefig("evidence/bp_traces.png", dpi=200, bbox_inches="tight")

with open("evidence/bp_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Saved evidence/bp_traces.png")
for item in summary:
    print(item)
