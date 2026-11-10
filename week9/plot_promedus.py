import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("evidence", exist_ok=True)

# Contraction-order comparison.
methods = ["greedy", "treesa"]
labels = ["OMECO Greedy", "Time-only TreeSA"]
work = []
largest = []

for method in methods:
    with open(f"runs/promedus/{method}/complexity.json", encoding="utf-8") as f:
        d = json.load(f)
    work.append(d["work"])
    largest.append(d["largest_tensor_elements"])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
axes[0].bar(labels, work)
axes[0].set_yscale("log")
axes[0].set_title("Contraction work")
axes[0].set_ylabel("Sum of pairwise index-space sizes")
axes[0].tick_params(axis="x", rotation=12)
axes[0].grid(axis="y", alpha=0.25)

axes[1].bar(labels, largest)
axes[1].set_yscale("log")
axes[1].set_title("Largest generated tensor")
axes[1].set_ylabel("Number of entries")
axes[1].tick_params(axis="x", rotation=12)
axes[1].grid(axis="y", alpha=0.25)

fig.suptitle("Promedus_14: Greedy vs time-only TreeSA")
fig.tight_layout()
fig.savefig("evidence/promedus_orders.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# BP residual and selected-query marginal error.
with open("artifacts/bp-promedus/run.json", encoding="utf-8") as f:
    bp = json.load(f)

history = bp["history"]
sweeps = [x["sweep"] for x in history]
residual = [max(x["undamped_residual"], 1e-18) for x in history]
error = [max(x["max_selected_marginal_error"], 1e-18) for x in history]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].semilogy(sweeps, residual, label="Undamped fixed-point residual")
axes[0].axhline(1e-10, linestyle="--", label="Tolerance = 1e-10")
axes[0].set_title("BP fixed-point residual")
axes[0].set_xlabel("Completed sweeps")
axes[0].set_ylabel("Max |F(m) - m|")
axes[0].grid(True, alpha=0.25)
axes[0].legend()

axes[1].semilogy(sweeps, error, label="Maximum selected-query error")
axes[1].set_title("BP marginal accuracy")
axes[1].set_xlabel("Completed sweeps")
axes[1].set_ylabel("Max absolute error vs exact")
axes[1].grid(True, alpha=0.25)
axes[1].legend()

fig.suptitle("Promedus_14: Loopy BP (queries 59, 102, 139)")
fig.tight_layout()
fig.savefig("evidence/promedus_bp.png", dpi=220, bbox_inches="tight")
plt.close(fig)

print("Saved evidence/promedus_orders.png")
print("Saved evidence/promedus_bp.png")
print("BP converged:", bp["converged"])
print("BP sweeps:", bp["sweeps"])
print("BP final residual:", bp["final_undamped_residual"])
print("BP cumulative update work:", bp["cumulative_update_work"])
for var, result in bp["selected_queries"].items():
    print(
        f"Variable {var}: BP state-1={result['state1']:.10f}, "
        f"exact state-1={result['exact_state1']:.10f}, "
        f"absolute error={result['state1_absolute_error']:.6e}"
    )
