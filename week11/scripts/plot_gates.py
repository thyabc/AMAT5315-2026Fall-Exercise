import json
from pathlib import Path
import matplotlib.pyplot as plt

RUN_JSON = Path("runs/gadgets/run.json")
OUT_PNG = Path("artifacts/gates.png")

with open(RUN_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

gate_order = ["AND", "OR", "NOT", "COPY", "SET0", "SET1", "XOR"]
gates = data["gates"]
circuit_states = data["circuit_states"]
ground_e = data["circuit_ground_energy"]

# ---------- 左图：七种逻辑门 ----------
gate_positions = []
gate_labels = []
valid_x = []
valid_y = []
invalid_x = []
invalid_y = []
tick_positions = []

x = 0
gap = 1.5

for gate in gate_order:
    rows = [r for r in gates if r["gate"] == gate]
    rows = sorted(rows, key=lambda r: r["row"])

    local_positions = []
    for r in rows:
        gate_positions.append(x)
        gate_labels.append(r["row"])
        local_positions.append(x)

        if r["valid"]:
            valid_x.append(x)
            valid_y.append(r["energy"])
        else:
            invalid_x.append(x)
            invalid_y.append(r["energy"])
        x += 1

    tick_positions.append(sum(local_positions) / len(local_positions))
    x += gap

# ---------- 右图：电路16种状态 ----------
circuit_states = sorted(
    circuit_states,
    key=lambda r: (r["a"], r["b"], r["t"], r["out"])
)

circuit_x = list(range(len(circuit_states)))
circuit_labels = [f'{r["a"]}{r["b"]}{r["t"]}{r["out"]}' for r in circuit_states]
circuit_y = [r["energy"] for r in circuit_states]

ground_x = [i for i, r in enumerate(circuit_states) if r["energy"] == ground_e]
ground_y = [r["energy"] for r in circuit_states if r["energy"] == ground_e]

# ---------- 画图 ----------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左图
ax = axes[0]
ax.scatter(invalid_x, invalid_y, marker="x", s=80, label="invalid row")
ax.scatter(valid_x, valid_y, marker="o", s=60, label="valid row")
ax.set_xticks(tick_positions)
ax.set_xticklabels(gate_order, rotation=0)
ax.set_ylabel("energy of a truth-table row")
ax.set_title("(a) Gate energies")
ax.legend()
ax.grid(True, alpha=0.3)

# 在底部标一些行标签（为了不太乱，只在每个点下方轻微标注）
for xpos, label in zip(gate_positions, gate_labels):
    ax.text(xpos, -4.9, label, rotation=90, ha="center", va="top", fontsize=7)

# 右图
ax2 = axes[1]
ax2.scatter(circuit_x, circuit_y, s=60, label="all states")
ax2.scatter(ground_x, ground_y, marker="*", s=160, label="ground states")
ax2.set_xticks(circuit_x)
ax2.set_xticklabels(circuit_labels, rotation=90)
ax2.set_ylabel("energy of a state")
ax2.set_title("(b) Three gates joined")
ax2.legend()
ax2.grid(True, alpha=0.3)

# 按 ab 分组做一点视觉提示
for i, r in enumerate(circuit_states):
    ax2.text(i, min(circuit_y) - 0.7, f'{r["a"]}{r["b"]}', ha="center", va="top", fontsize=7)

ax2.text(len(circuit_states)/2, min(circuit_y) - 1.6,
         "the 16 states of a, b, t, out",
         ha="center", va="top", fontsize=9)

plt.tight_layout()
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
print(f"Saved {OUT_PNG}")
