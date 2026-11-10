import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

names = {
    "a": "A", "b": "T", "c": "S", "d": "L",
    "e": "B", "f": "E", "g": "X", "h": "D"
}

factor_names = [
    "P(A)", "P(T,A)", "P(S)", "P(L,S)", "P(B,S)",
    "P(E,L,T)", "P(X,E)", "P(D,E,B)",
    "A=0 evidence", "D=0 evidence"
]

def draw_tree(ax, method):
    with open(f"runs/{method}/order.json") as f:
        plan = json.load(f)

    with open(f"runs/{method}/complexity.json") as f:
        complexity = json.load(f)

    scopes = plan["input_scopes"]
    largest = complexity["largest_tensor_elements"]

    nodes = []
    edges = []
    counter = [0]
    max_depth = [0]

    def layout(node, depth=0):
        max_depth[0] = max(max_depth[0], depth)

        if "Leaf" in node:
            idx = node["Leaf"]["tensor_index"]
            scope = scopes[idx]
            size = 2 ** len(scope)

            label = (
                f"F{idx}: {factor_names[idx]}\n"
                f"{','.join(names[x] for x in scope)}\n"
                f"{size} elements"
            )

            item = {
                "x": counter[0],
                "y": -depth,
                "label": label,
                "size": size
            }

            counter[0] += 1
            nodes.append(item)
            return item

        data = node["Node"]
        left = layout(data["args"][0], depth + 1)
        right = layout(data["args"][1], depth + 1)

        out = "".join(data["eins"]["iy"])
        size = 2 ** len(out)

        out_name = ",".join(names[x] for x in out) if out else "Scalar"

        item = {
            "x": (left["x"] + right["x"]) / 2,
            "y": -depth,
            "label": f"{out_name}\n{size} elements",
            "size": size
        }

        edges.append((item, left))
        edges.append((item, right))
        nodes.append(item)

        return item

    layout(plan["tree"])

    for parent, child in edges:
        ax.plot(
            [parent["x"], child["x"]],
            [parent["y"], child["y"]],
            linewidth=0.9
        )

    for node in nodes:
        ax.text(
            node["x"], node["y"], node["label"],
            ha="center", va="center", fontsize=6.5,
            bbox=dict(
                boxstyle="round,pad=0.3",
                alpha=0.8,
                linewidth=1.6 if node["size"] == largest else 0.7
            )
        )

    ax.set_title(
        f"{method.upper()}\n"
        f"Work = {complexity['work']}, "
        f"Largest tensor = {largest}",
        fontsize=11
    )

    ax.set_xlim(-1, len(scopes))
    ax.set_ylim(-max_depth[0] - 1, 1)
    ax.axis("off")


fig, axes = plt.subplots(1, 2, figsize=(17, 9))

draw_tree(axes[0], "random")
draw_tree(axes[1], "greedy")

fig.suptitle(
    "ASIA AD: Random vs Greedy Contraction Trees",
    fontsize=14
)

fig.tight_layout()
fig.savefig("evidence/orders.png", dpi=200, bbox_inches="tight")

print("Saved evidence/orders.png")
