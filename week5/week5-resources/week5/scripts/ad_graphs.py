import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

jax.config.update("jax_enable_x64", True)


# =========================================================
# Lennard-Jones computational graph
# =========================================================

def energy(r):
    a = r ** -6
    b = a ** 2
    c = b - a
    U = 4.0 * c
    return U


def var_key(v):
    return str(v)


def operation_label(eqn):
    """Create a readable label for one JAX primitive."""

    name = eqn.primitive.name

    # Match the useful style in the assignment reference.
    if name == "integer_pow":
        y = eqn.params.get("y", "")
        return f"integer_pow[y={y}]"

    return name


def draw_jaxpr(closed_jaxpr, filename, title):
    jaxpr = closed_jaxpr.jaxpr
    eqns = jaxpr.eqns

    # -----------------------------------------------------
    # Map every produced variable to the operation
    # that produced it.
    # -----------------------------------------------------

    producer = {}

    for i, eqn in enumerate(eqns):
        for outvar in eqn.outvars:
            producer[var_key(outvar)] = i

    # -----------------------------------------------------
    # Determine graph depth.
    # -----------------------------------------------------

    depths = []

    for i, eqn in enumerate(eqns):
        parent_depths = []

        for inv in eqn.invars:
            key = var_key(inv)

            if key in producer:
                p = producer[key]

                # Only use previous producer nodes.
                if p < i:
                    parent_depths.append(depths[p])

        if parent_depths:
            depth = max(parent_depths) + 1
        else:
            depth = 0

        depths.append(depth)

    # -----------------------------------------------------
    # Position nodes by depth.
    # -----------------------------------------------------

    depth_groups = defaultdict(list)

    for i, depth in enumerate(depths):
        depth_groups[depth].append(i)

    positions = {}

    for depth, nodes in depth_groups.items():
        n = len(nodes)

        if n == 1:
            positions[nodes[0]] = (depth, 0.0)

        else:
            spacing = 1.5

            for j, node in enumerate(nodes):
                y = (n - 1) * spacing / 2.0 - j * spacing
                positions[node] = (depth, y)

    max_depth = max(depths) if depths else 0

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------

    fig_width = max(10, 2.2 * (max_depth + 3))
    fig_height = max(6, 1.2 * max(len(v) for v in depth_groups.values()) + 4)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Draw operation nodes
    for i, eqn in enumerate(eqns):
        x, y = positions[i]

        ax.text(
            x,
            y,
            operation_label(eqn),
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="black"
            )
        )

    # Draw connections between operations
    for i, eqn in enumerate(eqns):

        child_x, child_y = positions[i]

        for inv in eqn.invars:
            key = var_key(inv)

            if key in producer:
                parent = producer[key]

                if parent < i:
                    parent_x, parent_y = positions[parent]

                    ax.annotate(
                        "",
                        xy=(child_x - 0.15, child_y),
                        xytext=(parent_x + 0.15, parent_y),
                        arrowprops=dict(
                            arrowstyle="->",
                            linewidth=0.8
                        )
                    )

    # -----------------------------------------------------
    # Input node
    # -----------------------------------------------------

    if len(jaxpr.invars) > 0:
        input_var = var_key(jaxpr.invars[0])

        input_x = -1
        input_y = 0

        ax.text(
            input_x,
            input_y,
            "r",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="black"
            )
        )

        for i, eqn in enumerate(eqns):

            used = any(
                var_key(v) == input_var
                for v in eqn.invars
            )

            if used:
                x, y = positions[i]

                ax.annotate(
                    "",
                    xy=(x - 0.15, y),
                    xytext=(input_x + 0.15, input_y),
                    arrowprops=dict(
                        arrowstyle="->",
                        linewidth=0.8
                    )
                )

    # -----------------------------------------------------
    # Output node
    # -----------------------------------------------------

    output_x = max_depth + 1
    output_y = 0

    ax.text(
        output_x,
        output_y,
        "output",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="black"
        )
    )

    for outvar in jaxpr.outvars:
        key = var_key(outvar)

        if key in producer:
            p = producer[key]
            x, y = positions[p]

            ax.annotate(
                "",
                xy=(output_x - 0.15, output_y),
                xytext=(x + 0.15, y),
                arrowprops=dict(
                    arrowstyle="->",
                    linewidth=0.8
                )
            )

    ax.set_title(title, fontsize=13)
    ax.axis("off")

    ax.set_xlim(-1.7, max_depth + 1.7)

    all_y = [p[1] for p in positions.values()]

    ymin = min(all_y + [0]) - 1.5
    ymax = max(all_y + [0]) + 1.5

    ax.set_ylim(ymin, ymax)

    fig.tight_layout()

    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out, dpi=200, bbox_inches="tight")

    plt.close(fig)

    print("Saved:", out)


def main():

    r = jnp.array(1.3, dtype=jnp.float64)

    # =====================================================
    # Original computational graph
    # =====================================================

    primal_jaxpr = jax.make_jaxpr(energy)(r)

    print()
    print("========== ORIGINAL JAXPR ==========")
    print(primal_jaxpr)

    # =====================================================
    # Gradient computational graph
    # =====================================================

    grad_energy = jax.grad(energy)

    grad_jaxpr = jax.make_jaxpr(grad_energy)(r)

    print()
    print("========== GRADIENT JAXPR ==========")
    print(grad_jaxpr)

    print()

    # Verify add_any appears
    grad_text = str(grad_jaxpr)

    if "add_any" in grad_text:
        print("PASS: add_any found in gradient graph.")
    else:
        print("WARNING: add_any was not found.")

    # =====================================================
    # Draw PNG files
    # =====================================================

    draw_jaxpr(
        primal_jaxpr,
        "artifacts/ad/graph.png",
        "JAX computational graph of U(r)"
    )

    draw_jaxpr(
        grad_jaxpr,
        "artifacts/ad/grad-graph.png",
        "JAX computational graph of grad U(r)"
    )


if __name__ == "__main__":
    main()
