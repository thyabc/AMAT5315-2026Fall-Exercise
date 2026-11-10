#!/usr/bin/env python3
"""Generate the ASIA query diagram and probability plots from saved run.json files."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
CASES = ["prior", "AD", "ADX"]
CASE_LABELS = {"prior": "Prior", "AD": "A=0, D=0", "ADX": "A=0, D=0, X=0"}
DISEASES = [("T", 1, "Tuberculosis"), ("L", 3, "Lung cancer"), ("B", 4, "Bronchitis")]


def load_case(case):
    with (ROOT / "artifacts" / case / "run.json").open() as f:
        return json.load(f)


def write_query_diagram():
    # Bipartite factor graph. Variable order and factor scopes follow asia.uai.
    W, H = 1500, 650
    vars_ = [
        ("A", 110, 125), ("T", 300, 125), ("S", 490, 125), ("L", 680, 125),
        ("B", 870, 125), ("E", 1060, 125), ("X", 1250, 125), ("D", 1390, 125),
    ]
    var_pos = {name: (x, y) for name, x, y in vars_}
    factors = [
        ("p(A)", ["A"], 110, 330),
        ("p(T|A)", ["T", "A"], 250, 330),
        ("p(S)", ["S"], 420, 330),
        ("p(L|S)", ["L", "S"], 600, 330),
        ("p(B|S)", ["B", "S"], 790, 330),
        ("p(E|T,L)", ["E", "L", "T"], 960, 330),
        ("p(X|E)", ["X", "E"], 1160, 330),
        ("p(D|E,B)", ["D", "E", "B"], 1330, 330),
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.var{fill:white;stroke:#202124;stroke-width:2.5}.factor{fill:#f4f4f4;stroke:#202124;stroke-width:2}.edge{stroke:#555;stroke-width:2;fill:none}.observed{fill:#fff4df;stroke:#a66a00;stroke-width:3}.small{font-size:19px}.label{font-size:23px;font-weight:bold}.title{font-size:30px;font-weight:bold}</style>',
        '<text x="750" y="52" text-anchor="middle" class="title">ASIA tensor network for the AD tuberculosis query</text>',
        '<text x="750" y="85" text-anchor="middle" class="small">UAI factor scopes; observed variables A=0 and D=0 are fixed, T remains open</text>',
    ]

    # Connect factor boxes to variables. Curves route the longer hyperedges.
    for label, scope, fx, fy in factors:
        for name in scope:
            vx, vy = var_pos[name]
            # Simple curved links make the factor graph readable.
            midy = (vy + fy) / 2
            parts.append(
                f'<path class="edge" d="M {vx} {vy+28} Q {(vx+fx)/2} {midy+12} {fx} {fy-28}"/>'
            )

    # Factor boxes and labels.
    for label, scope, fx, fy in factors:
        parts.append(f'<rect class="factor" x="{fx-58}" y="{fy-28}" width="116" height="56" rx="9"/>')
        parts.append(f'<text x="{fx}" y="{fy+8}" text-anchor="middle" class="label">{label}</text>')
        parts.append(f'<text x="{fx}" y="{fy+58}" text-anchor="middle" class="small">scope: {", ".join(scope)}</text>')

    # Variables; observed nodes are visually distinguished.
    for name, x, y in vars_:
        cls = "var observed" if name in ("A", "D") else "var"
        parts.append(f'<circle class="{cls}" cx="{x}" cy="{y}" r="28"/>')
        parts.append(f'<text x="{x}" y="{y+8}" text-anchor="middle" class="label">{name}</text>')

    parts += [
        '<text x="110" y="205" text-anchor="middle" class="label">A = 0</text>',
        '<text x="1390" y="205" text-anchor="middle" class="label">D = 0</text>',
        '<path d="M 300 166 L 300 220" stroke="#176b3a" stroke-width="3"/>',
        '<text x="300" y="250" text-anchor="middle" font-size="21" font-weight="bold" fill="#176b3a">QUERY: T</text>',
        '<text x="750" y="570" text-anchor="middle" class="small">Contract all eight factor tables; sum every unobserved variable except T.</text>',
        '<text x="750" y="605" text-anchor="middle" class="small">The resulting two entries are P(T=0, A=0, D=0) and P(T=1, A=0, D=0); normalize by P(A=0,D=0).</text>',
        '</svg>',
    ]
    (EVIDENCE / "graphs.svg").write_text("\n".join(parts), encoding="utf-8")


def plot_posteriors(data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    x = np.arange(len(DISEASES))
    width = 0.34

    for ax, case in zip(axes, CASES):
        d = data[case]
        enum = [d["marginals_enumeration"][idx][0] for _, idx, _ in DISEASES]
        tens = [d["marginals_contraction"][idx][0] for _, idx, _ in DISEASES]
        ax.bar(x - width/2, enum, width, label="Enumeration")
        ax.bar(x + width/2, tens, width, label="Tensor contraction")
        ax.set_title(f'{CASE_LABELS[case]}\nEvidence mass = {d["evidence_mass_enumeration"]:.8g}')
        ax.set_xticks(x, [short for _, _, short in DISEASES], rotation=18, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.text(
            0.02, 0.96,
            f"max marginal error = {d['max_marginal_error']:.1e}",
            transform=ax.transAxes, va="top", fontsize=8
        )

    axes[0].set_ylabel("Posterior probability of state 0 (disease present)")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("ASIA disease probabilities: enumeration vs tensor contraction", fontsize=14)
    fig.tight_layout()
    fig.savefig(EVIDENCE / "posteriors.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_joint_dependence(data):
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11))
    col_titles = [
        "Exact joint P(T,L | evidence)",
        "Product of marginals P(T)P(L)",
        "Dependence: joint − product",
    ]

    for row, case in enumerate(CASES):
        d = data[case]
        joint = np.asarray(d["joint_tl_enumeration"], dtype=float)
        p_t = np.asarray(d["marginals_enumeration"][1], dtype=float)
        p_l = np.asarray(d["marginals_enumeration"][3], dtype=float)
        product = np.outer(p_t, p_l)
        dependence = joint - product
        dependence[np.abs(dependence) < 5e-5] = 0.0  # suppress floating-point signed-zero noise

        mats = [joint, product, dependence]
        for col, (ax, mat) in enumerate(zip(axes[row], mats)):
            if col < 2:
                im = ax.imshow(mat, vmin=0, vmax=1, aspect="equal")
            else:
                lim = max(0.05, float(np.max(np.abs(mat))))
                im = ax.imshow(mat, vmin=-lim, vmax=lim, aspect="equal")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{mat[i,j]:.4f}", ha="center", va="center", fontsize=9)
            ax.set_xticks([0, 1], ["L=0", "L=1"])
            ax.set_yticks([0, 1], ["T=0", "T=1"])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{CASE_LABELS[case]}\nT state")
            ax.set_xlabel("Lung-cancer state")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("ASIA T–L joint probability and dependence", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(EVIDENCE / "joint.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    EVIDENCE.mkdir(exist_ok=True)
    data = {case: load_case(case) for case in CASES}
    write_query_diagram()
    plot_posteriors(data)
    plot_joint_dependence(data)
    print("Saved evidence/graphs.svg")
    print("Saved evidence/posteriors.png")
    print("Saved evidence/joint.png")


if __name__ == "__main__":
    main()
