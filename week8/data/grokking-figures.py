# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Draw the Week 8 Part 4 grokking figures from saved curves and measurements.

Run from the repository root after week8/grokking/run_study.py:
    .venv/bin/python week8/data/grokking-figures.py
No training runs here. Writes PNGs to week8/results/.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "week8/results"
BLUE, CORAL, GRAY, GREEN, PURPLE = "#3269a0", "#b85135", "#8a8a8a", "#3d8a5a", "#7a4fa0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 10, "legend.fontsize": 8.5,
                     "savefig.dpi": 190})


def curve(z, name, key):
    s = z[f"curve/{name}/step"]
    v = z[f"curve/{name}/{key}"]
    keep = s > 0                                   # log x axis: drop step 0
    return s[keep], v[keep]


def fig_curves(z, m):
    g = m["groups"]["main"]
    r0 = g["runs"][0]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    ax = axes[0]
    for key, col, lab in [("train_acc", BLUE, "train"), ("test_acc", CORAL, "test")]:
        ax.plot(*curve(z, "main_s0", key), color=col, lw=1.6, label=f"{lab} (seed 0)")
    for r in g["runs"][1:]:
        ax.plot(*curve(z, r["name"], "test_acc"), color=CORAL, lw=0.8, alpha=0.45)
    ax.plot([], [], color=CORAL, lw=0.8, alpha=0.45, label="test (other seeds)")
    for step, lab, dy, dx, ha in [(r0["memorization_step"], "train ≥ 0.99", 0.86, 4, "left"),
                                  (r0["grokking_step"], "test ≥ 0.99", 0.62, -4, "right")]:
        ax.axvline(step, color=GRAY, ls=":", lw=1)
        ax.annotate(f"{lab}\nstep {step}", (step, dy), xytext=(dx, 0), textcoords="offset points", fontsize=8,
                    color="#444", ha=ha)
    ax.set_xscale("log"); ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel("training step"); ax.set_ylabel("accuracy")
    ax.set_title(f"Accuracy: memorize first, generalize later\n(delay ratio {r0['delay_ratio']:.1f} for seed 0)")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.5), frameon=False)

    ax = axes[1]
    for key, col, lab in [("train_loss", BLUE, "train"), ("test_loss", CORAL, "test")]:
        ax.plot(*curve(z, "main_s0", key), color=col, lw=1.4, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training step"); ax.set_ylabel("cross-entropy (nats)")
    ax.axhline(np.log(m["protocol"]["p"]), color=GRAY, ls="--", lw=0.8, label="ln p (uniform guess)")
    ax.set_title("Loss (seed 0)")
    ax.legend(frameon=False, loc="lower left")

    ax = axes[2]
    for r in g["runs"]:
        s, v = curve(z, r["name"], "norm")
        ax.plot(s, v, color=BLUE, lw=1.2 if r["seed"] == 0 else 0.7, alpha=1 if r["seed"] == 0 else 0.5)
    if "wd0" in m["groups"]:
        for r in m["groups"]["wd0"]["runs"]:
            ax.plot(*curve(z, r["name"], "norm"), color=GRAY, lw=0.9, alpha=0.8)
        ax.plot([], [], color=GRAY, label="weight decay 0")
    ax.plot([], [], color=BLUE, label="weight decay 1")
    ax.axvline(r0["grokking_step"], color=GRAY, ls=":", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("parameter norm ‖θ‖₂")
    ax.set_title("Weight norm")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "grokking-curves.png"); plt.close(fig)


def fig_controls(z, m):
    G = m["groups"]
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    specs = [("main", CORAL, f"train fraction {m['protocol']['frac']}, wd 1"),
             ("frac05", GREEN, "train fraction 0.5, wd 1"),
             ("wd0", GRAY, f"train fraction {m['protocol']['frac']}, wd 0"),
             ("random", PURPLE, "random labels")]
    for g, col, lab in specs:
        if g not in G:
            continue
        for i, r in enumerate(G[g]["runs"]):
            ax.plot(*curve(z, r["name"], "test_acc"), color=col, lw=1.1, alpha=0.85, label=lab if i == 0 else None)
    ax.axhline(1 / m["protocol"]["p"], color="#444", ls=":", lw=0.8, label="chance 1/p")
    ax.set_xscale("log"); ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel("training step"); ax.set_ylabel("test accuracy")
    ax.set_title("Test accuracy: train fraction, weight decay and random labels")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(OUT / "grokking-controls.png"); plt.close(fig)


def fig_fourier(m):
    mech = m["mechanism"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 5.8), sharex=True)
    titles = {"init": "initialization", "memorized": "memorization", "final": "after grokking"}
    for ax, s in zip(axes, ["init", "memorized", "final"]):
        pw = np.asarray(mech["snapshots"][s]["power"])[1:]
        f = np.arange(1, pw.size + 1)
        cols = [CORAL if k in mech["key_frequencies"] else BLUE for k in f]
        ax.bar(f, pw / pw.sum(), color=cols, width=0.8)
        ax.set_ylabel("power share")
        ax.set_title(f"{titles[s]}: top-{mech['k']} share {mech['snapshots'][s]['top_k_fraction']:.2f}, "
                     f"test accuracy {mech['snapshots'][s]['test_acc']:.2f}", loc="left")
    for k in mech["key_frequencies"]:
        axes[2].annotate(str(k), (k, 0), xytext=(0, -1), textcoords="offset points", ha="center", va="top",
                         fontsize=7, color=CORAL)
    axes[2].set_xlabel("frequency k of cos(2πka/p), sin(2πka/p) over the residue a")
    fig.suptitle("Fourier power of the residue embedding W_E (orange: key frequencies of the final model)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "grokking-fourier.png"); plt.close(fig)


def fig_ablation(m):
    ab = m["mechanism"]["ablation"]
    key = m["mechanism"]["key_frequencies"]
    rs = ab["remove_random_nonkey_summary"]
    labels = ["trained\nmodel", f"keep only\nkey {len(key)}", f"remove\nkey {len(key)}",
              f"remove {len(key)} random\nnon-key (×{len(ab['remove_random_nonkey'])})", f"keep only {len(key)}\nrandom non-key"]
    vals = [ab["none"]["test_acc"], ab["keep_key"]["test_acc"], ab["remove_key"]["test_acc"], rs["mean"],
            ab["keep_random_nonkey"]["test_acc"]]
    cols = [BLUE, GREEN, CORAL, GRAY, GRAY]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bars = ax.bar(range(len(vals)), vals, color=cols, width=0.62)
    ax.errorbar([3], [rs["mean"]], yerr=[[rs["mean"] - rs["min"]], [rs["max"] - rs["mean"]]], color="#333", capsize=4, lw=1)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=8.5)
    ax.axhline(1 / m["protocol"]["p"], color="#444", ls=":", lw=0.8)
    ax.set_xticks(range(len(vals)), labels, fontsize=8.5)
    ax.set_ylim(0, 1.12); ax.set_ylabel("test accuracy")
    ax.set_title(f"Ablating frequencies of the residue embedding (key frequencies {key})")
    fig.tight_layout()
    fig.savefig(OUT / "grokking-ablation.png"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = json.loads((ROOT / "week8/data/grokking-measurements.json").read_text())
    z = np.load(ROOT / "week8/data/grokking-evidence.npz")
    fig_curves(z, m)
    fig_controls(z, m)
    fig_fourier(m)
    fig_ablation(m)
    print("wrote", ", ".join(f"week8/results/grokking-{n}.png" for n in ["curves", "controls", "fourier", "ablation"]))


if __name__ == "__main__":
    main()
