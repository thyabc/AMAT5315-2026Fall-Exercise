#!/usr/bin/env python3
"""Draw Week 8 reference figures from recorded JAX artifacts, without training.

python week8/data/generate.py --artifacts fixtures/week8-reference/artifacts
python week8/data/generate.py  # redraw from the committed reference.json
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "speedrun"))
from plots import COLORS, load_artifacts  # noqa: E402

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "axes.prop_cycle": plt.cycler(color=COLORS)})


def save(fig, name, caption):
    fig.text(.02, .015, caption, fontsize=8, va="bottom", wrap=True)
    fig.tight_layout(rect=(0, .12, 1, 1))
    fig.savefig(HERE / f"{name}.svg", metadata={"Date": None})
    fig.savefig(HERE / f"{name}.png", dpi=180)
    plt.close(fig)


def generate(data):
    baseline = data["baselines"]
    probs = baseline["train_probabilities"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(np.arange(len(probs)), probs, color=COLORS[0], width=.85)
    ax.axhline(1 / len(probs), color=COLORS[1], ls="--", label=f'Uniform: PPL {baseline["uniform"]["ppl"]:.0f}')
    ax.set(xlabel="Character ID (sorted vocabulary)", ylabel="Probability", ylim=(0, max(probs)*1.2),
           title=f'Training-frequency predictor: validation PPL {baseline["frequency"]["ppl"]:.2f}')
    ax.legend(frameon=False, loc="upper right")
    save(fig, "baselines", "TinyShakespeare, contiguous 90% training / 10% validation. Add-one smoothing; 65 characters.")

    probe = data["causality"]
    attention = np.asarray(probe["attention"])
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    heat = ax.imshow(attention, cmap="Blues", vmin=0, vmax=1, origin="upper")
    n = len(attention)
    ax.set_xticks(range(n), [str(i) for i in range(n)], fontsize=8)
    ax.set_yticks(range(n), [f"{i}: {repr(c)[1:-1]}" for i, c in enumerate(probe["input"])], fontsize=8)
    ax.set(xlabel="Key position (character attended to)", ylabel="Query position",
           title="Causal attention: future positions have zero weight")
    fig.colorbar(heat, ax=ax, label="Attention probability", fraction=.05)
    state = (f'Checkpoint {probe["checkpoint_sha256"][:12]}.' if probe.get("checkpoint_sha256")
             else f'Untrained, seed {probe["initialization_seed"]}.')
    save(fig, "attention", f'Layer 0, head 0. {state} Input {probe["input"]!r}. Prefix-logit change {probe["prefix_error"]:.2g}.')

    if "resume" in data:
        resume = data["resume"]
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.plot(resume["steps"], resume["uninterrupted_losses"], label="Uninterrupted", lw=3)
        ax.plot(resume["steps"], resume["resumed_losses"], label="Fresh-process resume", ls="--", marker="o", ms=4)
        ax.set(xlabel="Training step", ylabel="Minibatch cross-entropy\n[nats/character]",
               ylim=(0, max(resume["uninterrupted_losses"] + resume["resumed_losses"])*1.15),
               title="Training continuation reproduces the same updates")
        ax.legend(frameon=False)
        save(fig, "resume", f'Maximum absolute errors: loss {resume["max_loss_error"]:.2g}; parameters {resume["max_parameter_error"]:.2g}; optimizer {resume["max_optimizer_error"]:.2g}. Short deterministic probe, same backend.')

    if not all(key in data for key in ("config", "log", "eval")):
        return
    cfg, log, evaluation = data["config"], data["log"], data["eval"]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for field, label in (("train_loss", "Training minibatch"), ("val_loss", "Validation")):
        ax.plot([row["step"] for row in log], [row[field] for row in log], label=label, marker=".")
    ax.axhline(baseline["frequency"]["loss"], color=COLORS[2], ls=":", label="Frequency baseline")
    ax.set(xlabel="Training step", ylabel="Cross-entropy\n[nats/character]", ylim=(0, None),
           title=f'Saved-weight validation: loss {evaluation["loss"]:.3f}, PPL {evaluation["ppl"]:.2f}')
    ax.legend(frameon=False)
    save(fig, "loss", f'JAX, width {cfg["n_embd"]}, {cfg["n_layer"]} layers, {cfg["n_head"]} heads, seed {cfg["seed"]}. Validation: fixed 128-character blocks, tail included.')

    runs = data["comparison"]
    if len(runs) < 2:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for run in runs:
        rows = run["log"]
        ax.plot([r["tokens"]/1e6 for r in rows], [r["val_loss"] for r in rows], marker=".",
                label=f'Width {run["config"]["n_embd"]}: PPL {run["eval"]["ppl"]:.2f}')
    ax.set(xlabel="Training tokens [million]", ylabel="Validation cross-entropy\n[nats/character]", ylim=(0, None),
           title="Equal-token comparison")
    ax.legend(frameon=False)
    save(fig, "comparison", "Same corpus, context, data order, optimization schedule, and backend. One pair of runs; no seed-averaged claim.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    if args.artifacts:
        data = load_artifacts(args.artifacts)
        data["source"] = str(args.artifacts)
        (HERE / "reference.json").write_text(json.dumps(data, indent=2) + "\n")
    else:
        data = json.loads((HERE / "reference.json").read_text())
    generate(data)
