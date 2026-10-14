"""Week 6 Part 4: modern versus classical Hopfield retrieval on MNIST.

Run from week6/:
    .venv/bin/python -u src/part4_retrieval.py --preview
    .venv/bin/python -u src/part4_retrieval.py

Data provenance and original MNIST split are supplied by src/part1_datasets.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

# All required experiments run on the CPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
ARTIFACTS = ROOT / "artifacts"
NUM_SPINS = 784
HALF = 14 * 28
P_VALUES = (3, 10, 30, 100, 300, 1000)
BETAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
N_QUERIES = 100
SEED = 2026
MAX_CLASSICAL_SWEEPS = 50


@jax.jit
def modern_hopfield(patterns: jax.Array, queries: jax.Array, beta: float) -> jax.Array:
    """With stored pattern rows X, compute softmax(beta Q X^T) X."""
    weights = jax.nn.softmax(beta * (queries @ patterns.T), axis=-1)
    return weights @ patterns


def classical_hopfield(patterns: np.ndarray, queries: np.ndarray, *, seed: int,
                       max_sweeps: int = MAX_CLASSICAL_SWEEPS):
    """Zero-temperature asynchronous Hebbian recall, random-order sweeps.

    Uses an exact incremental update for the local fields. Each sweep has
    a newly shuffled spin order; every query is updated sequentially in that
    order. The same sweep order is shared between queries for efficiency.
    """
    p, n = patterns.shape
    assert n == NUM_SPINS
    coupling = (patterns.T @ patterns) / float(n)
    np.fill_diagonal(coupling, 0.0)
    assert np.all(np.diag(coupling) == 0)
    assert np.allclose(coupling, coupling.T, atol=1e-5)

    states = queries.copy().astype(np.float32)
    fields = states @ coupling
    rng = np.random.default_rng(seed)
    sweeps_done = 0
    remaining = np.ones(len(states), dtype=bool)

    for sweep in range(max_sweeps):
        flips_per_query = np.zeros(len(states), dtype=np.int32)
        for i in rng.permutation(n):
            old = states[:, i].copy()
            h_i = fields[:, i]
            new = np.where(h_i > 0, 1.0, np.where(h_i < 0, -1.0, old))
            delta = new - old
            changed = delta != 0
            if changed.any():
                # For a zero-diagonal symmetric coupling, every flip reduces E.
                if np.any(delta[changed] * h_i[changed] < -1e-4):
                    raise AssertionError("Asynchronous update increased energy")
                states[:, i] = new
                fields += delta[:, None] * coupling[i][None, :]
                flips_per_query += changed.astype(np.int32)
        sweeps_done = sweep + 1
        remaining = flips_per_query > 0
        if not remaining.any():
            break

    return states, sweeps_done, int(remaining.sum())


def make_queries(patterns: np.ndarray, random_bottom: np.ndarray):
    target_indices = np.arange(len(random_bottom)) % len(patterns)
    targets = patterns[target_indices]
    queries = targets.copy()
    queries[:, HALF:] = random_bottom
    assert np.array_equal(queries[:, :HALF], targets[:, :HALF])
    assert np.all(np.isin(queries, [-1, 1]))
    return targets, target_indices, queries


def success_metrics(outputs: np.ndarray, targets: np.ndarray,
                    target_is_nearest: np.ndarray):
    # sign(0) is defined as +1 for a deterministic binary-state comparison.
    retrieved_spins = np.where(outputs >= 0, 1.0, -1.0)
    overlap = (retrieved_spins * targets).mean(axis=1)
    success = (overlap >= 0.95) & target_is_nearest
    return float(np.mean(success)), float(np.mean(overlap)), retrieved_spins


def plot_success(rows: list[dict], suffix: str):
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for beta in BETAS:
        chosen = sorted((r for r in rows if r["model"] == "modern" and r["beta"] == beta),
                        key=lambda r: r["P"])
        if chosen:
            ax.plot([r["P"] for r in chosen], [r["success"] for r in chosen],
                    marker="o", label=f"Modern beta={beta:g}")
    classic = sorted((r for r in rows if r["model"] == "classical"),
                     key=lambda r: r["P"])
    ax.plot([r["P"] for r in classic], [r["success"] for r in classic],
            marker="s", linewidth=2, linestyle="--", color="black",
            label="Classical Hopfield")
    ax.set_xscale("log")
    ps = sorted(set(r["P"] for r in rows))
    ax.set_xticks(ps)
    ax.set_xticklabels([str(p) for p in ps])
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Number of stored MNIST patterns P")
    ax.set_ylabel("Retrieval success fraction")
    ax.set_title("Modern vs classical Hopfield retrieval")
    ax.grid(alpha=0.22)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    fig.tight_layout()
    filename = EVIDENCE / f"part4_retrieval_success{suffix}.png"
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return filename


def plot_gallery(targets: np.ndarray, queries: np.ndarray,
                 modern: dict[float, np.ndarray], classical: np.ndarray,
                 suffix: str):
    columns = [("Target", targets), ("Corrupted\nquery", queries)]
    columns += [(f"beta={b:g}", modern[b]) for b in BETAS if b in modern]
    columns += [("Classical", classical)]
    n_rows = min(4, len(targets))
    fig, axes = plt.subplots(n_rows, len(columns),
                             figsize=(1.6 * len(columns), 2.0 * n_rows),
                             squeeze=False)
    for r in range(n_rows):
        for c, (name, block) in enumerate(columns):
            ax = axes[r, c]
            ax.imshow(block[r].reshape(28, 28), cmap="gray", vmin=-1, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"Query {r+1}")
    fig.suptitle("P=100: retrieval from a corrupted lower half", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output = EVIDENCE / f"part4_retrieval_gallery_p100{suffix}.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true",
                        help="Quick smoke test using P=3,10,100 and 12 queries")
    args = parser.parse_args()

    from part1_datasets import load_mnist

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    p_values = (3, 10, 100) if args.preview else P_VALUES
    betas = (0.003, 0.03, 0.3) if args.preview else BETAS
    n_queries = 12 if args.preview else N_QUERIES
    suffix = "_preview" if args.preview else ""

    imgs, _ = load_mnist("train")
    assert len(imgs) == 60000
    # Each stored digit is a spin configuration of +/-1, in original order.
    all_patterns = (2.0 * (imgs[:max(p_values)].reshape(-1, NUM_SPINS) >= 128)
                    .astype(np.float32) - 1.0)
    rng = np.random.default_rng(SEED)
    random_bottom = (2.0 * rng.integers(0, 2, size=(n_queries, HALF),
                                        dtype=np.int8).astype(np.float32) - 1.0)

    results = []
    p100_gallery = None
    started = time.perf_counter()
    print(f"Mode: {'preview' if args.preview else 'full'}", flush=True)
    print(f"P={p_values}, beta={betas}, queries per P={n_queries}", flush=True)

    for p in p_values:
        t0 = time.perf_counter()
        patterns = all_patterns[:p]
        targets, target_indices, queries = make_queries(patterns, random_bottom)
        # Success also requires that the target is the nearest stored pattern to the query.
        scores = queries @ patterns.T
        nearest = scores.argmax(axis=1)
        target_is_nearest = nearest == target_indices
        fraction_nearest = float(target_is_nearest.mean())
        print(f"\nP={p:4d}: target nearest in {fraction_nearest:.1%} of queries", flush=True)

        retrieved_classical, sweeps, unfinished = classical_hopfield(
            patterns, queries, seed=SEED + 1000 + p)
        classical_rate, classical_overlap, _ = success_metrics(
            retrieved_classical, targets, target_is_nearest)
        results.append({"P": p, "model": "classical", "beta": None,
                        "success": classical_rate, "mean_overlap": classical_overlap,
                        "fraction_target_nearest": fraction_nearest,
                        "sweeps": sweeps, "nonconverged_queries": unfinished})
        print(f"  Classical: success={classical_rate:.3f}, "
              f"sweeps={sweeps}, nonconverged={unfinished}", flush=True)

        jp = jnp.asarray(patterns)
        jq = jnp.asarray(queries)
        modern_for_gallery = {}
        for beta in betas:
            output = np.asarray(modern_hopfield(jp, jq, jnp.float32(beta)))
            rate, mean_overlap, _ = success_metrics(output, targets, target_is_nearest)
            results.append({"P": p, "model": "modern", "beta": beta,
                            "success": rate, "mean_overlap": mean_overlap,
                            "fraction_target_nearest": fraction_nearest})
            print(f"  Modern beta={beta:>5g}: success={rate:.3f}", flush=True)
            if p == 100:
                modern_for_gallery[beta] = output

        if p == 100:
            p100_gallery = (targets, queries, modern_for_gallery, retrieved_classical)
            # Raw data required to reproduce and inspect the gallery.
            np.savez_compressed(
                ARTIFACTS / f"part4_retrieval_p100{suffix}.npz",
                targets=targets, queries=queries, classical=retrieved_classical,
                target_indices=target_indices,
                **{f"beta_{beta:g}": out for beta, out in modern_for_gallery.items()})
        print(f"  Elapsed for P={p}: {time.perf_counter()-t0:.1f}s", flush=True)

    # Success plot uses all beta values available for the requested mode.
    plot = plot_success(results, suffix)
    print("Saved:", plot, flush=True)
    if p100_gallery is not None:
        gallery = plot_gallery(*p100_gallery, suffix=suffix)
        print("Saved:", gallery, flush=True)

    csv_path = ARTIFACTS / f"part4_retrieval_success{suffix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["P", "model", "beta", "success",
            "mean_overlap", "fraction_target_nearest", "sweeps", "nonconverged_queries"],
            extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "settings": {"P": p_values, "betas": betas, "queries_per_P": n_queries,
                     "seed": SEED, "half_observed": 392,
                     "nearest_rule": "argmax of query dot stored pattern",
                     "success_rule": "sign(output) overlap >= 0.95 AND target nearest",
                     "classical_max_sweeps": MAX_CLASSICAL_SWEEPS,
                     "elapsed_seconds": time.perf_counter() - started},
        "results": results,
    }
    summary_path = ARTIFACTS / f"part4_retrieval_summary{suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved:", csv_path, flush=True)
    print("Saved:", summary_path, flush=True)

    if not args.preview:
        print("\n===== Reference thresholds =====", flush=True)
        modern_03 = [r for r in results if r["model"] == "modern" and r["beta"] == 0.3]
        classic = [r for r in results if r["model"] == "classical"]
        for r in modern_03:
            print(f"P={r['P']:4d}, beta=0.3: {r['success']:.3f} "
                  f"({'PASS' if r['success'] >= 0.95 else 'BELOW REFERENCE'})", flush=True)
        print("Classical success rates:",
              [(r["P"], r["success"]) for r in classic], flush=True)
        if any(r["nonconverged_queries"] for r in classic):
            print("WARNING: Some classical recalls did not converge within the sweep cap.",
                  flush=True)
        print("Reference comparison is diagnostic; results are never forced to pass.", flush=True)
    print("Part 4 retrieval experiment completed.", flush=True)


if __name__ == "__main__":
    main()
