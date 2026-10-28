#!/usr/bin/env python3
"""Week 8 Part 4 — attention as a context-built key–value memory.

Run from any directory with:
    uv run --no-project --with numpy --with matplotlib python week8_1/part4_attention.py
or from the week8_1 folder:
    uv run --no-project --with numpy --with matplotlib python part4_attention.py

Uses only NumPy and Matplotlib; no extra downloads, GPU, or JAX necessary.
Matches Week 8 learning sheet, Part 4 DO/VERIFY, pp. 13–15.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = 64
BETA = 8.0
SEED = 1730
MEMORY_COUNTS = (64, 256, 1024, 4096)
METHODS = ("Outer product", "MLP, ReLU", "MLP, ReLU²", "Attention, softmax β=8")
SCRIPT_DIR = Path(__file__).resolve().parent
EVIDENCE = SCRIPT_DIR / "evidence"
ARTIFACTS = SCRIPT_DIR / "artifacts"


def make_character() -> np.ndarray:
    """A font-independent, 32x32 black/white pixel-art glyph resembling 亮.

    The supplied PDF has an unextractable/unsupported glyph at 'character □',
    so we use a reproducible hand-drawn glyph, not a downloaded font.
    Returns ±1 pixels with +1 for ink and -1 for background.
    """
    y, x = np.mgrid[0:32, 0:32]
    ink = np.zeros((32, 32), dtype=bool)

    def stroke(x0: float, y0: float, x1: float, y1: float, thickness: float = 1.0) -> None:
        nonlocal ink
        dx, dy = x1 - x0, y1 - y0
        t = np.clip(((x - x0) * dx + (y - y0) * dy) / max(dx * dx + dy * dy, 1e-9), 0, 1)
        dist = np.hypot(x - (x0 + t * dx), y - (y0 + t * dy))
        ink |= dist <= thickness / 2

    # Upper dot, long stroke and inner square.
    stroke(16, 1, 16, 3, 1.5)
    stroke(5, 5, 27, 5, 1.3)
    stroke(10, 9, 22, 9, 1.1)
    stroke(10, 9, 10, 15, 1.1)
    stroke(22, 9, 22, 15, 1.1)
    stroke(10, 15, 22, 15, 1.1)
    # Lower cover and two legs, a simple 亮-shaped pixel symbol.
    stroke(6, 19, 26, 19, 1.4)
    stroke(7, 19, 7, 23, 1.1)
    stroke(25, 19, 25, 23, 1.1)
    stroke(12, 22, 11, 27, 1.1)
    stroke(11, 27, 5, 30, 1.1)
    stroke(19, 22, 19, 29, 1.1)
    stroke(19, 29, 26, 29, 1.1)
    assert np.any(ink) and np.any(~ink)
    return np.where(ink, 1.0, -1.0).astype(np.float32)


def recall_one(M: int, rng: np.random.Generator, picture: np.ndarray) -> list[dict]:
    # M random unit keys in R^64; memory 0 is queried.
    keys = rng.standard_normal((M, D)).astype(np.float32)
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    values = rng.integers(0, 2, size=(M, 1024), dtype=np.int8)
    values = (2 * values - 1).astype(np.float32)
    values[0] = picture.reshape(-1)

    scores = keys @ keys[0]
    relu = np.maximum(scores, 0.0)
    # Softmax uses a stable, shifted exponential and explicit normalization.
    logits = BETA * scores
    exp_logits = np.exp(logits - logits.max())
    probabilities = exp_logits / exp_logits.sum()
    assert abs(float(probabilities.sum()) - 1.0) < 1e-6

    weights = (scores, relu, relu ** 2, probabilities)
    output: list[dict] = []
    for method, w in zip(METHODS, weights):
        own_weight = float(w[0])
        assert own_weight > 0.0, f"Nonpositive target weight for {method}"
        # Divide by the true stored picture's own weight, as in the PDF.
        interference = (w[1:] @ values[1:]) / own_weight
        recovered = values[0] + interference
        R = float(np.linalg.norm(interference) / np.linalg.norm(values[0]))
        output.append({
            "method": method,
            "M": M,
            "R": R,
            "own_weight": own_weight,
            "recovered": recovered.reshape(32, 32),
        })
    return output


def plot_figure(results: list[list[dict]], path: Path, picture: np.ndarray) -> None:
    # 4 rows: memory count; 4 columns: stores. Dot area grows with recall pixel.
    fig, axs = plt.subplots(4, 4, figsize=(12.5, 12.2), constrained_layout=True)
    yy, xx = np.mgrid[0:32, 0:32]
    xx, yy = xx.ravel(), yy.ravel()
    for i, row in enumerate(results):
        for j, item in enumerate(row):
            ax = axs[i, j]
            recalled = item["recovered"].ravel()
            # Ink (+1) should give visible dots, background (-1) should not.
            # Noise produces spurious dots when R is large.
            sizes = np.clip((recalled + 1.0) * 6.5, 0.0, 24.0)
            good = item["R"] < 1.0
            color = "#be3838" if good else "#717171"
            mask = sizes > 0.3
            ax.scatter(xx[mask], yy[mask], s=sizes[mask], c=color, marker='o', linewidths=0)
            ax.set_xlim(-1, 32)
            ax.set_ylim(32, -1)
            ax.set_aspect('equal')
            ax.axis('off')
            if i == 0:
                ax.set_title(METHODS[j], fontsize=11.0, pad=12)
            if j == 0:
                ax.text(-0.05, .5, f"M = {item['M']:,}", rotation=90,
                        transform=ax.transAxes, ha='right', va='center', fontsize=10.5)
            ax.text(0.50, -0.02, f"R = {item['R']:.4f}", transform=ax.transAxes,
                    ha='center', va='top', fontsize=10.0)

    fig.suptitle("Week 8 Part 4 — Recall from key–value memories (d = 64, β = 8)", fontsize=14.5)
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_small_glyph(picture: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.3, 3.3))
    ax.imshow(picture, cmap='gray_r', vmin=-1, vmax=1, interpolation='nearest')
    ax.set_title("Stored 32 × 32 binary picture", fontsize=11)
    ax.axis('off')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def update_readme() -> None:
    readme = SCRIPT_DIR / 'README.md'
    text = readme.read_text(encoding='utf-8') if readme.exists() else '# Week 8 experiments\n'
    header = '## Part 4: Attention — recall from context-built memory'
    if header in text:
        return
    addition = f'''\n\n{header}

Reproduce from `week8_1/`:

```bash
uv run --no-project --with numpy --with matplotlib python part4_attention.py
```

In d=64 with β=8 and seed={SEED}, store a 32×32 binary character
(+1 ink, -1 background) among M=64/256/1024/4096 memories.
Compare outer-product, ReLU, ReLU² and softmax stores.
Normalize each recalled image by the target memory's own weight;
R is the L2 norm of interference divided by L2 norm of target.

- Main evidence: `evidence/part4_attention_recall.png` (4×4 dot panels)
- Original picture: `evidence/part4_original_picture.png`
- Raw results: `artifacts/part4_recall_ratios.csv` and `artifacts/part4_summary.json`
- Implementation: `part4_attention.py`

The character in the supplied PDF is not machine-readable (missing glyph);
this experiment uses a reproducible, manually drawn 32×32 亮-like glyph.
The image's shape does not affect the theoretical R comparisons.
The color of each panel is red when R < 1, gray otherwise.
'''
    readme.write_text(text.rstrip() + addition, encoding='utf-8')


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    picture = make_character()
    print(f"Part 4: d={D}, beta={BETA}, seed={SEED}", flush=True)
    print("Four methods, four memory sizes; single reproducible draw per size.", flush=True)
    data = []
    for M in MEMORY_COUNTS:
        row = recall_one(M, rng, picture)
        data.append(row)
        print(f"M = {M:4d}: " + " | ".join(
            f"{r['method']}: R={r['R']:.4f}" for r in row
        ), flush=True)

    # Numerical checks grounded in the PDF Part 4 VERIFY.
    r1024 = {r['method']: r['R'] for r in data[2]}
    r4096 = {r['method']: r['R'] for r in data[3]}
    checks = {
        'outer_M1024_within_15pct_of_theory_4': abs(r1024[METHODS[0]] / 4.0 - 1) <= .15,
        'relu_M1024_within_15pct_of_theory_2_828': abs(r1024[METHODS[1]] / math.sqrt(1024/(2*D)) - 1) <= .15,
        'relu2_M4096_R_above_one': r4096[METHODS[2]] > 1.0,
        'softmax_all_R_below_0_1': all(row[3]['R'] < 0.1 for row in data),
        'softmax_M4096_theoretical_neighborhood': 0.03 < r4096[METHODS[3]] < 0.09,
    }
    print("\n=== VERIFY ===", flush=True)
    for name, good in checks.items():
        print(f"{name}: {'PASS' if good else 'CHECK'}", flush=True)

    plot_figure(data, EVIDENCE / 'part4_attention_recall.png', picture)
    plot_small_glyph(picture, EVIDENCE / 'part4_original_picture.png')
    with (ARTIFACTS / 'part4_recall_ratios.csv').open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=['M', 'method', 'R', 'own_weight'])
        wr.writeheader()
        for row in data:
            for r in row:
                wr.writerow({key: r[key] for key in ('M','method','R','own_weight')})
    payload = {
        'settings': {'d': D, 'beta': BETA, 'seed': SEED, 'M': MEMORY_COUNTS,
                     'repetitions': 1, 'picture': '32x32 hand-drawn binary glyph (+1/-1)'},
        'results': [{key: r[key] for key in ('M','method','R','own_weight')} for row in data for r in row],
        'checks': checks,
        'all_pass': all(checks.values()),
        'date_utc': datetime.now(timezone.utc).isoformat(),
    }
    (ARTIFACTS / 'part4_summary.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    update_readme()
    print(f"\nEvidence: {EVIDENCE / 'part4_attention_recall.png'}", flush=True)
    print(f"Original: {EVIDENCE / 'part4_original_picture.png'}", flush=True)
    print(f"Summary: {ARTIFACTS / 'part4_summary.json'}", flush=True)
    print("Overall:", "PASS" if payload['all_pass'] else "CHECK", flush=True)

if __name__ == '__main__':
    main()
