"""Week 8 Part 1 — semantic offsets in potion-base-8M, no PyTorch.

Run from repository root with:
  uv run --no-project --with numpy --with matplotlib --with safetensors --with huggingface_hub --with tokenizers python week8_1/part1_embeddings.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file
from tokenizers import Tokenizer

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "data" / "potion-base-8M"
EVIDENCE_DIR = BASE / "evidence"
ARTIFACTS_DIR = BASE / "artifacts"
REPO = "minishlab/potion-base-8M"
REV = "bf8b056"
SEED = 1730

RELATIONS = {
    "country -> capital": [
        ("iran", "tehran"), ("turkey", "ankara"), ("russia", "moscow"),
        ("china", "beijing"), ("france", "paris"), ("sweden", "stockholm"),
        ("japan", "tokyo"), ("peru", "lima"), ("kenya", "nairobi"),
    ],
    "male -> female": [
        ("king", "queen"), ("man", "woman"), ("he", "she"),
        ("brother", "sister"), ("father", "mother"), ("son", "daughter"),
        ("uncle", "aunt"), ("boy", "girl"), ("husband", "wife"),
    ],
    "adjective -> comparative": [
        ("big", "bigger"), ("small", "smaller"), ("fast", "faster"),
        ("dark", "darker"), ("cold", "colder"), ("old", "older"),
        ("strong", "stronger"), ("long", "longer"), ("tall", "taller"),
    ],
}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den else float("nan")


def relation_scores(emb: np.ndarray, word_ids: dict[str, int], pairs: list[tuple[str, str]]) -> np.ndarray:
    vectors = emb[[word_ids[word] for pair in pairs for word in pair]]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms == 0).any():
        raise RuntimeError("Found zero-length embedding; cannot normalize")
    vectors = vectors / norms
    offsets = vectors[1::2] - vectors[0::2]
    out = []
    for i, d in enumerate(offsets):
        others_mean = (offsets.sum(axis=0) - d) / (len(offsets) - 1)
        out.append(cosine(d, others_mean))
    return np.array(out)


def main() -> None:
    for d in (MODEL_DIR, EVIDENCE_DIR, ARTIFACTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Working directory: {BASE}")
    print(f"Downloading pinned Hugging Face files from {REPO}@{REV}...")
    paths = {}
    for name in ("model.safetensors", "tokenizer.json"):
        paths[name] = hf_hub_download(
            repo_id=REPO, filename=name, revision=REV, local_dir=str(MODEL_DIR)
        )
        print(f"  {name}: {paths[name]}")

    tensors = load_file(paths["model.safetensors"])
    print("Available tensor keys:", list(tensors))
    candidates = [(name, a) for name, a in tensors.items() if a.ndim == 2 and a.shape[1] == 256]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one 256-dimensional embedding matrix, got {[(k, v.shape) for k, v in tensors.items()]}")
    key, embedding = candidates[0]
    embedding = embedding.astype(np.float64)
    print(f"Embedding tensor '{key}' shape: {embedding.shape}")
    if embedding.shape != (29528, 256):
        raise RuntimeError(f"Unexpected embedding dimensions {embedding.shape}; expected (29528, 256)")

    tokenizer = Tokenizer.from_file(paths["tokenizer.json"])
    words = sorted({w for pairs in RELATIONS.values() for a, b in pairs for w in (a, b)})
    word_ids: dict[str, int] = {}
    for word in words:
        idx = tokenizer.token_to_id(word)
        if idx is None or idx >= len(embedding):
            raise RuntimeError(f"Missing or invalid single-token word {word!r}: id={idx}")
        # Check tokenizer tokenization does not secretly split the word into pieces.
        encoded_ids = tokenizer.encode(word, add_special_tokens=False).ids
        if encoded_ids != [idx]:
            raise RuntimeError(f"Word {word!r} does not tokenize as a single row: {encoded_ids}, direct id={idx}")
        word_ids[word] = idx
    print("Tokenizer sanity ids (PDF example):", {w: tokenizer.token_to_id(w) for w in ("cat", "kitten")})
    print(f"Verified {len(words)} distinct words all have exactly one matching token.")

    shuffled = embedding[np.random.default_rng(SEED).permutation(len(embedding))]
    results = {}
    rows = []
    for relation, pairs in RELATIONS.items():
        orig = relation_scores(embedding, word_ids, pairs)
        ctrl = relation_scores(shuffled, word_ids, pairs)
        mean_orig, mean_ctrl = float(orig.mean()), float(ctrl.mean())
        passed = bool(mean_orig >= 0.3 and abs(mean_ctrl) <= 0.15)
        results[relation] = {"original": orig, "shuffled": ctrl, "mean_original": mean_orig,
                             "mean_shuffled": mean_ctrl, "passed": passed}
        print(f"{relation:27s} original={mean_orig:+.4f}  shuffled={mean_ctrl:+.4f}  {'PASS' if passed else 'CHECK'}")
        for i, (a, b) in enumerate(pairs):
            rows.append({"relation": relation, "word_a": a, "word_b": b,
                         "cosine_original": float(orig[i]), "cosine_shuffled": float(ctrl[i])})

    csv_path = ARTIFACTS_DIR / "part1_pair_cosines.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = ARTIFACTS_DIR / "part1_summary.json"
    json_path.write_text(json.dumps({
        "model": REPO, "revision": REV, "seed": SEED,
        "embedding_shape": list(embedding.shape),
        "results": {name: {"mean_original": val["mean_original"],
                           "mean_shuffled": val["mean_shuffled"],
                           "passed": val["passed"]} for name, val in results.items()}
    }, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6.1), sharex=True, sharey=True)
    for ax, (relation, result) in zip(axes, results.items()):
        for i, c in enumerate(result["original"]):
            y = np.sqrt(max(0, 1 - c*c))
            ax.annotate("", xy=(c, y), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", lw=1.7, color="#3173ad", alpha=0.6 + 0.04*(i % 4)))
        for i, c in enumerate(result["shuffled"]):
            y = -np.sqrt(max(0, 1 - c*c))
            ax.annotate("", xy=(c, y), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color="#808080", alpha=0.55 + 0.04*(i % 4)))
        ax.annotate("", xy=(1, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", lw=1.7, color="black"))
        ax.axvline(0, color="#999", ls="--", lw=0.8)
        ax.axhline(0, color="#999", lw=0.8)
        ax.text(-1.06, 0.55, "original", color="#3173ad", rotation=90, va="center", ha="center", fontsize=9)
        ax.text(-1.06, -0.55, "shuffled", color="#808080", rotation=90, va="center", ha="center", fontsize=9)
        ax.set_title(f"{relation}\nmean: {result['mean_original']:.2f} / {result['mean_shuffled']:.2f}", fontsize=12)
        ax.set_xlim(-1.17, 1.07)
        ax.set_ylim(-1.12, 1.12)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("cosine with other 8 offsets")
        ax.grid(alpha=0.12)
    fig.suptitle("Week 8 Part 1 — semantic relations as embedding directions", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    image_path = EVIDENCE_DIR / "part1_embedding_directions.png"
    fig.savefig(image_path, dpi=190, bbox_inches="tight")
    plt.close(fig)

    readme_path = BASE / "README.md"
    command = ("uv run --no-project --with numpy --with matplotlib --with safetensors "
               "--with huggingface_hub --with tokenizers python week8_1/part1_embeddings.py")
    section = (
        "## Part 1: From tokens to vectors (embeddings)\n\n"
        f"Run from repository root:\n\n```bash\n{command}\n```\n\n"
        f"Source: `{REPO}`, revision `{REV}`. Model: `{tuple(embedding.shape)}`. "
        f"Control: shuffled embedding rows, seed `{SEED}`.\n\n"
        "Word vectors were L2-normalized, and each pair's offset was compared by cosine "
        "with the mean of the other eight pair offsets (leave-one-out).\n\n"
        "| Relation | Original mean | Shuffled mean | Acceptance |\n"
        "|---|---:|---:|---|\n" +
        "\n".join(f"| {name} | {r['mean_original']:.4f} | {r['mean_shuffled']:.4f} | {'PASS' if r['passed'] else 'CHECK'} |"
                  for name, r in results.items()) +
        "\n\nAcceptance: original means >= 0.3; shuffled means between -0.15 and 0.15.\n\n"
        "Evidence: `evidence/part1_embedding_directions.png`\n\n"
        "Raw pair results: `artifacts/part1_pair_cosines.csv`; "
        "summary/settings: `artifacts/part1_summary.json` (both ignored by git).\n"
    )
    start_mark, end_mark = "<!-- BEGIN WEEK8 PART1 -->", "<!-- END WEEK8 PART1 -->"
    old = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# Week 8 — Language models and the transformer\n"
    new_section = f"{start_mark}\n{section}{end_mark}"
    if start_mark in old and end_mark in old:
        prefix, remainder = old.split(start_mark, 1)
        _, suffix = remainder.split(end_mark, 1)
        content = prefix + new_section + suffix
    else:
        content = old.rstrip() + "\n\n" + new_section + "\n"
    readme_path.write_text(content, encoding="utf-8")
    print(f"\nSaved figure: {image_path}")
    print(f"Saved CSV:    {csv_path}")
    print(f"Saved JSON:   {json_path}")
    print(f"Updated README: {readme_path}")
    print("Overall:", "PASS" if all(r["passed"] for r in results.values()) else "CHECK — examine failures before submitting")


if __name__ == "__main__":
    main()
