"""Independent NumPy check of the grokking answer key from saved weights.

Reads week8/data/grokking-evidence.npz and grokking-measurements.json, then
recomputes with NumPy only (no JAX, no fourier.py):
  * train and test accuracy of the memorized and final snapshots (own forward pass);
  * the frequency power spectrum of the residue embedding via np.fft.rfft;
  * the top-k power fraction and the key frequencies;
  * the test accuracy after keeping only / removing the key frequencies (np.fft based
    filtering) and after removing the same number of other frequencies.
Every recomputed number must agree with the JSON; exit status 1 otherwise.

    .venv/bin/python week8/grokking/numpy_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def softmax_rows(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def transformer_logits(P, a, b, E=None):
    p = P["W_U"].shape[1]
    WE = P["W_E"].astype(np.float64).copy()
    if E is not None:
        WE[:p] = E
    x = np.stack([WE[a], WE[b], np.broadcast_to(WE[p], (len(a), WE.shape[1]))], 1) + P["W_pos"]
    out = x[:, 2].copy()
    dh = P["W_Q"].shape[-1]
    for h in range(P["W_Q"].shape[0]):
        q = x[:, 2] @ P["W_Q"][h]
        k = x @ P["W_K"][h]
        v = x @ P["W_V"][h]
        att = softmax_rows((k @ q[:, :, None])[:, :, 0] / np.sqrt(dh))
        out += (att[:, :, None] * v).sum(1) @ P["W_O"][h]
    hid = np.maximum(out @ P["W_in"] + P["b_in"], 0)
    out = out + hid @ P["W_out"] + P["b_out"]
    return out @ P["W_U"]


def mlp_logits(P, a, b, E=None):
    Wa, Wb = (P["W_a"], P["W_b"]) if E is None else (E[0], E[1])
    h = Wa[a].astype(np.float64) + Wb[b]
    return (h * h) @ P["W_U"]


def residue_embedding(model, P):
    if model == "transformer":
        return P["W_E"][: P["W_U"].shape[1]].astype(np.float64)
    return np.stack([P["W_a"], P["W_b"]]).astype(np.float64)


def power(E):
    """Power per frequency 0..(p-1)/2 from the real FFT along the residue axis."""
    if E.ndim == 3:
        return sum(power(e) for e in E)
    p = E.shape[0]
    F = np.fft.rfft(E, axis=0)                      # (p//2 + 1, d)
    pw = (np.abs(F) ** 2).sum(1)
    pw[1:] *= 2.0 / p                               # Parseval: cos and sin parts
    pw[0] /= p
    return pw


def fft_filter(E, freqs, keep):
    """Keep (with the constant) or remove the given frequencies along the residue axis."""
    if E.ndim == 3:
        return np.stack([fft_filter(e, freqs, keep) for e in E])
    p = E.shape[0]
    F = np.fft.rfft(E, axis=0)
    mask = np.zeros(F.shape[0], bool)
    mask[list(freqs)] = True
    if keep:
        mask[0] = True
        F = np.where(mask[:, None], F, 0)
    else:
        F = np.where(mask[:, None], 0, F)
    return np.fft.irfft(F, n=p, axis=0)


def main():
    meas = json.loads((ROOT / "week8/data/grokking-measurements.json").read_text())
    z = np.load(ROOT / "week8/data/grokking-evidence.npz")
    model, p, k = meas["protocol"]["model"], meas["protocol"]["p"], meas["mechanism"]["k"]
    logits = transformer_logits if model == "transformer" else mlp_logits
    snap = lambda s: {n.split("/", 1)[1]: z[n].astype(np.float64) for n in z.files if n.startswith(s + "/")}
    a = np.repeat(np.arange(p), p); b = np.tile(np.arange(p), p); c = (a + b) % p
    tr, te = z["train_idx"], z["test_idx"]
    failures = []

    def agree(name, got, want, tol):
        ok = abs(got - want) <= tol
        print(f"{'ok ' if ok else 'BAD'} {name:44s} numpy {got:.6f}  answer key {want:.6f}")
        if not ok:
            failures.append(name)

    assert np.intersect1d(tr, te).size == 0 and tr.size + te.size == p * p, "split is not a partition"
    acc = lambda P, idx, E=None: float(np.mean(np.argmax(logits(P, a[idx], b[idx], E), -1) == c[idx]))
    m = meas["mechanism"]
    for s in ["memorized", "final"]:
        P = snap(s)
        agree(f"{s}: train accuracy", acc(P, tr), m["snapshots"][s]["train_acc"], 1e-3)
        agree(f"{s}: test accuracy", acc(P, te), m["snapshots"][s]["test_acc"], 1e-3)
    for s in ["init", "memorized", "final"]:
        pw = power(residue_embedding(model, snap(s)))[1:]
        agree(f"{s}: top-{k} power fraction", float(np.sort(pw)[::-1][:k].sum() / pw.sum()),
              m["snapshots"][s]["top_k_fraction"], 1e-4)
    P = snap("final")
    E = residue_embedding(model, P)
    pw = power(E)[1:]
    key = sorted((np.argsort(pw)[::-1][:k] + 1).tolist())
    print(f"key frequencies numpy {key}  answer key {m['key_frequencies']}")
    if key != m["key_frequencies"]:
        failures.append("key frequencies")
    ab = m["ablation"]
    agree("keep only key frequencies: test acc", acc(P, te, fft_filter(E, key, True)), ab["keep_key"]["test_acc"], 2e-3)
    agree("remove key frequencies: test acc", acc(P, te, fft_filter(E, key, False)), ab["remove_key"]["test_acc"], 2e-3)
    for i, ctrl in enumerate(ab["remove_random_nonkey"]):
        agree(f"remove non-key set {i}: test acc", acc(P, te, fft_filter(E, ctrl["freqs"], False)), ctrl["test_acc"], 2e-3)
    if failures:
        print("FAIL:", failures)
        sys.exit(1)
    print("PASS: NumPy recomputation agrees with the answer key")


if __name__ == "__main__":
    main()
