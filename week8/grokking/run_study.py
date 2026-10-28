"""Run the grokking study and write the measurements the sheet quotes.

    .venv/bin/python week8/grokking/run_study.py            # everything (cached per run)
    .venv/bin/python week8/grokking/run_study.py --only main # one group

Each training run is cached in week8/results/grokking-runs/<name>.npz so groups can be
rerun or extended without retraining. Outputs:
  week8/data/grokking-measurements.json   every number the sheet quotes
  week8/data/grokking-evidence.npz        seed-0 snapshots (init, memorized, final),
                                          the split, and every run's curves
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fourier  # noqa: E402
import grok  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "week8/results/grokking-runs"
OUT_JSON = ROOT / "week8/data/grokking-measurements.json"
OUT_NPZ = ROOT / "week8/data/grokking-evidence.npz"

# The protocol the sheet asks for (frozen after the exploration in README.md).
PROTOCOL = dict(model="transformer", p=97, frac=0.3, wd=1.0, lr=3e-3, steps=10000,
                arch=dict(d_model=64, n_heads=4, d_mlp=256), log_every=100, stop_after_grok=1000)
SEEDS = [0, 1, 2, 3, 4]
K = 5

GROUPS = {
    "main": [(f"main_s{s}", dict(seed=s)) for s in SEEDS],
    "frac05": [(f"frac05_s{s}", dict(seed=s, frac=0.5)) for s in SEEDS[:3]],
    "wd0": [(f"wd0_s{s}", dict(seed=s, wd=0.0, stop_after_grok=None)) for s in SEEDS[:3]],
    "random": [("random_s0", dict(seed=0, labels="random", steps=3000))],
    "all": [("all_s0", dict(seed=0, frac=1.0, steps=3000)),
            ("all_lr1e-3_s0", dict(seed=0, frac=1.0, steps=3000, lr=1e-3))],
}


def run(name, overrides, force=False):
    path = CACHE / f"{name}.npz"
    if path.exists() and not force:
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        hist = {k: z["h_" + k] for k in meta["history_keys"]}
        snaps = {}
        for n in z.files:
            if n.startswith("snap__"):
                _, s, prm = n.split("__")
                snaps.setdefault(s, {})[prm] = z[n]
        meta.update(history=hist, snapshots=snaps, split={"train": z["train"], "test": z["test"], "labels": z["labels"]})
        return meta
    cfg = {**PROTOCOL, **overrides}
    print(f"[{name}] {cfg}", flush=True)
    r = grok.train(**cfg, verbose=True)
    print(f"[{name}] memorized {r['memorization_step']} grokked {r['grokking_step']} in {r['seconds']:.0f} s", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    meta = {k: r[k] for k in ["config", "memorization_step", "grokking_step", "steps_run", "seconds"]}
    meta["history_keys"] = list(r["history"])
    arrays = {"h_" + k: np.asarray(v) for k, v in r["history"].items()}
    for s, prm in r["snapshots"].items():
        for pn, v in prm.items():
            arrays[f"snap__{s}__{pn}"] = np.asarray(v, np.float32)
    np.savez_compressed(path, meta=json.dumps(meta), train=r["split"]["train"], test=r["split"]["test"],
                        labels=r["split"]["labels"], **arrays)
    return run(name, overrides)


def first_step(hist, key, thr):
    idx = np.nonzero(np.asarray(hist[key]) >= thr)[0]
    return int(hist["step"][idx[0]]) if idx.size else None


def spread(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return dict(values=xs, median=float(np.median(xs)), min=float(min(xs)), max=float(max(xs)),
                mean=float(np.mean(xs)), std=float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0)


def summarize(name, r):
    h = r["history"]
    mem, grk = r["memorization_step"], r["grokking_step"]
    return dict(name=name, seed=r["config"]["seed"], memorization_step=mem, grokking_step=grk,
                delay_ratio=(grk / mem) if (mem and grk) else None,
                test_acc_at_memorization=float(h["test_acc"][list(h["step"]).index(mem)]) if mem else None,
                final_train_acc=float(h["train_acc"][-1]),
                final_test_acc=None if np.isnan(h["test_acc"][-1]) else float(h["test_acc"][-1]),
                norm_init=float(h["norm"][0]),
                norm_at_memorization=float(h["norm"][list(h["step"]).index(mem)]) if mem else None,
                norm_final=float(h["norm"][-1]), steps_run=int(r["steps_run"]), seconds=float(r["seconds"]))


def mechanism(r, rng_seed=0, n_random=10):
    cfg = r["config"]
    model, p = cfg["model"], cfg["p"]
    snaps = r["snapshots"]
    (a, b, c), tr, te = grok.make_split(p, cfg["frac"], cfg["data_seed"], cfg["labels"])
    a_te, b_te, c_te = a[te], b[te], c[te]
    out = {"k": K, "snapshots": {}}
    for s in ["init", "memorized", "final"]:
        E = grok.residue_embedding(model, snaps[s])
        pw = fourier.frequency_power(E)
        out["snapshots"][s] = dict(
            top_k_fraction=fourier.top_k_fraction(E, K),
            key_frequencies=fourier.key_frequencies(E, K),
            power=pw.tolist(),
            train_acc=grok.accuracy(model, snaps[s], p, (a[tr], b[tr]), c[tr]),
            test_acc=grok.accuracy(model, snaps[s], p, (a_te, b_te), c_te))
    P = snaps["final"]
    E = grok.residue_embedding(model, P)
    key = fourier.key_frequencies(E, K)
    out["key_frequencies"] = key
    # Control: the same spectrum computed after shuffling the residue order of the rows.
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(p)
    out["shuffled_order_top_k_fraction"] = fourier.top_k_fraction(E[..., perm, :], K)
    # Ablations on the residue embedding of the final weights.
    acc = lambda Ex: grok.accuracy(model, P, p, (a_te, b_te), c_te, E=Ex)
    los = lambda Ex: grok.loss_on(model, P, a_te, b_te, c_te, E=Ex)
    ab = {"none": dict(test_acc=acc(E.astype(np.float32)), test_loss=los(E.astype(np.float32))),
          "keep_key": dict(freqs=key, test_acc=acc(fourier.keep_only(E, key)), test_loss=los(fourier.keep_only(E, key))),
          "remove_key": dict(freqs=key, test_acc=acc(fourier.remove(E, key)), test_loss=los(fourier.remove(E, key)))}
    nonkey = [f for f in range(1, (p - 1) // 2 + 1) if f not in key]
    rand = []
    for _ in range(n_random):
        fr = sorted(rng.choice(nonkey, size=len(key), replace=False).tolist())
        rand.append(dict(freqs=fr, test_acc=acc(fourier.remove(E, fr)), test_loss=los(fourier.remove(E, fr))))
    ab["remove_random_nonkey"] = rand
    ta = [x["test_acc"] for x in rand]
    ab["remove_random_nonkey_summary"] = dict(min=min(ta), max=max(ta), mean=float(np.mean(ta)))
    # Keeping k random non-key frequencies instead of the key ones.
    fr = sorted(rng.choice(nonkey, size=len(key), replace=False).tolist())
    ab["keep_random_nonkey"] = dict(freqs=fr, test_acc=acc(fourier.keep_only(E, fr)))
    out["ablation"] = ab
    # Same ablations on the memorized snapshot: key frequencies not yet there.
    Em = grok.residue_embedding(model, snaps["memorized"])
    out["memorized_keep_key_train_acc"] = grok.accuracy(model, snaps["memorized"], p, (a[tr], b[tr]), c[tr],
                                                        E=fourier.keep_only(Em, key))
    return out


def leakage(r):
    cfg = r["config"]
    p = cfg["p"]
    (a, b, c), tr, te = grok.make_split(p, cfg["frac"], cfg["data_seed"])
    mem = r["snapshots"]["memorized"]
    model = cfg["model"]
    honest = grok.accuracy(model, mem, p, (a[te], b[te]), c[te])
    leaky = np.arange(p * p)                       # a "test set" that contains the training pairs
    leaky_acc = grok.accuracy(model, mem, p, (a[leaky], b[leaky]), c[leaky])
    pairs_tr = set(zip(a[tr].tolist(), b[tr].tolist()))
    swapped = float(np.mean([(bb, aa) in pairs_tr for aa, bb in zip(a[te].tolist(), b[te].tolist())]))
    return dict(overlap=int(np.intersect1d(tr, te).size), test_acc_at_memorization=honest,
                all_pairs_as_test_acc_at_memorization=leaky_acc,
                fraction_test_pairs_whose_swap_is_in_train=swapped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(GROUPS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    runs = {}
    for g in args.only:
        for name, ov in GROUPS[g]:
            runs[name] = run(name, ov, args.force)
    # Aggregate from whatever is cached.
    for g in GROUPS:
        for name, ov in GROUPS[g]:
            if name not in runs and (CACHE / f"{name}.npz").exists():
                runs[name] = run(name, ov)
    meas = {"protocol": {**PROTOCOL, "seeds": SEEDS, "optimizer": "AdamW full batch, betas (0.9, 0.98), 10-step warmup",
                         "grokking_step": "first logged step with test accuracy >= 0.99",
                         "memorization_step": "first logged step with train accuracy >= 0.99"},
            "groups": {}}
    for g in GROUPS:
        rows = [summarize(n, runs[n]) for n, _ in GROUPS[g] if n in runs]
        if not rows:
            continue
        meas["groups"][g] = dict(
            runs=rows,
            memorization_step=spread([r["memorization_step"] for r in rows]),
            grokking_step=spread([r["grokking_step"] for r in rows]),
            delay_ratio=spread([r["delay_ratio"] for r in rows]),
            final_test_acc=spread([r["final_test_acc"] for r in rows]),
            norm_at_memorization=spread([r["norm_at_memorization"] for r in rows]),
            norm_final=spread([r["norm_final"] for r in rows]),
            seconds=spread([r["seconds"] for r in rows]),
            seconds_per_step=spread([r["seconds"] / r["steps_run"] for r in rows]),
            grokked=sum(r["grokking_step"] is not None for r in rows), n=len(rows))
    if "main_s0" in runs:
        r0 = runs["main_s0"]
        meas["mechanism"] = mechanism(r0)
        meas["mechanism"]["key_frequencies_by_seed"] = {
            n: fourier.key_frequencies(grok.residue_embedding(PROTOCOL["model"], runs[n]["snapshots"]["final"]), K)
            for n, _ in GROUPS["main"] if n in runs}
        meas["mechanism"]["top_k_fraction_final_by_seed"] = {
            n: fourier.top_k_fraction(grok.residue_embedding(PROTOCOL["model"], runs[n]["snapshots"]["final"]), K)
            for n, _ in GROUPS["main"] if n in runs}
        per_seed = {}
        for n, _ in GROUPS["main"]:
            if n not in runs:
                continue
            r = runs[n]
            cfg = r["config"]
            (a, b, c), tr, te = grok.make_split(cfg["p"], cfg["frac"], cfg["data_seed"])
            E = grok.residue_embedding(cfg["model"], r["snapshots"]["final"])
            key = fourier.key_frequencies(E, K)
            acc = lambda Ex: grok.accuracy(cfg["model"], r["snapshots"]["final"], cfg["p"], (a[te], b[te]), c[te], E=Ex)
            per_seed[n] = dict(key_frequencies=key,
                               top_k_fraction_init=fourier.top_k_fraction(grok.residue_embedding(cfg["model"], r["snapshots"]["init"]), K),
                               top_k_fraction_memorized=fourier.top_k_fraction(grok.residue_embedding(cfg["model"], r["snapshots"]["memorized"]), K),
                               top_k_fraction_final=fourier.top_k_fraction(E, K),
                               keep_key_test_acc=acc(fourier.keep_only(E, key)),
                               remove_key_test_acc=acc(fourier.remove(E, key)))
        meas["mechanism"]["per_seed"] = per_seed
        meas["mechanism"]["per_seed_spread"] = {
            k: spread([d[k] for d in per_seed.values()])
            for k in ["top_k_fraction_init", "top_k_fraction_memorized", "top_k_fraction_final",
                      "keep_key_test_acc", "remove_key_test_acc"]}
        meas["controls"] = {"leakage": leakage(r0)}
        arrays = {f"{s}/{k}": v for s, prm in r0["snapshots"].items() for k, v in prm.items()}
        arrays.update(train_idx=r0["split"]["train"], test_idx=r0["split"]["test"])
        for n, r in runs.items():
            for k, v in r["history"].items():
                arrays[f"curve/{n}/{k}"] = np.asarray(v, np.float32 if k != "step" else np.int32)
        np.savez_compressed(OUT_NPZ, **arrays)
    if "random_s0" in runs:
        r = runs["random_s0"]
        meas["controls"]["random_labels"] = dict(final_train_acc=float(r["history"]["train_acc"][-1]),
                                                  final_test_acc=float(r["history"]["test_acc"][-1]),
                                                  chance=1 / PROTOCOL["p"],
                                                  memorization_step=r["memorization_step"],
                                                  top_k_fraction_final=fourier.top_k_fraction(
                                                      grok.residue_embedding(PROTOCOL["model"], r["snapshots"]["final"]), K))
    if "all_s0" in runs:
        r = runs["all_s0"]
        meas["controls"]["all_pairs"] = dict(memorization_step=r["memorization_step"], test_set_size=0,
                                             final_train_acc=float(r["history"]["train_acc"][-1]),
                                             steps=int(r["steps_run"]),
                                             top_k_fraction_final=fourier.top_k_fraction(
                                                 grok.residue_embedding(PROTOCOL["model"], r["snapshots"]["final"]), K),
                                             key_frequencies=fourier.key_frequencies(
                                                 grok.residue_embedding(PROTOCOL["model"], r["snapshots"]["final"]), K))
    if "all_lr1e-3_s0" in runs:
        r = runs["all_lr1e-3_s0"]
        E = grok.residue_embedding(PROTOCOL["model"], r["snapshots"]["final"])
        meas["controls"]["all_pairs_lr1e-3"] = dict(memorization_step=r["memorization_step"], steps=int(r["steps_run"]),
                                                    final_train_acc=float(r["history"]["train_acc"][-1]),
                                                    top_k_fraction_final=fourier.top_k_fraction(E, K),
                                                    key_frequencies=fourier.key_frequencies(E, K))
    OUT_JSON.write_text(json.dumps(meas, indent=1))
    print(f"wrote {OUT_JSON} and {OUT_NPZ}")


if __name__ == "__main__":
    main()
