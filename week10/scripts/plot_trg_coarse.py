#!/usr/bin/env python3
"""Plot existing periodic-TRG coarse-scan evidence. This script does not run TRG."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SIZES = (64, 256, 1024)
EXPECTED_COLUMNS = {"L", "K", "chi", "g"}


def read_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No data rows in {path}")
    missing = EXPECTED_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return rows


def analyze_raw(path: Path):
    rows = read_rows(path)
    rows.sort(key=lambda r: float(r["K"]))
    k = np.array([float(r["K"]) for r in rows], dtype=float)
    g = np.array([float(r["g"]) for r in rows], dtype=float)
    if len(k) < 5:
        raise ValueError(f"Need at least five K points in {path}")
    dk = np.diff(k)
    h = float(np.median(dk))
    if not np.allclose(dk, h, rtol=0, atol=1e-9):
        raise ValueError(f"K grid in {path} is not uniform")
    cv = np.full_like(k, np.nan)
    cv[1:-1] = k[1:-1] ** 2 * (g[2:] - 2 * g[1:-1] + g[:-2]) / (h * h)

    i = int(np.nanargmax(cv))
    k_peak = float(k[i])
    cv_peak = float(cv[i])
    if 0 < i < len(k) - 1:
        coeff = np.polyfit(k[i - 1:i + 2], cv[i - 1:i + 2], 2)
        if coeff[0] < 0:
            vertex = -coeff[1] / (2 * coeff[0])
            if k[i - 1] <= vertex <= k[i + 1]:
                k_peak = float(vertex)
                cv_peak = float(np.polyval(coeff, vertex))
    return {
        "L": int(float(rows[0]["L"])),
        "K": k,
        "T": 1.0 / k,
        "cV": cv,
        "K_peak": k_peak,
        "T_peak": 1.0 / k_peak,
        "cV_peak": cv_peak,
    }


def read_peak_table(path: Path):
    rows = read_rows(path)
    peaks = []
    for row in rows:
        peaks.append({
            "L": int(float(row["L"])),
            "K_peak": float(row["K_peak"]),
            "T_peak": float(row["T_peak"]),
            "cV_peak": float(row["cV_peak"]),
        })
    return sorted(peaks, key=lambda x: x["L"])


def fit_tc(peaks):
    if len(peaks) < 2:
        return None
    x = np.array([1.0 / p["L"] for p in peaks], dtype=float)
    y = np.array([p["T_peak"] for p in peaks], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="runs/trg_scan")
    parser.add_argument("--peak-file", default="coarse_peak_estimates.csv")
    parser.add_argument("--output", default="artifacts/trg-critical.png")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    peak_file = Path(args.peak_file)
    raw = {}
    missing = []
    for L in SIZES:
        path = data_dir / f"coarse_L{L}_chi16.csv"
        if path.exists():
            raw[L] = analyze_raw(path)
        else:
            missing.append(str(path))

    if len(raw) == len(SIZES):
        peaks = [
            {"L": L, "K_peak": d["K_peak"], "T_peak": d["T_peak"], "cV_peak": d["cV_peak"]}
            for L, d in sorted(raw.items())
        ]
        source_label = "raw g(K) tables"
    elif peak_file.exists():
        peaks = read_peak_table(peak_file)
        source_label = "saved peak-only table"
    else:
        raise FileNotFoundError(
            "No complete raw scan set and no peak table found. "
            f"Missing raw files: {missing}; peak table: {peak_file}"
        )

    exact_tc = 2.0 / np.log(1.0 + np.sqrt(2.0))
    all_fit = fit_tc(peaks)
    large_fit = fit_tc(peaks[-2:]) if len(peaks) >= 2 else None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    if raw:
        for L, d in sorted(raw.items()):
            mask = np.isfinite(d["cV"])
            order = np.argsort(d["T"][mask])
            ax.plot(d["T"][mask][order], d["cV"][mask][order], marker=".", label=f"L={L}, chi=16")
        ax.set_xlabel("Temperature T")
        ax.set_ylabel("Specific heat per spin")
        ax.set_title("Periodic TRG specific heat (coarse chi=16)")
        ax.legend()
    else:
        ax.scatter(
            [p["T_peak"] for p in peaks],
            [p["cV_peak"] for p in peaks],
            label="coarse peak values",
        )
        for p in peaks:
            ax.annotate(f"L={p['L']}", (p["T_peak"], p["cV_peak"]), xytext=(5, 5),
                        textcoords="offset points")
        ax.set_xlabel("Peak temperature")
        ax.set_ylabel("Peak specific heat per spin")
        ax.set_title("Peak values only — raw g(K) curves unavailable")
        ax.text(
            0.04, 0.96,
            "Full cV(T) curves require the raw coarse_L*_chi16.csv files.",
            transform=ax.transAxes, va="top",
        )
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    x = np.array([1.0 / p["L"] for p in peaks])
    y = np.array([p["T_peak"] for p in peaks])
    ax.scatter(x, y, label="coarse chi=16 peaks")
    xx = np.linspace(0, max(x) * 1.08, 100)
    if all_fit:
        tc, slope = all_fit
        ax.plot(xx, tc + slope * xx, label=f"all sizes: Tc={tc:.6f}")
    if large_fit:
        tc2, slope2 = large_fit
        ax.plot(xx, tc2 + slope2 * xx, linestyle="--",
                label=f"largest two: Tc={tc2:.6f}")
    ax.axhline(exact_tc, linestyle=":", label=f"Onsager={exact_tc:.6f}")
    ax.set_xlabel("Inverse side length 1/L")
    ax.set_ylabel("Peak temperature")
    ax.set_title("Finite-size extrapolation (coarse only)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.suptitle("Periodic TRG — coarse scan only (chi=16, h≈0.0025; no fine scan)")
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(f"Saved: {output}")
    print(f"Peak source: {source_label}")
    if missing and len(raw) != len(SIZES):
        print("Raw scan files not found:")
        for item in missing:
            print(f"  - {item}")
    if all_fit:
        tc, _ = all_fit
        print(f"All-size coarse Tc = {tc:.10f}")
    if large_fit:
        tc, _ = large_fit
        print(f"Largest-two coarse Tc = {tc:.10f}")
    print("No TRG calculations were launched.")


if __name__ == "__main__":
    main()
