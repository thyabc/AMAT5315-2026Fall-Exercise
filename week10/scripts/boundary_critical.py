import csv
import math
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.cwd()
BIN = ROOT / "target/release/thermo"
RUNS = ROOT / "runs/critical"
ARTIFACTS = ROOT / "artifacts"

L_VALUES = [16, 32, 64]
CHI_VALUES = [16, 32]

K_MIN = 0.40
K_MAX = 0.50
COARSE_H = 0.0025
FINE_H = 0.00125

ONSAGER_TC = 2.0 / math.log(1.0 + math.sqrt(2.0))


def ensure_binary():
    if not BIN.exists():
        print("Building Rust thermo binary...")
        subprocess.run(
            ["cargo", "build", "--release", "--bin", "thermo"],
            cwd=ROOT,
            check=True
        )


def read_summary(path):
    values = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in ("g", "log_Z", "solver_seconds"):
            values[key] = float(value.strip())
    return values


def run_case(L, K, chi, group):
    k_tag = f"{K:.6f}"
    out = RUNS / group / f"L{L}_chi{chi}_K{k_tag}"
    summary = out / "summary.txt"

    if summary.exists():
        data = read_summary(summary)
        if all(k in data for k in ("g", "log_Z", "solver_seconds")):
            return {
                "L": L, "K": K, "chi": chi,
                "g": data["g"],
                "log_Z": data["log_Z"],
                "seconds": data["solver_seconds"],
                "out": str(out)
            }

    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(BIN),
        "--method", "boundary",
        "--l", str(L),
        "--k", f"{K:.12f}",
        "--chi", str(chi),
        "--out", str(out)
    ]

    result = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True
    )

    (out / "run.log").write_text(
        result.stdout + "\n" + result.stderr
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Rust calculation failed: L={L}, K={K}, chi={chi}\n"
            + result.stderr[-3000:]
        )

    if not summary.exists():
        raise RuntimeError(f"Missing summary: {summary}")

    data = read_summary(summary)

    return {
        "L": L, "K": K, "chi": chi,
        "g": data["g"],
        "log_Z": data["log_Z"],
        "seconds": data["solver_seconds"],
        "out": str(out)
    }


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def specific_heat(ks, gs, h):
    # cV(K) = K^2 * [g(K+h)-2g(K)+g(K-h)] / h^2
    result = []
    for i in range(1, len(ks) - 1):
        cv = ks[i] ** 2 * (
            gs[i + 1] - 2.0 * gs[i] + gs[i - 1]
        ) / (h * h)

        result.append((ks[i], cv))

    return result


def parabola_peak(points):
    # Fit a parabola through exactly three equally spaced points.
    if len(points) != 3:
        raise ValueError("Exactly three points are required")

    (x0, y0), (x1, y1), (x2, y2) = points

    h1 = x1 - x0
    h2 = x2 - x1

    if abs(h1 - h2) > 1e-10:
        raise ValueError("Points must be equally spaced")

    h = h1
    denominator = y0 - 2.0 * y1 + y2

    if abs(denominator) < 1e-20:
        return x1, y1

    offset = 0.5 * h * (y0 - y2) / denominator
    x_peak = x1 + offset
    y_peak = y1 - (
        (y0 - y2) ** 2 / (8.0 * denominator)
    )

    return x_peak, y_peak


def fit_peak(curve):
    # Find the maximum and fit it with its two neighbours.
    j = max(range(len(curve)), key=lambda i: curve[i][1])
    j = max(1, min(j, len(curve) - 2))
    return parabola_peak(curve[j - 1:j + 2])


def linear_fit(xs, ys):
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("Cannot fit identical x values")

    slope = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in zip(xs, ys)
    ) / denominator

    intercept = mean_y - slope * mean_x
    return intercept, slope


def exact_energy_variance_cv(L, K):
    # Independent enumeration of all 2^(L*L) spin states.
    n = L * L
    total = 1 << n
    energies = []

    for state in range(total):
        spins = [
            1.0 if ((state >> i) & 1) == 0 else -1.0
            for i in range(n)
        ]

        bond_sum = 0.0

        for y in range(L):
            for x in range(L):
                i = y * L + x

                if x + 1 < L:
                    bond_sum += spins[i] * spins[i + 1]

                if y + 1 < L:
                    bond_sum += spins[i] * spins[i + L]

        energies.append(-bond_sum)

    log_weights = [-K * e for e in energies]
    max_log = max(log_weights)

    weights = [math.exp(x - max_log) for x in log_weights]
    z = sum(weights)

    mean_h = sum(
        w * e for w, e in zip(weights, energies)
    ) / z

    mean_h2 = sum(
        w * e * e for w, e in zip(weights, energies)
    ) / z

    variance = mean_h2 - mean_h * mean_h

    return K * K * variance / n


def main():
    start_all = time.time()
    ensure_binary()

    RUNS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("WEEK 10 - BOUNDARY MPS SPECIFIC HEAT AND CRITICAL TEMPERATURE")
    print("=" * 65)

    # ---------------------------------------------------------
    # Stage 1: Coarse scan, L=16,32,64, chi=16.
    # K = 0.40 ... 0.50, step 0.0025.
    # ---------------------------------------------------------
    print("\n[1/5] Coarse temperature scan")

    coarse_ks = [
        round(K_MIN + i * COARSE_H, 10)
        for i in range(41)
    ]

    coarse = {}
    coarse_rows = []

    for L in L_VALUES:
        coarse[L] = []

        for i, K in enumerate(coarse_ks):
            result = run_case(
                L, K, 16, "coarse"
            )

            coarse[L].append(result)

            coarse_rows.append({
                "L": L,
                "K": K,
                "T": 1.0 / K,
                "chi": 16,
                "g": result["g"],
                "log_Z": result["log_Z"],
                "solver_seconds": result["seconds"],
                "run_directory": result["out"]
            })

            if i % 10 == 0 or i == 40:
                print(
                    f"L={L:2d}: {i+1:2d}/41 "
                    f"K={K:.4f} g={result['g']:.12f}"
                )

    save_csv(
        RUNS / "boundary_g_coarse.csv",
        coarse_rows
    )

    # Find coarse peak centres.
    centres = {}

    for L in L_VALUES:
        ks = [x["K"] for x in coarse[L]]
        gs = [x["g"] for x in coarse[L]]

        curve = specific_heat(ks, gs, COARSE_H)
        peak_k, peak_cv = fit_peak(curve)

        centre_index = max(
            range(len(curve)),
            key=lambda i: curve[i][1]
        )

        centre_k = curve[centre_index][0]
        centres[L] = centre_k

        print(
            f"Coarse L={L}: K_peak~{peak_k:.8f}, "
            f"T_peak~{1.0/peak_k:.8f}, cV~{peak_cv:.8f}"
        )

    # ---------------------------------------------------------
    # Stage 2: Refined 9-point scans, chi=16 and 32.
    # ---------------------------------------------------------
    print("\n[2/5] Refined peak scans")

    refined_rows = []
    peak_rows = []
    refined_curves = {}

    for L in L_VALUES:
        centre = centres[L]

        refined_ks = [
            round(centre + (i - 4) * FINE_H, 10)
            for i in range(9)
        ]

        for chi in CHI_VALUES:
            results = []

            for i, K in enumerate(refined_ks):
                result = run_case(
                    L, K, chi, "refined"
                )
                results.append(result)

                refined_rows.append({
                    "L": L,
                    "chi": chi,
                    "K": K,
                    "T": 1.0 / K,
                    "g": result["g"],
                    "log_Z": result["log_Z"],
                    "solver_seconds": result["seconds"],
                    "run_directory": result["out"]
                })

            ks = [x["K"] for x in results]
            gs = [x["g"] for x in results]

            fine_curve = specific_heat(
                ks, gs, FINE_H
            )

            fine_k, fine_cv = fit_peak(fine_curve)
            refined_curves[(L, chi)] = fine_curve

            peak_rows.append({
                "L": L,
                "chi": chi,
                "step": FINE_H,
                "K_peak": fine_k,
                "T_peak": 1.0 / fine_k,
                "cV_peak": fine_cv,
                "method": "fine_h_0.00125"
            })

            print(
                f"L={L:2d}, chi={chi:2d}, "
                f"fine T_peak={1.0/fine_k:.8f}"
            )

            # At chi=32, use every other sample to repeat
            # the wider h=0.0025 estimate.
            if chi == 32:
                wide_ks = ks[::2]
                wide_gs = gs[::2]

                wide_curve = specific_heat(
                    wide_ks, wide_gs, COARSE_H
                )

                wide_k, wide_cv = fit_peak(wide_curve)

                peak_rows.append({
                    "L": L,
                    "chi": chi,
                    "step": COARSE_H,
                    "K_peak": wide_k,
                    "T_peak": 1.0 / wide_k,
                    "cV_peak": wide_cv,
                    "method": "wide_h_0.0025"
                })

                print(
                    f"L={L:2d}, chi=32, "
                    f"wide T_peak={1.0/wide_k:.8f}"
                )

    save_csv(
        RUNS / "boundary_g_refined.csv",
        refined_rows
    )

    save_csv(
        RUNS / "peak_results.csv",
        peak_rows
    )

    # ---------------------------------------------------------
    # Stage 3: Compare rank and step-size stability.
    # ---------------------------------------------------------
    print("\n[3/5] Rank and step-size comparisons")

    comparison_rows = []

    for L in L_VALUES:
        p16 = next(
            x for x in peak_rows
            if x["L"] == L
            and x["chi"] == 16
            and x["method"] == "fine_h_0.00125"
        )

        p32fine = next(
            x for x in peak_rows
            if x["L"] == L
            and x["chi"] == 32
            and x["method"] == "fine_h_0.00125"
        )

        p32wide = next(
            x for x in peak_rows
            if x["L"] == L
            and x["chi"] == 32
            and x["method"] == "wide_h_0.0025"
        )

        rank_diff = abs(
            p32fine["T_peak"] - p16["T_peak"]
        ) / p32fine["T_peak"] * 100.0

        step_diff = abs(
            p32fine["T_peak"] - p32wide["T_peak"]
        ) / p32fine["T_peak"] * 100.0

        comparison_rows.append({
            "L": L,
            "Tpeak_chi16_fine": p16["T_peak"],
            "Tpeak_chi32_fine": p32fine["T_peak"],
            "Tpeak_chi32_wide": p32wide["T_peak"],
            "rank_change_percent": rank_diff,
            "step_change_percent": step_diff
        })

        print(
            f"L={L}: rank shift={rank_diff:.4f}%, "
            f"step shift={step_diff:.4f}%"
        )

    save_csv(
        RUNS / "peak_stability.csv",
        comparison_rows
    )

    # ---------------------------------------------------------
    # Stage 4: Critical-temperature extrapolation at chi=32.
    # ---------------------------------------------------------
    print("\n[4/5] Critical-temperature extrapolation")

    refined32 = sorted(
        [
            x for x in peak_rows
            if x["chi"] == 32
            and x["method"] == "fine_h_0.00125"
        ],
        key=lambda x: x["L"]
    )

    all_x = [1.0 / x["L"] for x in refined32]
    all_y = [x["T_peak"] for x in refined32]

    tc_all, slope_all = linear_fit(all_x, all_y)

    largest = refined32[-2:]
    large_x = [1.0 / x["L"] for x in largest]
    large_y = [x["T_peak"] for x in largest]

    tc_large, slope_large = linear_fit(
        large_x, large_y
    )

    onsager = ONSAGER_TC

    print(f"Onsager Tc       = {onsager:.9f}")
    print(f"Tc, all sizes    = {tc_all:.9f}")
    print(f"Tc, largest two  = {tc_large:.9f}")

    err_all = abs(tc_all - onsager) / onsager * 100.0
    err_large = abs(tc_large - onsager) / onsager * 100.0

    print(f"All-size error   = {err_all:.5f}%")
    print(f"Two-size error   = {err_large:.5f}%")

    fit_rows = [
        {
            "fit": "all_sizes",
            "Tc": tc_all,
            "slope_a": slope_all,
            "Onsager_Tc": onsager,
            "relative_error_percent": err_all
        },
        {
            "fit": "largest_two",
            "Tc": tc_large,
            "slope_a": slope_large,
            "Onsager_Tc": onsager,
            "relative_error_percent": err_large
        }
    ]

    save_csv(
        RUNS / "critical_temperature_fit.csv",
        fit_rows
    )

    # ---------------------------------------------------------
    # Stage 5: 4x4 specific heat versus exact energy variance.
    # ---------------------------------------------------------
    print("\n[5/5] Independent 4x4 energy-variance check")

    K0 = 0.44
    h = 0.0025

    small_results = {}

    for K in [K0 - h, K0, K0 + h]:
        small_results[K] = run_case(
            4, K, 8, "small_check"
        )

    g_minus = small_results[K0-h]["g"]
    g_zero = small_results[K0]["g"]
    g_plus = small_results[K0+h]["g"]

    cv_mps = K0**2 * (
        g_plus - 2.0*g_zero + g_minus
    ) / h**2

    cv_exact = exact_energy_variance_cv(4, K0)

    cv_error = abs(cv_mps - cv_exact)

    print(f"MPS finite-difference cV = {cv_mps:.10f}")
    print(f"Exact variance cV        = {cv_exact:.10f}")
    print(f"Absolute difference      = {cv_error:.6e}")

    check_text = (
        f"L = 4\nK = {K0}\nh = {h}\nchi = 8\n"
        f"cV_finite_difference = {cv_mps:.15e}\n"
        f"cV_exact_variance = {cv_exact:.15e}\n"
        f"absolute_error = {cv_error:.15e}\n"
        f"pass = {cv_error < 1e-3}\n"
    )

    (RUNS / "specific_heat_check.txt").write_text(
        check_text
    )

    # ---------------------------------------------------------
    # Plot: specific heat curves and peak extrapolation.
    # ---------------------------------------------------------
    print("\nGenerating boundary-critical.png...")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]

    for L in L_VALUES:
        ks = [x["K"] for x in coarse[L]]
        gs = [x["g"] for x in coarse[L]]

        curve = specific_heat(ks, gs, COARSE_H)

        ts = [1.0 / x[0] for x in curve]
        cvs = [x[1] for x in curve]

        order = sorted(range(len(ts)), key=lambda i: ts[i])

        ax.plot(
            [ts[i] for i in order],
            [cvs[i] for i in order],
            label=f"L={L}, chi=16"
        )

    for L in L_VALUES:
        curve = refined_curves[(L, 32)]

        ts = [1.0 / x[0] for x in curve]
        cvs = [x[1] for x in curve]

        order = sorted(range(len(ts)), key=lambda i: ts[i])

        ax.plot(
            [ts[i] for i in order],
            [cvs[i] for i in order],
            marker="o",
            markersize=3,
            linestyle="--",
            label=f"L={L}, chi=32 refined"
        )

    ax.axvline(
        onsager,
        linestyle=":",
        label="Onsager Tc"
    )

    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Specific heat per spin")
    ax.set_title("Open-square specific heat")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    # Extrapolation plot.
    ax = axes[1]

    ax.scatter(
        all_x, all_y,
        label="chi=32 peak temperatures",
        zorder=3
    )

    x_line = [
        0.0,
        max(all_x) * 1.08
    ]

    ax.plot(
        x_line,
        [tc_all + slope_all*x for x in x_line],
        linestyle="--",
        label=f"All sizes: Tc={tc_all:.6f}"
    )

    ax.plot(
        x_line,
        [tc_large + slope_large*x for x in x_line],
        linestyle="-.",
        label=f"L=32,64: Tc={tc_large:.6f}"
    )

    ax.axhline(
        onsager,
        linestyle=":",
        label=f"Onsager={onsager:.6f}"
    )

    ax.set_xlabel("Inverse side length 1/L")
    ax.set_ylabel("Peak temperature")
    ax.set_title("Critical-temperature extrapolation")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle(
        "Week 10 - Boundary MPS Ising Critical Behaviour"
    )

    fig.tight_layout()

    fig.savefig(
        ARTIFACTS / "boundary-critical.png",
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(fig)

    total_time = time.time() - start_all

    report = f"""
WEEK 10 BOUNDARY MPS CRITICAL ANALYSIS

Model: zero-field square-lattice Ising
Boundary: open
Coarse scan: K=0.40 to 0.50, step=0.0025
Refined scan: 9 points, step=0.00125
Ranks: chi=16 and chi=32
Sizes: L=16,32,64

Onsager Tc: {onsager:.12f}

Tc from all three sizes: {tc_all:.12f}
Tc from largest two: {tc_large:.12f}

All-size relative error: {err_all:.8f}%
Largest-two relative error: {err_large:.8f}%

4x4 finite-difference cV: {cv_mps:.12f}
4x4 exact variance cV: {cv_exact:.12f}
4x4 cV absolute error: {cv_error:.12e}

Total script time: {total_time:.3f} seconds

4x4 cV check passed: {cv_error < 1e-3}
"""

    (RUNS / "final_report.txt").write_text(report)

    print(report)
    print("Figure:", ARTIFACTS / "boundary-critical.png")
    print("Evidence:", RUNS)
    print("All calculations finished.")


if __name__ == "__main__":
    main()
