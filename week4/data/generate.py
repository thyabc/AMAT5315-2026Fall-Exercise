#!/usr/bin/env python3
"""Regenerate the computed data behind week4/learning-sheet.typ figures.

2D incompressible Navier-Stokes in vorticity-streamfunction form, Fourier
pseudospectral on [0, 2pi)^2 (week4/DESIGN.md conventions). The solver is the
NumPy baseline `fixtures/week4-reference/sim.py`, imported, not re-implemented.

Outputs (small JSON + PNGs, committed so the sheet compiles from a fresh clone):
  omega-t{0,2,5,10}.png  random case (N=128, nu=4e-3, seed 2026) vorticity,
  omega-colorbar.png     RdBu_r symmetric at +-max|omega(0)|, no axes  (Part 2)
  snapshots.json         E, Z and energy-weighted mean |k| at those four times
  tg-decay.json          Taylor-Green N=64 nu=0.1: E(t) numerical vs exact
                         1/4 e^(-4 nu t), Z(t), and the forward-Euler E(t)  (Part 1)
  budget.json            random case: E(t) vs E(0) - 2 nu int Z dt, for RK4,
                         forward Euler (dt=0.001) and no advection      (Part 2)
  transfer.json          shell-averaged |omega_k| at t = 0 and t = 10 beside the
                         pure-diffusion prediction, plus the transfer values (Part 2)
  convergence.json       N and dt refinement errors and the fitted slope (Part 3)
  stability-region.json  |R(z)| = 1 boundaries of forward Euler, midpoint and RK4
                         as polylines in the lambda*h plane, their axis crossings,
                         and where the contract runs' eigenvalues sit (Part 2)
  blowup.json            E(t) of the bracketing (stable, unstable) RK4 runs of both
                         cases, reshaped from stability.json for the sheet (Part 2)
  concepts-{main,vortex,stream}.png, concepts.json
                         text-free panels of the random field at t = 5 (the box with
                         its periodic margin, streamlines, arrows; two zoom tiles) and
                         the anchor points the sheet's CeTZ annotations use
  line-stability.png     Part 1: RK4's measured growth factor over the complex plane
                         with the three |R| = 1 curves and this toy's modes, beside
                         space-time pictures of the pulse just below and just above
                         the predicted step limit                       (Part 1)
  line-accuracy.png      Part 1: the pulse race after one lap, and the order fan of
                         the four steppers with their fitted slopes     (Part 1)
  line-spiral.png        Part 1 UNDERSTAND: one Fourier mode's complex amplitude under
                         the three steppers against the exact spiral    (Part 1)
  line-figures.json      the parameters and the measured values those three carry
  line-setup.json        Part 1 UNDERSTAND: the exact pulse at three times, drawn in CeTZ

convergence.json is copied from week4/data/measurements.json (`sim.py --measure`,
which owns the expensive N = 256 reference runs). stability-region.json and
blowup.json take the answer key's measurements from stability.json (Rust key,
week4/experiments/stability/README.md) as they are: nothing there is recomputed
in NumPy, only the stability functions are evaluated. Everything else is
recomputed here, ~2 min total.

Run:  python3 week4/data/generate.py
      python3 week4/data/generate.py --only budget --only transfer
      python3 week4/data/generate.py --only stability --only blowup   (seconds)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "fixtures",
                                                 "week4-reference")))
import line  # noqa: E402  (the Part 1 toy, beside sim.py)
import sim  # noqa: E402  (needs the path above)

KEEP = (0.0, 2.0, 5.0, 10.0)
C_LINE = 1.0       # the advection speed of the Part 1 toy
NBIN = 42          # dealiased k_max at N = 128
EULER_DT = 0.001   # the plausible wrong stepper, at its smallest stable step
_t0 = time.time()


def r(x, p=6):
    """Plain float rounded to p significant digits (no numpy types in the JSON)."""
    return float(f"{float(x):.{p}g}")


def lap(x, p=6):
    return [r(v, p) for v in np.asarray(x).ravel()]


def write(name, obj):
    path = os.path.join(HERE, name)
    with open(path, "w") as fh:
        json.dump(obj, fh)
    print(f"  {name}  {os.path.getsize(path) / 1024:.1f} KB "
          f"[{time.time() - _t0:.0f} s]", flush=True)


def load(name):
    with open(os.path.join(HERE, name)) as fh:
        return json.load(fh)


def mean_k(w, g):
    """Energy-weighted mean |k|: E_k ~ |omega_hat_k|^2 / k^2."""
    e = np.abs(np.fft.fft2(w)) ** 2 * g["inv"]
    kk = np.hypot(g["kx"], g["ky"])
    return float((kk * e).sum() / e.sum())


def cumtrap(y, x):
    return np.concatenate([[0.0], np.cumsum(np.diff(x) * (y[1:] + y[:-1]) / 2)])


# ------------------------------------------------------------------- figures
def figure_snapshots(snaps, t, e, z, g):
    """Fig 1: bare RdBu_r vorticity images plus one colorbar strip."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize

    vmax = float(np.abs(snaps[0.0]).max())
    for tv in KEEP:
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        ax.imshow(snaps[tv], origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_axis_off()
        fig.savefig(os.path.join(HERE, "omega-t%d.png" % tv), dpi=150,
                    bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.2, 0.38))
    cb = ColorbarBase(ax, cmap=plt.get_cmap("RdBu_r"),
                      norm=Normalize(-vmax, vmax), orientation="horizontal")
    cb.set_ticks([-vmax, 0.0, vmax])
    cb.set_ticklabels(["%.0f" % -vmax, "0", "%.0f" % vmax])
    cb.ax.tick_params(labelsize=7, length=2)
    fig.savefig(os.path.join(HERE, "omega-colorbar.png"), dpi=150,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    i = [int(round(tv / sim.RND["snapshot_every"])) for tv in KEEP]
    write("snapshots.json", {
        "t": [0, 2, 5, 10],
        "E": [r(e[j], 5) for j in i],
        "Z": [r(z[j], 5) for j in i],
        "mean_k": [r(mean_k(snaps[tv], g), 4) for tv in KEEP],
        "vmax": r(float(np.abs(snaps[0.0]).max()), 4),
        "n": sim.RND["n"], "nu": sim.RND["nu"], "seed": sim.SEED,
    })


def figure_concepts():
    """Opening figure: text-free panels of the random run at t = 5, plus concepts.json
    with the data-true anchor points; the sheet adds every label and equation in CeTZ."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cfg = dict(sim.RND, t_end=5.0)
    g = sim.grid(cfg["n"])
    _, _, _, snaps, w_hat, _ = sim.frames_of(cfg, keep=(5.0,))
    w = snaps[5.0]
    n = cfg["n"]
    psi = np.fft.ifft2(w_hat * g["inv"]).real
    u, v = sim.velocity(w_hat, g)
    L = 2 * np.pi
    x = np.arange(n) * L / n
    vmax = float(np.abs(w).max())
    ink = "#1c1c1c"
    pad = 0.22                       # periodic margin shown faded, as a fraction of L

    def bare(size):
        fig = plt.figure(figsize=size)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        return fig, ax

    # main panel: the box with its periodic continuation, streamlines, velocity arrows
    fig, ax = bare((4.6, 4.6))
    ax.imshow(np.tile(w, (3, 3)), origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
              extent=(-L, 2 * L, -L, 2 * L), interpolation="bilinear", alpha=0.28)
    ax.imshow(w, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
              extent=(0, L, 0, L), interpolation="bilinear")
    ax.contour(x, x, psi, levels=11, colors="white", linewidths=0.55, alpha=0.9)
    st = 8
    sl = slice(st // 2, None, st)
    ax.quiver(x[sl], x[sl], u[sl, sl], v[sl, sl], color=ink, scale=34, width=0.0024,
              headwidth=3.6, headlength=4.5, alpha=0.85)
    ax.add_patch(Rectangle((0, 0), L, L, fill=False, lw=1.1, ec=ink))
    ax.set_xlim(-pad * L, (1 + pad) * L); ax.set_ylim(-pad * L, (1 + pad) * L)
    fig.savefig(os.path.join(HERE, "concepts-main.png"), dpi=220)
    plt.close(fig)

    # tiles: zoom windows of the same field
    def window(cx, cy, half):
        cx = min(max(cx, half), L - half); cy = min(max(cy, half), L - half)
        i0, i1 = int((cx - half) / L * n), int((cx + half) / L * n)
        j0, j1 = int((cy - half) / L * n), int((cy + half) / L * n)
        return slice(j0, j1), slice(i0, i1)

    def tile(name, wy, wx, lines, step=4):
        fig, t = bare((1.6, 1.6))
        t.imshow(w[wy, wx], origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                 extent=(x[wx][0], x[wx][-1], x[wy][0], x[wy][-1]), interpolation="bilinear")
        if lines:
            t.contour(x[wx], x[wy], psi[wy, wx], levels=7, colors="white", linewidths=0.5)
        s2 = slice(step // 2, None, step)
        t.quiver(x[wx][s2], x[wy][s2], u[wy, wx][s2, s2], v[wy, wx][s2, s2], color=ink,
                 scale=16, width=0.006, headwidth=3.2, headlength=4)
        t.set_xlim(x[wx][0], x[wx][-1]); t.set_ylim(x[wy][0], x[wy][-1])
        fig.savefig(os.path.join(HERE, name), dpi=220)
        plt.close(fig)

    half = 0.55
    jp, ip = np.unravel_index(np.argmax(w), w.shape)          # strongest positive core
    jn, in_ = np.unravel_index(np.argmin(w), w.shape)         # strongest negative core
    tile("concepts-vortex.png", *window(x[ip], x[jp], half), lines=False)
    speed = np.hypot(u, v)
    jq, iq = np.unravel_index(np.argmax(speed * (np.abs(w) < 0.25 * vmax)), w.shape)
    tile("concepts-stream.png", *window(x[iq], x[jq], half), lines=True)

    # anchor points as fractions of the box side, for the sheet's callouts
    iy = int(np.argmax(np.abs(w[:, 0])))                      # a strong feature on the left edge
    write("concepts.json", {
        "t": 5.0, "n": n, "nu": cfg["nu"], "seed": sim.SEED, "pad": pad,
        "core_positive": [r(x[ip] / L, 4), r(x[jp] / L, 4)],
        "core_negative": [r(x[in_] / L, 4), r(x[jn] / L, 4)],
        "streamline": [r(x[iq] / L, 4), r(x[jq] / L, 4)],
        "edge_row": r(x[iy] / L, 4),
        "tile_half": r(half / L, 4),
    })
    print(f"  concepts-*.png [{time.time() - _t0:.0f} s]", flush=True)


def figure_tg():
    """Part 1: Taylor-Green exact decay, RK4 against forward Euler."""
    cfg = sim.TG
    t, e, z, _, _, _ = sim.frames_of(cfg)
    te, ee = sim.frames_of(dict(cfg, dt=EULER_DT), scheme="euler")[:2]
    exact = 0.25 * np.exp(-4 * cfg["nu"] * t)
    write("tg-decay.json", {
        "t": lap(t, 4), "E": lap(e, 8), "E_exact": lap(exact, 8), "Z": lap(z, 8),
        "E_euler": lap(ee, 8), "euler_dt": EULER_DT,
        "err_rk4": r(abs(e[-1] - exact[-1]) / exact[-1], 3),
        "err_euler": r(abs(ee[-1] - exact[-1]) / exact[-1], 3),
        "n": cfg["n"], "nu": cfg["nu"], "dt": cfg["dt"],
    })
    print(f"    TG t=1: E = {e[-1]:.6f}, exact = {exact[-1]:.6f}, "
          f"Euler = {ee[-1]:.6f}", flush=True)
    return e[-1], exact[-1], ee[-1]


def figure_budget(t, e, z, series):
    """Fig 2: E(t) against the budget prediction E(0) - 2 nu int Z dt."""
    nu = sim.RND["nu"]
    out = {"t": lap(t, 4), "nu": nu, "euler_dt": EULER_DT, "residual": {}}
    for name, (ee, zz) in dict(rk4=(e, z), **series).items():
        out["E_" + name] = lap(ee, 8)
        out["Z_" + name] = lap(zz, 6)
        out["pred_" + name] = lap(ee[0] - 2 * nu * cumtrap(zz, t), 8)
        out["residual"][name] = r(sim.budget(t, ee, zz, nu), 3)
    write("budget.json", out)
    print("    budget residuals:", out["residual"], flush=True)
    return out["residual"]


def figure_transfer(w0_hat, wend_hat, wna_hat, g):
    """Fig 3: shell-averaged |omega_k| vs the pure-diffusion prediction."""
    n, nu, tend = sim.RND["n"], sim.RND["nu"], sim.RND["t_end"]
    shell = np.rint(np.hypot(g["kx"], g["ky"])).astype(int)
    decay = np.exp(-nu * g["k2"] * tend)
    h0, hT = w0_hat / (n * n), wend_hat / (n * n)

    def avg(a):
        return [r(float(a[shell == k].mean()), 4) for k in range(1, NBIN + 1)]

    def written(wh):
        """omega_hat as the checker gets it: recomputed from the 6-decimal u, v."""
        u, v = sim.velocity(wh, g)
        return (1j * g["kx"] * np.fft.fft2(np.round(v, 6))
                - 1j * g["ky"] * np.fft.fft2(np.round(u, 6)))

    a0 = written(w0_hat)

    def tr(wh):
        return float(np.linalg.norm(written(wh) - a0 * decay) / np.linalg.norm(a0))

    out = {"k": list(range(1, NBIN + 1)),
           "s0": avg(np.abs(h0)), "s10": avg(np.abs(hT)),
           "s_diff": avg(np.abs(h0) * decay),
           "transfer_real": r(tr(wend_hat), 3), "transfer_noadvect": r(tr(wna_hat), 3),
           "nu": nu, "t_end": tend, "n": n}
    write("transfer.json", out)
    print(f"    transfer: real {out['transfer_real']}, "
          f"no advection {out['transfer_noadvect']}", flush=True)
    return out["transfer_real"], out["transfer_noadvect"]


def figure_convergence():
    """Fig 4: copied from measurements.json (sim.py --measure owns the N=256 runs)."""
    with open(os.path.join(HERE, "measurements.json")) as fh:
        c = json.load(fh)["convergence"]
    out = {
        "t_end": 2.0, "reference": c["reference"], "n_dt": 0.01,
        "N": c["n_values"],
        "err_N": lap(c["n_errors"], 3),
        "err_N_no_dealias": lap(c["n_errors_no_dealias"], 3),
        "ratio_N": lap(c["n_ratios"], 3),
        "dt": c["dt_values_same_n"],
        "err_dt": lap(c["dt_errors_same_n"], 3),
        "dt_slope": r(c["dt_slope_same_n"], 4),
        "dt_reference": {"n": 128, "dt": 0.0025},
    }
    write("convergence.json", out)
    print(f"    N errors {out['err_N']}, dt errors {out['err_dt']}, "
          f"slope {out['dt_slope']}", flush=True)
    return out


# stability functions R(z) of the three explicit steppers, z = lambda h
STABILITY = {
    "euler": lambda z: 1 + z,
    "rk2": lambda z: 1 + z + z ** 2 / 2,
    "rk4": lambda z: 1 + z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24,
}


def figure_stability_region():
    """Part 2: |R(z)| = 1 boundaries as polylines (matplotlib's contour tracer on a
    fine grid), the RK4 axis crossings, and the eigenvalues of this week's runs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.linspace(-3.9, 0.8, 1201)
    y = np.linspace(-3.6, 3.6, 1201)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    boundaries = {}
    fig, ax = plt.subplots()
    for name, R in STABILITY.items():
        cs = ax.contour(X, Y, np.abs(R(Z)), levels=[1.0])
        segs = [np.asarray(p.vertices) for p in cs.get_paths()] if hasattr(cs, "get_paths") \
            else [np.asarray(v) for v in cs.allsegs[0]]
        seg = max(segs, key=len)             # the region's outer boundary
        step = max(1, len(seg) // 360)
        seg = np.vstack([seg[::step], seg[-1:]])
        boundaries[name] = [[r(a, 4), r(b, 4)] for a, b in seg]
    plt.close(fig)

    def real_crossing(R):                    # leftmost x < 0 with |R(x)| = 1
        xs = np.linspace(-4, -0.5, 70001)
        f = np.abs(R(xs + 0j)) - 1
        i = np.nonzero((f[:-1] > 0) & (f[1:] <= 0))[0][0]
        lo, hi = xs[i], xs[i + 1]
        for _ in range(60):
            mid = (lo + hi) / 2
            if np.abs(R(mid + 0j)) - 1 > 0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    crossings = {name: {"real": r(real_crossing(R), 5)} for name, R in STABILITY.items()}
    crossings["rk4"]["imag"] = r(2 * np.sqrt(2), 5)   # |R4(iy)|^2 = 1 - y^6/72 + y^8/576
    crossings["leapfrog"] = {"imag": 2.0}             # velocity-Verlet on an oscillator

    # where the solver's eigenvalues land, from the answer key's measurements
    st = load("stability.json")
    dl, al = st["diffusive_limit"], st["advective_limit"]
    dt = sim.TG["dt"]
    assert dt == sim.RND["dt"] == 0.01
    k2_rnd = 2 * al["k_cut"] ** 2
    out = {
        "boundaries": boundaries,
        "crossings": crossings,
        "contract_dt": dt,
        "eigen": {
            "diffusion_tg": {"n": dl["n"], "nu": dl["nu"], "k2_max": dl["k2_max"],
                             "z": r(-dl["nu"] * dl["k2_max"] * dt, 4)},
            "diffusion_random": {"n": al["n"], "nu": al["nu"], "k2_max": k2_rnd,
                                 "z": r(-al["nu"] * k2_rnd * dt, 4)},
            "advection_random": {"u_max": r(al["U_max_initial"], 4), "k_max": r(al["k_max"], 4),
                                 "z_imag": r(al["U_max_initial"] * al["k_max"] * dt, 4)},
        },
        "unstable_runs": {
            "tg": {"dt": sim.UNSTABLE["dt"],
                   "z": r(-dl["nu"] * dl["k2_max"] * sim.UNSTABLE["dt"], 4)},
            "random": {"dt": al["smallest_dt_non_finite"],
                       "z_imag": r(al["U_max_initial"] * al["k_max"]
                                   * al["smallest_dt_non_finite"], 4)},
        },
        "predicted_dt": {"tg_rk4": r(dl["predicted_dt"]["rk4"], 4),
                         "tg_euler": r(dl["predicted_dt"]["euler"], 4),
                         "random_rk4": r(al["predicted_dt"], 4)},
        "source": "boundaries: |R(z)| = 1 traced here; every other number from "
                  "stability.json (Rust answer key) and sim.py's contract configurations",
    }
    write("stability-region.json", out)
    print(f"    crossings {crossings}; eigen {out['eigen']}; unstable {out['unstable_runs']}",
          flush=True)
    return out


def figure_blowup():
    """Part 2: E(t) of the bracketing RK4 runs, reshaped from stability.json (the Rust
    answer key's stdout, sampled every 0.1) into strict JSON with the non-finite
    tail dropped; the stop time is the committed every-0.5 measurement."""
    st = load("stability.json")
    rk4 = st["diffusive_limit"]["per_method"]["rk4"]
    al = st["advective_limit"]

    def finite(series):
        return [[r(t, 5), r(e, 6)] for t, e, _ in series if np.isfinite(e)]

    tg_fine, rnd_fine = rk4["bracketing_series_every_0.1"], al["bracketing_series_every_0.1"]
    tg_lo, tg_hi = rk4["largest_dt_finite_no_growth"], rk4["smallest_dt_non_finite"]
    rn_lo, rn_hi = al["largest_dt_finite"], al["smallest_dt_non_finite"]
    out = {
        "taylor_green": {
            "n": st["diffusive_limit"]["n"], "nu": st["diffusive_limit"]["nu"],
            "predicted_dt": r(rk4["predicted_dt"], 4),
            "dt_stable": tg_lo, "dt_unstable": tg_hi,
            "stopped_at_t": rk4["stopped_at_t"],
            "E_stable": finite(tg_fine["dt%.3f" % tg_lo]),
            "E_unstable": finite(tg_fine["dt%.3f" % tg_hi]),
            "exact": "0.25 exp(-4 nu t)",
        },
        "random": {
            "n": al["n"], "nu": al["nu"], "seed": al["seed"],
            "predicted_dt": r(al["predicted_dt"], 4),
            "dt_stable": rn_lo, "dt_unstable": rn_hi,
            "stopped_at_t": al["stopped_at_t"],
            "E_stable": finite(rnd_fine["dt%.3f" % rn_lo]),
            "E_unstable": finite(rnd_fine["dt%.3f" % rn_hi]),
            "euler_dt": al["euler_contract_step_series"]["dt"],
            "euler_stopped_at_t": al["euler_contract_step_series"]["stopped_at_t"],
            "E_euler": finite(al["euler_contract_step_series"]["series"]),
        },
        "perturbation": {
            "eps": 1e-5, "dt": st["perturbation_divergence"]["dt"],
            "t_end": st["perturbation_divergence"]["t_end"],
            "random": [[r(t, 5), r(v, 4)] for t, v in st["perturbation_divergence"]["random"]["series"]],
            "taylor_green": [[r(t, 5), r(v, 4)] for t, v
                             in st["perturbation_divergence"]["taylor_green"]["series"]],
            "rounding_floor": 1e-6,
        },
        "source": "stability.json: bracketing_series_every_0.1, euler_contract_step_series and "
                  "perturbation_divergence "
                  "(Rust answer key stdout); stopped_at_t is the every-0.5 run's value, the one "
                  "the sheet quotes; euler_stopped_at_t is the every-0.1 measurement",
    }
    write("blowup.json", out)
    for k in ("taylor_green", "random"):
        c = out[k]
        print(f"    {k}: dt {c['dt_stable']} / {c['dt_unstable']}, "
              f"last finite E of the unstable run {c['E_unstable'][-1]}, "
              f"stopped at t = {c['stopped_at_t']}", flush=True)
    pr = out["perturbation"]
    print(f"    perturbation: random {pr['random'][0][1]} -> {pr['random'][20][1]} (t=10) -> "
          f"{pr['random'][-1][1]} (t=20); taylor-green {pr['taylor_green'][0][1]} -> "
          f"{pr['taylor_green'][4][1]} (t=2)", flush=True)
    c = out["random"]
    print(f"    random, forward Euler at dt {c['euler_dt']}: last finite E {c['E_euler'][-1]}, "
          f"non-finite at t = {c['euler_stopped_at_t']}", flush=True)
    return out


# --------------------------------------------------------------- Part 1: the line
# The toy of Part 1 (week4/DESIGN.md, decision of 2026-09-20): periodic
# advection-diffusion on [0, 2pi), N = 64, c = 1, four explicit steppers.  The
# pictures are drawn here from the NumPy baseline `line.py`; the numbers the sheet
# quotes are the Rust answer key's, in `line.json`.  The two agree (see the report
# in week4/DESIGN.md), except where a round-off-seeded mode decides the answer.
LINE_NU = 0.05                       # the viscosity of the stability panels
# The two steps the space-time panels and the dots show: 0.045 is below the toy's
# exact limit 0.04939 (every mode inside the curve), 0.056 is above it.  They are not
# fractions of the axis estimate 2.785/(nu k_max^2) = 0.0544, which is above the true
# limit: at 0.0517 the outermost dots already sit outside while the picture still
# looks clean, which is what the panel is supposed to rule out.
LINE_PANELS = (0.045, 0.056)


def _plain(ax):
    """No title: the sheet's caption carries the words, as for every other figure."""
    ax.tick_params(labelsize=8)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)


def _tag(ax, text, loc="upper left"):
    """A short parameter label inside the axes, so a panel names its own run."""
    at = dict(x=0.03, y=0.96, va="top", ha="left") if loc == "upper left" else \
        dict(x=0.97, y=0.04, va="bottom", ha="right")
    ax.text(at.pop("x"), at.pop("y"), text, transform=ax.transAxes, fontsize=8.5,
            bbox=dict(fc="white", ec="0.7", lw=0.5, pad=2.2, alpha=0.92), **at)


def figure_line_stability():
    """Part 1, left: RK4's growth factor per step measured on y' = z y over a grid of
    complex z, with the analytic |R| = 1 curves and this toy's modes on top.  Middle
    and right: the pulse in space and time just below and just above the predicted
    step limit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    nu = LINE_NU
    p = line.predicted_limits("rk4", nu)
    lam = line.eigenvalues(nu)
    hs = list(LINE_PANELS)

    # measured growth per step: 40 steps of RK4 on y' = z y at h = 1
    re = np.linspace(-4.2, 1.2, 271)
    im = np.linspace(-3.6, 3.6, 361)
    Z = re[None, :] + 1j * im[:, None]
    y, nsteps = np.ones_like(Z), 40
    for _ in range(nsteps):
        y = line.rk4(lambda v: Z * v, y, 1.0)
    growth = np.abs(y) ** (1 / nsteps)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3),
                             gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    ax = axes[0]
    mesh = ax.pcolormesh(re, im, growth, norm=LogNorm(vmin=0.3, vmax=3),
                         cmap="RdBu_r", shading="auto", rasterized=True)
    ax.contour(re, im, np.abs(line.R["rk4"](Z)), levels=[1.0], colors="k",
               linewidths=1.0)
    for name, col in (("euler", "0.55"), ("rk2", "0.35")):
        ax.contour(re, im, np.abs(line.R[name](Z)), levels=[1.0], colors=col,
                   linewidths=0.7, linestyles="--")
    for h, col in zip(hs, ("tab:green", "tab:red")):
        z = lam * h
        ax.plot(z.real, z.imag, "o", ms=3.5, color=col, mec="k", mew=0.3,
                label=r"$h$ = %.4f" % h)
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel(r"Re $\lambda h$")
    ax.set_ylabel(r"Im $\lambda h$")
    ax.legend(loc="upper left", fontsize=8, title=r"modes $\lambda_k h$",
              title_fontsize=8)
    ax.set_aspect("equal")
    _plain(ax)
    cb = fig.colorbar(mesh, ax=ax, shrink=0.85)
    cb.set_label("growth factor per step, RK4", size=9)
    cb.ax.tick_params(labelsize=8)

    F = line.make_F(nu)
    u0 = line.gaussian(line.SIGMA, line.X0)
    out = {"nu": nu, "predicted": r(p["predicted"], 5),
           "h_diffusive": r(p["diffusive"], 5), "h_advective": r(p["advective"], 5),
           "h_region": r(p["region"], 5), "h_panels": list(LINE_PANELS),
           "outermost_mode": [line.outermost("rk4", nu, h) for h in LINE_PANELS],
           "panels": []}
    for ax, h in zip(axes[1:], hs):
        t, hist = line.run(line.rk4, F, u0, h, line.T_END)
        ax.pcolormesh(line.x, t, np.clip(hist, -1.0, 1.0), cmap="RdBu_r",
                      vmin=-1.0, vmax=1.0, shading="auto", rasterized=True)
        ax.invert_yaxis()
        ax.set_xlabel("$x$")
        ax.set_ylabel("$t$")
        _plain(ax)
        b = line.blowup("rk4", nu, h)
        _tag(ax, r"$h$ = %g" % h)
        out["panels"].append({"h": r(h, 5),
                              "over_exact_limit": r(h / p["region"], 4), **b})
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "line-stability.png"), dpi=150)
    plt.close(fig)
    print(f"    line-stability: predicted {out['predicted']}, panels "
          f"{[(c['h'], c['t_exceeds_2'], c['dominant_k']) for c in out['panels']]} "
          f"[{time.time() - _t0:.0f} s]", flush=True)
    return out


def figure_line_accuracy():
    """Part 1, left: the pulse after one lap against its closed-form ghost, with
    spectral and centred-difference RK4 and with forward Euler.  Right: the error at
    a fixed time against the step, for the four steppers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg, finals, ref, u0 = line.race()
    o = line.order()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2),
                             gridspec_kw=dict(width_ratios=[1.4, 1]))
    ax = axes[0]
    ax.plot(line.x, ref, "k-", lw=3, alpha=0.25, label="exact, after one lap")
    ax.plot(line.x, finals["fourier_rk4"], "-", color="tab:blue", lw=1.4,
            label=r"Fourier derivative, RK4, $h$ = %g" % cfg["h_rk4"])
    ax.plot(line.x, finals["fd_rk4"], "-", color="tab:orange", lw=1.4,
            label=r"centred differences, RK4, $h$ = %g" % cfg["h_rk4"])
    ax.plot(line.x, finals["fourier_euler"], "-", color="tab:red", lw=0.9,
            label=r"Fourier derivative, forward Euler, $h$ = %g" % cfg["h_euler"])
    ax.plot(line.x, u0, ":", color="0.5", lw=1, label="start")
    ax.set_ylim(-0.6, 1.3)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u$")
    ax.legend(fontsize=8, loc="upper left")
    _tag(ax, r"$N$ = %d, $\nu$ = %g, $\sigma$ = %g" % (line.N, cfg["nu"], cfg["sigma"]),
         loc="lower right")
    _plain(ax)

    ax = axes[1]
    style = {"euler": ("tab:red", "o"), "rk2": ("tab:orange", "s"),
             "rk4": ("tab:blue", "^"), "rk4-broken": ("0.45", "x")}
    label = {"euler": "forward Euler", "rk2": "midpoint", "rk4": "RK4",
             "rk4-broken": "RK4, weights 1,1,1,1/4"}
    for m, (col, mark) in style.items():
        ax.loglog(o["h"], o["error"][m], "-", marker=mark, ms=4.5, lw=1.1, color=col,
                  label="%s, slope %.2f" % (label[m], o["slope"][m]))
    ax.set_xticks(o["h"])
    ax.set_xticklabels(["%g" % h for h in o["h"]])
    ax.minorticks_off()
    ax.set_xlabel("$h$")
    ax.set_ylabel(r"max $|u(T) - u_{\rm exact}(T)|$")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    _tag(ax, r"$T$ = %g, $\nu$ = %g" % (o["t_end"], o["nu"]))
    _plain(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "line-accuracy.png"), dpi=150)
    plt.close(fig)
    out = {"race": {**{k: v for k, v in cfg.items() if k != "max_error"},
                    "t_end": r(cfg["t_end"], 6),
                    "max_error": {k: r(v, 4) for k, v in cfg["max_error"].items()}},
           "order": {"nu": o["nu"], "sigma": o["sigma"], "t_end": o["t_end"],
                     "h": o["h"],
                     "error": {m: lap(v, 4) for m, v in o["error"].items()},
                     "slope": {m: r(v, 5) for m, v in o["slope"].items()}}}
    print(f"    line-accuracy: race {out['race']['max_error']}, "
          f"slopes {out['order']['slope']} [{time.time() - _t0:.0f} s]", flush=True)
    return out


def figure_line_spiral():
    """Part 1 UNDERSTAND: one mode's complex amplitude under the three steppers,
    against the exact spiral exp(lambda_k t)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nu, kk, h, t_end = LINE_NU, 4, 0.15, 6.0
    lam = -nu * kk ** 2 - 1j * C_LINE * kk
    F = (lambda y: lam * y)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    tt = np.linspace(0, t_end, 2000)
    ex = np.exp(lam * tt)
    ax.plot(ex.real, ex.imag, "k-", lw=2.5, alpha=0.3, label="exact")
    out = {"nu": nu, "k": kk, "h": h, "t_end": t_end,
           "lam": [r(lam.real, 4), r(lam.imag, 4)],
           "abs_exact": r(abs(ex[-1]), 4), "abs_final": {}, "abs_R": {}}
    for name, col, lab in (("euler", "tab:red", "forward Euler"),
                           ("rk2", "tab:orange", "midpoint"),
                           ("rk4", "tab:blue", "RK4")):
        y, path = 1.0 + 0j, [1.0 + 0j]
        for _ in range(int(t_end / h)):
            y = line.STEPPERS[name](F, y, h)
            path.append(y)
        p = np.array(path)
        ax.plot(p.real, p.imag, "-o", color=col, ms=2.2, lw=0.9, label=lab)
        out["abs_final"][name] = r(abs(p[-1]), 4)
        out["abs_R"][name] = r(abs(line.R[name](lam * h)), 4)
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel(r"Re $\hat u_k$")
    ax.set_ylabel(r"Im $\hat u_k$")
    ax.legend(fontsize=8, loc="lower right",
              title=r"$k$ = %d, $h$ = %g" % (kk, h), title_fontsize=8)
    _plain(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "line-spiral.png"), dpi=150)
    plt.close(fig)
    print(f"    line-spiral: |y(T)| {out['abs_final']}, exact {out['abs_exact']}, "
          f"|R| {out['abs_R']} [{time.time() - _t0:.0f} s]", flush=True)
    return out


def figure_line_setup():
    """Part 1 UNDERSTAND, the setup figure: the exact pulse of the line at three times,
    on a fine grid for the curves, the N grid points beside it.  Drawn in CeTZ from
    `line-setup.json` (labels in the sheet, data here)."""
    nu, times, m = LINE_NU, (0.0, 2.0, 5.0), 256
    u0 = line.gaussian(line.SIGMA, line.X0)
    uhat = np.fft.fft(u0) / line.N
    xf = np.arange(m) * line.L / m
    lam = -nu * line.k ** 2 - 1j * C_LINE * line.k
    curves = []
    for t in times:
        coef = uhat * np.exp(lam * t)
        coef[line.KMAX] = 0.0                     # the Nyquist wave, ~1e-16 here
        uf = np.real(coef[None, :] * np.exp(1j * np.outer(xf, line.k))).sum(axis=1)
        curves.append([r(v, 4) for v in uf])
    out = {"nu": nu, "c": C_LINE, "sigma": line.SIGMA, "x0": r(line.X0, 6),
           "times": list(times), "fine": [r(v, 5) for v in xf], "curves": curves,
           "xj": [r(v, 5) for v in line.x], "u0": [r(v, 4) for v in u0],
           "peak": [r(max(c), 4) for c in curves]}
    write("line-setup.json", out)
    print(f"    line-setup: peaks {out['peak']} at t = {times}", flush=True)


def figure_line():
    """The three Part 1 figures and the values they carry."""
    figure_line_setup()
    write("line-figures.json", {
        "toy": {"n": line.N, "c": C_LINE, "t_end": line.T_END,
                "sigma": line.SIGMA, "x0": r(line.X0, 6)},
        "stability": figure_line_stability(),
        "accuracy": figure_line_accuracy(),
        "spiral": figure_line_spiral(),
        "source": "fixtures/week4-reference/line.py (NumPy baseline), for the pictures; "
                  "the sheet quotes week4/data/line.json (Rust answer key), which "
                  "agrees with these to round-off except where a round-off-seeded "
                  "mode sets the answer (the blow-up time and the RK4 bracketing "
                  "step at nu = 0.05)",
    })


def main(only):
    def want(name):
        return not only or name in only

    if want("convergence"):
        figure_convergence()
    if want("tg"):
        figure_tg()
    if want("concepts"):
        figure_concepts()
    if want("line"):
        figure_line()
    elif want("line-setup"):
        figure_line_setup()

    if want("stability"):
        figure_stability_region()
    if want("blowup"):
        figure_blowup()

    if want("snapshots") or want("budget") or want("transfer"):
        cfg = sim.RND
        g = sim.grid(cfg["n"])
        print(f"  random RK4 run (N={cfg['n']}, nu={cfg['nu']}, "
              f"seed={sim.SEED}) ...", flush=True)
        t, e, z, snaps, wend_hat, _ = sim.frames_of(cfg, keep=KEEP)
        if want("snapshots"):
            figure_snapshots(snaps, t, e, z, g)
        if want("budget") or want("transfer"):
            print("  no-advection run ...", flush=True)
            tn, en, zn, _, wna_hat, _ = sim.frames_of(cfg, advect=False)
            if want("budget"):
                print(f"  forward Euler at dt={EULER_DT} ...", flush=True)
                te, ee, ze = sim.frames_of(dict(cfg, dt=EULER_DT),
                                           scheme="euler")[:3]
                figure_budget(t, e, z, {"euler": (ee, ze), "noadvect": (en, zn)})
            if want("transfer"):
                figure_transfer(sim.initial(cfg), wend_hat, wna_hat, g)
    print(f"done in {time.time() - _t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", action="append", default=[],
                    choices=["snapshots", "tg", "budget", "transfer", "convergence",
                             "concepts", "stability", "blowup", "line", "line-setup"],
                    help="regenerate only these outputs (repeatable)")
    main(ap.parse_args().only)
