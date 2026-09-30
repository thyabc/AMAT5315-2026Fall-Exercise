import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


jax.config.update("jax_enable_x64", True)


# ============================================================
# Build cubic lattice
# ============================================================

def make_cluster(n):
    spacing = 2.0 ** (1.0 / 6.0)

    # Smallest cubic grid containing at least n atoms
    side = 1
    while side ** 3 < n:
        side += 1

    grid = np.stack(
        np.meshgrid(
            np.arange(side),
            np.arange(side),
            np.arange(side),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)

    positions = grid[:n].astype(np.float64) * spacing

    # Reproducible Gaussian displacement, std = 0.05
    rng = np.random.default_rng(12345 + n)
    positions += rng.normal(
        loc=0.0,
        scale=0.05,
        size=positions.shape,
    )

    return positions


# ============================================================
# Analytic Lennard-Jones forces
# ============================================================

def analytic_forces(positions, ii, jj):
    """
    Return analytic forces for
        E = sum_{i<j} 4(r^-12 - r^-6)
    """

    d = positions[ii] - positions[jj]

    r2 = np.sum(d * d, axis=1)
    inv_r2 = 1.0 / r2

    # grad_i U =
    # 24 * (r^-8 - 2 r^-14) * (x_i - x_j)

    coeff = 24.0 * (
        inv_r2 ** 4
        - 2.0 * inv_r2 ** 7
    )

    pair_grad = coeff[:, None] * d

    gradient = np.zeros_like(positions)

    np.add.at(
        gradient,
        ii,
        pair_grad,
    )

    np.add.at(
        gradient,
        jj,
        -pair_grad,
    )

    # Force = - gradient
    return -gradient.reshape(-1)


# ============================================================
# Timing helper
# ============================================================

def average_time(fn, x, repeats=5):
    times = []

    for _ in range(repeats):
        start = time.perf_counter()

        result = fn(x)
        result.block_until_ready()

        end = time.perf_counter()

        times.append(end - start)

    return float(np.median(times))


# ============================================================
# One problem size
# ============================================================

def run_size(n):

    positions = make_cluster(n)

    p = 3 * n

    ii_np, jj_np = np.triu_indices(n, k=1)

    ii = jnp.asarray(ii_np)
    jj = jnp.asarray(jj_np)

    x = jnp.asarray(
        positions.reshape(-1),
        dtype=jnp.float64,
    )

    # --------------------------------------------------------
    # Cluster energy
    # --------------------------------------------------------

    def cluster_energy(x_flat):

        pos = x_flat.reshape((n, 3))

        d = pos[ii] - pos[jj]

        r2 = jnp.sum(
            d * d,
            axis=1,
        )

        # r^-6 = (r^2)^-3
        inv_r6 = r2 ** -3

        pair_energy = 4.0 * (
            inv_r6 ** 2 - inv_r6
        )

        return jnp.sum(pair_energy)

    # --------------------------------------------------------
    # Reverse mode
    # --------------------------------------------------------

    reverse_grad = jax.jit(
        jax.grad(cluster_energy)
    )

    # --------------------------------------------------------
    # Forward mode:
    # one input direction at a time
    # --------------------------------------------------------

    def one_forward_component(x_flat, k):

        tangent = jax.nn.one_hot(
            k,
            p,
            dtype=x_flat.dtype,
        )

        _, tangent_energy = jax.jvp(
            cluster_energy,
            (x_flat,),
            (tangent,),
        )

        return tangent_energy

    forward_component = jax.jit(
        one_forward_component
    )

    energy_jit = jax.jit(cluster_energy)

    # --------------------------------------------------------
    # Compile first -- do not include compilation in timing
    # --------------------------------------------------------

    print("  compiling...")

    energy_jit(x).block_until_ready()

    reverse_grad(x).block_until_ready()

    forward_component(
        x,
        jnp.int32(0)
    ).block_until_ready()

    # --------------------------------------------------------
    # Energy timing
    # --------------------------------------------------------

    energy_time = average_time(
        energy_jit,
        x,
        repeats=7,
    )

    # --------------------------------------------------------
    # Reverse gradient timing
    # --------------------------------------------------------

    reverse_time = average_time(
        reverse_grad,
        x,
        repeats=5,
    )

    reverse_values = np.asarray(
        reverse_grad(x)
    )

    # --------------------------------------------------------
    # Forward gradient timing
    # --------------------------------------------------------

    print(
        f"  forward mode: {p} JVPs..."
    )

    forward_values = np.empty(
        p,
        dtype=np.float64,
    )

    start = time.perf_counter()

    for k in range(p):

        value = forward_component(
            x,
            jnp.int32(k),
        )

        # float() synchronizes this scalar result
        forward_values[k] = float(value)

    forward_time = (
        time.perf_counter() - start
    )

    # --------------------------------------------------------
    # Analytic forces
    # --------------------------------------------------------

    force_true = analytic_forces(
        positions,
        ii_np,
        jj_np,
    )

    # AD gives gradient, force = -gradient
    force_forward = -forward_values
    force_reverse = -reverse_values

    denominator = np.max(
        np.abs(force_true)
    )

    forward_error = (
        np.max(
            np.abs(
                force_forward - force_true
            )
        )
        / denominator
    )

    reverse_error = (
        np.max(
            np.abs(
                force_reverse - force_true
            )
        )
        / denominator
    )

    forward_ratio = (
        forward_time / energy_time
    )

    reverse_ratio = (
        reverse_time / energy_time
    )

    print(
        f"  energy time       = "
        f"{energy_time:.6e} s"
    )

    print(
        f"  forward time      = "
        f"{forward_time:.6e} s"
    )

    print(
        f"  reverse time      = "
        f"{reverse_time:.6e} s"
    )

    print(
        f"  forward ratio     = "
        f"{forward_ratio:.3f}"
    )

    print(
        f"  reverse ratio     = "
        f"{reverse_ratio:.3f}"
    )

    print(
        f"  forward rel error = "
        f"{forward_error:.3e}"
    )

    print(
        f"  reverse rel error = "
        f"{reverse_error:.3e}"
    )

    return {
        "N": n,
        "P": p,
        "energy_time": energy_time,
        "forward_time": forward_time,
        "reverse_time": reverse_time,
        "forward_ratio": forward_ratio,
        "reverse_ratio": reverse_ratio,
        "forward_error": forward_error,
        "reverse_error": reverse_error,
    }


# ============================================================
# Main
# ============================================================

def main():

    sizes = [
        64,
        128,
        256,
        512,
        1024,
    ]

    results = []

    for n in sizes:

        print()
        print("=" * 55)
        print(
            f"N = {n}, inputs P = {3*n}"
        )
        print("=" * 55)

        results.append(
            run_size(n)
        )

    p_values = np.array(
        [r["P"] for r in results]
    )

    forward_ratios = np.array(
        [
            r["forward_ratio"]
            for r in results
        ]
    )

    reverse_ratios = np.array(
        [
            r["reverse_ratio"]
            for r in results
        ]
    )

    forward_errors = np.array(
        [
            r["forward_error"]
            for r in results
        ]
    )

    reverse_errors = np.array(
        [
            r["reverse_error"]
            for r in results
        ]
    )

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print()
    print("FINAL SUMMARY")
    print("=" * 78)

    print(
        f"{'N':>6} "
        f"{'P=3N':>7} "
        f"{'Forward ratio':>16} "
        f"{'Reverse ratio':>16} "
        f"{'Forward error':>16} "
        f"{'Reverse error':>16}"
    )

    for r in results:

        print(
            f"{r['N']:6d} "
            f"{r['P']:7d} "
            f"{r['forward_ratio']:16.3f} "
            f"{r['reverse_ratio']:16.3f} "
            f"{r['forward_error']:16.3e} "
            f"{r['reverse_error']:16.3e}"
        )

    print()
    print(
        "Largest forward relative error:",
        np.max(forward_errors),
    )

    print(
        "Largest reverse relative error:",
        np.max(reverse_errors),
    )

    final_speed_ratio = (
        forward_ratios[-1]
        / reverse_ratios[-1]
    )

    print(
        "At P=3072, forward/reverse ratio:",
        final_speed_ratio,
    )

    if (
        np.max(forward_errors) < 1e-12
        and np.max(reverse_errors) < 1e-12
    ):
        print(
            "PASS: both gradient errors < 1e-12"
        )
    else:
        print(
            "FAIL: gradient error too large"
        )

    if final_speed_ratio > 100:
        print(
            "PASS: forward ratio > "
            "100 x reverse ratio at P=3072"
        )
    else:
        print(
            "WARNING: timing criterion not reached"
        )

    # ========================================================
    # Plot
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7.5, 5.5)
    )

    ax.loglog(
        p_values,
        forward_ratios,
        "o-",
        label="forward mode, one JVP per input",
    )

    ax.loglog(
        p_values,
        reverse_ratios,
        "s-",
        label="reverse mode, one VJP",
    )

    # Reference line proportional to P
    proportional = (
        forward_ratios[0]
        * p_values
        / p_values[0]
    )

    ax.loglog(
        p_values,
        proportional,
        ":",
        label="proportional to P",
    )

    ax.set_xlabel(
        "Inputs P = 3N"
    )

    ax.set_ylabel(
        "Gradient time / energy time"
    )

    ax.set_title(
        "Cost of a Lennard-Jones cluster gradient"
    )

    ax.grid(
        True,
        which="both",
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    out = Path(
        "artifacts/ad/scaling.png"
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        out,
        dpi=200,
    )

    plt.close(fig)

    print()
    print(
        "Saved:",
        out,
    )


if __name__ == "__main__":
    main()
