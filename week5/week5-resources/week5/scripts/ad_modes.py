import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

jax.config.update("jax_enable_x64", True)


# =========================================================
# Lennard-Jones energy
# =========================================================

def node_a(r):
    return r ** -6


def node_b(a):
    return a ** 2


def node_c(b, a):
    return b - a


def node_U(c):
    return 4.0 * c


def energy(r):
    a = node_a(r)
    b = node_b(a)
    c = node_c(b, a)
    return node_U(c)


# Analytic derivative
def analytic_derivative(r):
    return 24.0 * (r ** -7 - 2.0 * r ** -13)


# =========================================================
# Hand-written forward mode
# =========================================================

def forward_derivative(r):
    r = jnp.array(r, dtype=jnp.float64)
    r_dot = jnp.array(1.0, dtype=jnp.float64)

    a, a_dot = jax.jvp(
        node_a,
        (r,),
        (r_dot,)
    )

    b, b_dot = jax.jvp(
        node_b,
        (a,),
        (a_dot,)
    )

    c, c_dot = jax.jvp(
        node_c,
        (b, a),
        (b_dot, a_dot)
    )

    U, U_dot = jax.jvp(
        node_U,
        (c,),
        (c_dot,)
    )

    return float(U_dot)


# =========================================================
# Hand-written reverse mode
# =========================================================

def reverse_derivative(r):
    r = jnp.array(r, dtype=jnp.float64)

    a = node_a(r)
    b = node_b(a)
    c = node_c(b, a)

    U_bar = jnp.array(1.0, dtype=jnp.float64)

    _, pb_U = jax.vjp(node_U, c)
    (c_bar,) = pb_U(U_bar)

    _, pb_c = jax.vjp(node_c, b, a)
    b_bar, a_bar_from_c = pb_c(c_bar)

    _, pb_b = jax.vjp(node_b, a)
    (a_bar_from_b,) = pb_b(b_bar)

    # Shared value a receives two reverse contributions
    a_bar = a_bar_from_c + a_bar_from_b

    _, pb_a = jax.vjp(node_a, r)
    (r_bar,) = pb_a(a_bar)

    return float(r_bar)


# =========================================================
# Finite difference
# =========================================================

def finite_difference(r, h=1e-6):
    rp = jnp.array(r + h, dtype=jnp.float64)
    rm = jnp.array(r - h, dtype=jnp.float64)

    return float((energy(rp) - energy(rm)) / (2.0 * h))


def main():

    # 601 equally spaced points, endpoints included
    r_values = np.linspace(0.95, 2.5, 601)

    analytic = np.array(
        [analytic_derivative(r) for r in r_values]
    )

    forward = np.array(
        [forward_derivative(r) for r in r_values]
    )

    reverse = np.array(
        [reverse_derivative(r) for r in r_values]
    )

    finite = np.array(
        [finite_difference(r) for r in r_values]
    )

    # Errors
    forward_error = np.abs(forward - analytic)
    reverse_error = np.abs(reverse - analytic)
    finite_error = np.abs(finite - analytic)

    print("Maximum errors")
    print("-----------------------------")
    print("Forward AD       :", np.max(forward_error))
    print("Reverse AD       :", np.max(reverse_error))
    print("Finite difference:", np.max(finite_error))

    zero_position = 2.0 ** (1.0 / 6.0)
    print()
    print("Analytic zero at r =", zero_position)

    # Avoid exact zero values on logarithmic scale
    eps = np.finfo(float).tiny

    forward_plot_error = np.maximum(forward_error, eps)
    reverse_plot_error = np.maximum(reverse_error, eps)
    finite_plot_error = np.maximum(finite_error, eps)

    # =====================================================
    # Plot
    # =====================================================

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Derivative plot
    ax = axes[0]

    ax.plot(
        r_values,
        analytic,
        label="analytic",
        linewidth=2
    )

    ax.plot(
        r_values,
        forward,
        "--",
        label="forward mode"
    )

    ax.plot(
        r_values,
        reverse,
        ":",
        label="reverse mode"
    )

    ax.plot(
        r_values,
        finite,
        "-.",
        label="finite difference"
    )

    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(
        zero_position,
        linestyle=":",
        linewidth=1
    )

    ax.set_xlabel("Separation r")
    ax.set_ylabel("dU/dr")
    ax.set_title("Lennard-Jones derivative")
    ax.legend()

    # Error plot
    ax = axes[1]

    ax.semilogy(
        r_values,
        forward_plot_error,
        label="forward mode"
    )

    ax.semilogy(
        r_values,
        reverse_plot_error,
        label="reverse mode"
    )

    ax.semilogy(
        r_values,
        finite_plot_error,
        label="finite difference, h=1e-6"
    )

    ax.set_xlabel("Separation r")
    ax.set_ylabel("Absolute error")
    ax.set_title("Error against analytic derivative")
    ax.legend()

    fig.tight_layout()

    out_dir = Path("artifacts/ad")
    out_dir.mkdir(parents=True, exist_ok=True)

    output = out_dir / "modes.png"
    fig.savefig(output, dpi=200)

    plt.close(fig)

    print()
    print("Saved:", output)


if __name__ == "__main__":
    main()
