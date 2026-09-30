import json
from pathlib import Path

import jax
import jax.numpy as jnp


# Enable 64-bit floating point.
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------
# Individual nodes of the computational graph
# a = r^-6
# b = a^2
# c = b - a
# U = 4c
# ---------------------------------------------------------

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


def main():
    r = jnp.array(1.3, dtype=jnp.float64)

    # =====================================================
    # Forward mode
    # =====================================================

    # Seed tangent dr/dr = 1
    r_dot = jnp.array(1.0, dtype=jnp.float64)

    # a = r^-6
    a, a_dot = jax.jvp(
        node_a,
        (r,),
        (r_dot,),
    )

    # b = a^2
    b, b_dot = jax.jvp(
        node_b,
        (a,),
        (a_dot,),
    )

    # c = b - a
    c, c_dot = jax.jvp(
        node_c,
        (b, a),
        (b_dot, a_dot),
    )

    # U = 4c
    U, U_dot = jax.jvp(
        node_U,
        (c,),
        (c_dot,),
    )

    # =====================================================
    # Reverse mode
    # =====================================================

    # Seed output adjoint dU/dU = 1
    U_bar = jnp.array(1.0, dtype=jnp.float64)

    # U = 4c
    _, pullback_U = jax.vjp(node_U, c)
    (c_bar,) = pullback_U(U_bar)

    # c = b - a
    _, pullback_c = jax.vjp(node_c, b, a)
    b_bar, a_bar_from_c = pullback_c(c_bar)

    # b = a^2
    _, pullback_b = jax.vjp(node_b, a)
    (a_bar_from_b,) = pullback_b(b_bar)

    # a is shared by two paths, so ADD the contributions
    a_bar = a_bar_from_c + a_bar_from_b

    # a = r^-6
    _, pullback_a = jax.vjp(node_a, r)
    (r_bar,) = pullback_a(a_bar)

    # JAX built-in gradient for comparison
    jax_grad = jax.grad(energy)(r)

    result = {
        "r": float(r),
        "energy": float(U),
        "tangents": {
            "r": float(r_dot),
            "a": float(a_dot),
            "b": float(b_dot),
            "c": float(c_dot),
            "U": float(U_dot),
        },
        "adjoints": {
            "r": float(r_bar),
            "a": float(a_bar),
            "b": float(b_bar),
            "c": float(c_bar),
            "U": float(U_bar),
        },
        "jax_grad": float(jax_grad),
    }

    out_dir = Path("artifacts/ad")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "derivatives.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("r =", float(r))
    print("energy =", float(U))
    print("forward derivative =", float(U_dot))
    print("reverse derivative =", float(r_bar))
    print("jax.grad =", float(jax_grad))
    print("adjoint a =", float(a_bar))
    print()
    print("Saved:", out_file)


if __name__ == "__main__":
    main()
