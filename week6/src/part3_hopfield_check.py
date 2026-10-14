from pathlib import Path
import json
import math

import numpy as np
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

N = 100
P = 10
SEED = 2026
MAX_SWEEPS = 100

rng = np.random.default_rng(SEED)

# Random stored patterns: each spin is independently -1 or +1.
patterns_np = (
    2 * rng.integers(0, 2, size=(P, N)) - 1
).astype(np.float32)

patterns = jnp.asarray(patterns_np)

# Hebbian couplings, with diagonal exactly zero.
J = (patterns.T @ patterns) / N
J = J.at[jnp.diag_indices(N)].set(0.0)

assert bool(jnp.allclose(J, J.T, atol=1e-6))
assert bool(jnp.all(jnp.diag(J) == 0))


@jax.jit
def energy(s, coupling):
    return -0.5 * (s @ coupling @ s)


@jax.jit
def async_sweep(s, coupling, order):
    """Update spins individually in the given random order."""
    h = coupling @ s

    def step(carry, i):
        spin, field, flips = carry

        old_value = spin[i]
        new_value = jnp.where(
            field[i] > 0,
            1.0,
            jnp.where(field[i] < 0, -1.0, old_value)
        )

        delta = new_value - old_value
        spin = spin.at[i].set(new_value)

        # Incremental local-field update.
        field = field + delta * coupling[:, i]
        flips = flips + (delta != 0).astype(jnp.int32)

        return (spin, field, flips), None

    (s_final, _, flips), _ = jax.lax.scan(
        step,
        (s, h, jnp.array(0, dtype=jnp.int32)),
        order
    )

    return s_final, flips


# Unstable bits at the stored configurations.
fields = patterns @ J
unstable_fraction = float(
    jnp.mean((patterns * fields) < 0)
)

alpha = P / N

theory_large = 0.5 * math.erfc(
    1.0 / math.sqrt(2.0 * alpha)
)

theory_finite = 0.5 * math.erfc(
    math.sqrt((N - 1) / (2.0 * (P - 1)))
)

print("===== Hopfield Network Test =====")
print("N =", N)
print("P =", P)
print("alpha =", alpha)
print("J shape =", J.shape)
print("Diagonal J_ii = 0: PASSED")
print("Symmetric coupling matrix: PASSED")
print()
print("Measured unstable fraction:", unstable_fraction)
print("Large-N Gaussian prediction:", theory_large)
print("Finite-N Gaussian prediction:", theory_finite)

final_overlaps = []
converged = []

print("\n===== Recall from stored patterns =====")

for p in range(min(P, 10)):
    s = patterns[p]
    initial_energy = float(energy(s, J))
    previous_energy = initial_energy
    is_converged = False

    for sweep in range(MAX_SWEEPS):
        order = jnp.asarray(rng.permutation(N))
        s, flips = async_sweep(s, J, order)

        current_energy = float(energy(s, J))

        # Zero-temperature asynchronous recall must
        # never increase the Ising energy.
        assert current_energy <= previous_energy + 1e-4
        previous_energy = current_energy

        if int(flips) == 0:
            is_converged = True
            break

    overlap = float(jnp.mean(patterns[p] * s))
    final_overlaps.append(overlap)
    converged.append(is_converged)

    print(
        f"Pattern {p+1:02d} | "
        f"overlap={overlap:.4f} | "
        f"energy={initial_energy:.3f}->{previous_energy:.3f} | "
        f"sweeps={sweep+1} | "
        f"converged={is_converged}"
    )

assert all(converged), "Some recalls failed to converge."

summary = {
    "N": N,
    "P": P,
    "alpha": alpha,
    "seed": SEED,
    "measured_unstable_fraction": unstable_fraction,
    "theory_large_N": theory_large,
    "theory_finite_N": theory_finite,
    "final_overlaps": final_overlaps,
    "mean_final_overlap": float(np.mean(final_overlaps)),
    "converged": converged
}

path = ARTIFACTS / "part3_initial_check.json"
path.write_text(json.dumps(summary, indent=2))

print("\n===== Summary =====")
print("Mean final overlap:", np.mean(final_overlaps))
print("All recalls converged: PASSED")
print("Zero-temperature energy check: PASSED")
print("Saved:", path)
