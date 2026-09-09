# Part 3: dimer simulation design

This records the design reviewed and approved in the conversation before implementation.

Reuse the existing analytical `energy(r)` and `force(r)` functions and retain the
Part 1 and Part 2 tests. Use unit masses, two spatial dimensions, open boundaries,
and the plain Lennard-Jones potential.

`System` owns positions, velocities, and cached accelerations as `Vec<[f64; 2]>`.
Initialize the cache in the constructor. Keep the arrays private, expose shared
slice borrows for inspection, and let integrators mutate state through `&mut System`.
Compute each pair once with `d = x_i - x_j`, `r = |d|`, and `F_i = force(r) d/r`;
apply the opposite force to atom j. Energy measurements borrow state without copying.

`Integrator::step(&self, &mut System, dt)` is implemented by `Euler` and
`VelocityVerlet`. The generic `advance` function invokes that shared interface.
Euler uses old velocities and accelerations for both updates. Verlet performs a
half kick, drift, one new acceleration evaluation, and a second half kick, retaining
the new accelerations for the next step.

One `run_dimer` driver initializes `(0, 0)` and `(1.2, 0)` with zero velocities,
advances through the trait, and records every step's energy. Both comparison calls
pass `dt = 0.01` and 500 steps; only the integrator value differs. Include step zero.
Record signed relative error `(E(t) - E0) / |E0|`. Use its absolute value to measure
the maximum error.

Acceptance: Verlet maximum absolute error below 1e-3, Euler final signed error
above 0.5. Extend Verlet to 5000 steps and require the same 1e-3 bound throughout.
The plotting example must call the same driver, with both 500-step curves on the
left and the 5000-step Verlet error multiplied by 1000 on the right.

Modules: `system.rs` for state and vector physics, `integrator.rs` for update rules,
`experiment.rs` for sampling. The binary writes comparison CSV. An example renders
`week2/dimer.png`; plotting dependencies are development-only.
