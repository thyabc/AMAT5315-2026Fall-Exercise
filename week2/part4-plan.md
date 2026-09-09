# Part 4 approved design

Extend the existing 2D, unit-mass md crate while preserving Part 3 behavior.
Use a periodic rectangular box, minimum-image pair displacements, and energy-shifted
Lennard-Jones interactions with rc=2.5 (ordinary LJ force below rc, zero at/above rc).
Initialize 100 particles in 10 staggered rows with rho=0.8, Gaussian velocities
from seed 2026, zero center-of-mass momentum, and initial T=1.0.
Use dt=0.005 and velocity-Verlet. Rescale velocities after each of 2000 equilibration
steps; run 10000 production steps without a thermostat. Define T=2K/(2N-2).
Record production step zero and every 50 steps (201 frames), with energy and
temperature every step (10001 rows). Store positions and velocities in frames.

Commands: md run, md check, md video; make reproduce regenerates the experiment.
Keep the no-argument Part 3 CSV command. Stream CSV data and store parameters in
metadata.json. Reject invalid configuration and malformed/incomplete run data.

Acceptance thresholds fixed before experiments: max |E-E0|/K0 < 0.01;
mean production T within 5% of target; speed empirical CDF maximum discrepancy
< 0.05 against the 2D Maxwell distribution at measured temperature, correcting
for the center-of-mass constraint. Report drift slope and block temperatures.
These are deterministic acceptance bounds, not independent-sample p-values.

TDD sequence: boundary/potential; lattice/velocities; periodic integration;
thermostat and phase sequencing; recording/CLI; checker with synthetic failures;
video; fresh-checkout reproduction. Record RED/GREEN milestones in Git and verify
Rust tests plus pytest throughout. User approved this design before implementation.
