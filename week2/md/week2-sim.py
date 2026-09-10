#!/usr/bin/env python3
"""Reference student solver: 2D Lennard-Jones MD via velocity-Verlet.

Velocity-Verlet is symplectic, so total energy stays bounded over the 10^4-step
NVE production run -- the week-2 gate PASSes on this trajectory. Emits the RAW
per-frame trajectory (pos, vel, E_pot, E_kin); the checker recomputes every LJ
energy itself from the logged positions, so the artifact must be self-consistent.

Reduced units: eps = sigma = m = kB = 1. Cutoff rc = 2.5, potential shifted so
U(rc) = 0. Density rho = 0.8 on a 10x10 triangular lattice (N = 100).
"""
import json
import os

import numpy as np

INTEGRATOR = "velocity-verlet"   # reference fixture: symplectic -> energy bounded

RC = 2.5
RC2 = RC * RC
U_SHIFT = 4.0 * (RC**-12 - RC**-6)   # shift so U(rc) = 0

NX, NY = 10, 10
RHO = 0.8
TEMP = 0.5
DT = 0.01
EQ_STEPS = 2000
PROD_STEPS = 10000
SAMPLE_EVERY = 50
SEED = 42


def tri_lattice(nx, ny, rho):
    """Triangular lattice, nx columns x ny rows, at density rho -> pos, [Lx, Ly]."""
    a = np.sqrt(2.0 / (np.sqrt(3.0) * rho))
    h = a * np.sqrt(3.0) / 2.0
    Lx, Ly = nx * a, ny * h
    pos = np.zeros((nx * ny, 2))
    k = 0
    for j in range(ny):
        for i in range(nx):
            pos[k] = ((i + 0.5 * (j % 2)) * a, j * h)
            k += 1
    return pos, np.array([Lx, Ly])


def forces_energy(pos, box):
    """O(N^2) minimum-image LJ forces + shifted potential energy (cutoff RC)."""
    d = pos[:, None, :] - pos[None, :, :]
    d -= box * np.round(d / box)
    r2 = np.einsum('ijk,ijk->ij', d, d)
    np.fill_diagonal(r2, np.inf)
    mask = r2 < RC2
    inv_r2 = np.where(mask, 1.0 / r2, 0.0)
    inv_r6 = inv_r2 ** 3
    u = np.where(mask, 4.0 * (inv_r6 * inv_r6 - inv_r6) - U_SHIFT, 0.0)
    fac = 24.0 * (2.0 * inv_r6 * inv_r6 - inv_r6) * inv_r2
    f = np.einsum('ij,ijk->ik', fac, d)
    return f, 0.5 * u.sum()


def ekin(vel):
    return 0.5 * float(np.einsum('ij,ij->', vel, vel))


def temperature(vel):
    n = vel.shape[0]
    return 2.0 * ekin(vel) / (2.0 * n - 2.0)   # 2D, COM removed


def init_vel(n, T, rng):
    v = rng.normal(0.0, np.sqrt(T), size=(n, 2))
    v -= v.mean(axis=0)
    v *= np.sqrt(T / temperature(v))
    return v


def step_verlet(pos, vel, f, box, dt):
    vel = vel + 0.5 * dt * f
    pos = (pos + dt * vel) % box
    f_new, epot = forces_energy(pos, box)
    vel = vel + 0.5 * dt * f_new
    return pos, vel, f_new, epot


def step_rk4(pos, vel, box, dt):
    """Classic RK4 on the coupled (pos, vel) ODE. 4 force evaluations per step."""
    def deriv(p, v):
        f, _ = forces_energy(p, box)
        return v, f
    k1p, k1v = deriv(pos, vel)
    k2p, k2v = deriv(pos + 0.5 * dt * k1p, vel + 0.5 * dt * k1v)
    k3p, k3v = deriv(pos + 0.5 * dt * k2p, vel + 0.5 * dt * k2v)
    k4p, k4v = deriv(pos + dt * k3p, vel + dt * k3v)
    pos = (pos + dt / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)) % box
    vel = vel + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    f, epot = forces_energy(pos, box)
    return pos, vel, f, epot


def r9(a):
    """Round an array to 9 significant digits (round-trip through %.9g)."""
    flat = [float("%.9g" % x) for x in np.asarray(a).ravel()]
    return np.asarray(flat).reshape(np.asarray(a).shape)


def main():
    os.makedirs("artifacts", exist_ok=True)
    rng = np.random.default_rng(SEED)
    pos, box = tri_lattice(NX, NY, RHO)
    n = pos.shape[0]
    vel = init_vel(n, TEMP, rng)
    f, epot = forces_energy(pos, box)

    # equilibration: velocity-Verlet + velocity rescaling to TEMP every 50 steps
    for s in range(EQ_STEPS):
        pos, vel, f, epot = step_verlet(pos, vel, f, box, DT)
        if s % 50 == 0:
            vel *= np.sqrt(TEMP / temperature(vel))
    vel -= vel.mean(axis=0)   # zero COM momentum before production

    run = {
        "n": n, "rho": RHO, "box": [float(box[0]), float(box[1])],
        "dt": DT, "temperature": TEMP, "eq_steps": EQ_STEPS,
        "steps": PROD_STEPS, "sample_every": SAMPLE_EVERY, "seed": SEED,
        "integrator": INTEGRATOR,
    }
    with open(os.path.join("artifacts", "run.json"), "w") as fh:
        json.dump(run, fh, indent=2)

    # production: NVE with the chosen integrator; sample every SAMPLE_EVERY steps
    nframes = 0
    with open(os.path.join("artifacts", "traj.jsonl"), "w") as fh:
        for s in range(PROD_STEPS):
            if INTEGRATOR == "velocity-verlet":
                pos, vel, f, epot = step_verlet(pos, vel, f, box, DT)
            else:
                pos, vel, f, epot = step_rk4(pos, vel, box, DT)
            if (s + 1) % SAMPLE_EVERY == 0:
                # Round the logged config, then compute its energies from the
                # rounded values so the artifact is self-consistent at full
                # precision (the checker recomputes E_pot/E_kin from raw pos/vel).
                pos_r = r9(pos % box)
                vel_r = r9(vel)
                _, epot_r = forces_energy(pos_r, box)
                ekin_r = ekin(vel_r)
                frame = {
                    "step": s + 1, "t": (s + 1) * DT,
                    "pos": pos_r.tolist(), "vel": vel_r.tolist(),
                    "E_pot": epot_r, "E_kin": ekin_r,
                }
                fh.write(json.dumps(frame) + "\n")
                nframes += 1
    print(f"wrote artifacts/run.json + traj.jsonl "
          f"({nframes} production frames, {INTEGRATOR})")


if __name__ == "__main__":
    main()
