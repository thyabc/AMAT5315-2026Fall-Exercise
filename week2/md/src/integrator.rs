use crate::System;

/// Shared time-step contract. Implementations mutate the borrowed particle state.
pub trait Integrator {
    fn step(&self, system: &mut System, dt: f64);
}

pub fn advance(method: &impl Integrator, system: &mut System, dt: f64) {
    assert!(dt.is_finite() && dt > 0.0);
    method.step(system, dt);
}

/// Forward Euler: both updates use the old state.
pub struct Euler;

impl Integrator for Euler {
    fn step(&self, system: &mut System, dt: f64) {
        for ((x, v), a) in system
            .positions
            .iter_mut()
            .zip(&mut system.velocities)
            .zip(&system.accelerations)
        {
            for axis in 0..2 {
                x[axis] += dt * v[axis];
                v[axis] += dt * a[axis];
            }
        }
        system.update_accelerations();
    }
}

pub struct VelocityVerlet;

impl Integrator for VelocityVerlet {
    fn step(&self, system: &mut System, dt: f64) {
        // Half kick with cached old accelerations, then drift.
        for ((x, v), a) in system
            .positions
            .iter_mut()
            .zip(&mut system.velocities)
            .zip(&system.accelerations)
        {
            for axis in 0..2 {
                v[axis] += 0.5 * dt * a[axis];
                x[axis] += dt * v[axis];
            }
        }
        // One new force evaluation; retain these accelerations for the next step.
        system.update_accelerations();
        for (v, a) in system.velocities.iter_mut().zip(&system.accelerations) {
            for axis in 0..2 {
                v[axis] += 0.5 * dt * a[axis];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_euler_uses_old_velocities() {
        let mut system = System::new(vec![[0.0, 0.0], [1.2, 0.0]], vec![[0.0; 2]; 2]);
        let initial = system.clone();
        advance(&Euler, &mut system, 0.01);
        assert_eq!(system.positions(), initial.positions());
        for i in 0..2 {
            assert_eq!(
                system.velocities()[i][0],
                0.01 * initial.accelerations()[i][0]
            );
        }
    }

    fn free_particle(method: &impl Integrator) {
        let mut system = System::new(vec![[0.0; 2]], vec![[1.0, -2.0]]);
        advance(method, &mut system, 0.5);
        assert_eq!(system.positions(), &[[0.5, -1.0]]);
        assert_eq!(system.velocities(), &[[1.0, -2.0]]);
    }

    #[test]
    fn both_methods_preserve_free_flight() {
        free_particle(&Euler);
        free_particle(&VelocityVerlet);
    }

    #[test]
    fn verlet_reverses_after_200_steps() {
        let mut system = System::new(vec![[0.0, 0.0], [1.2, 0.0]], vec![[0.0; 2]; 2]);
        let initial = system.clone();
        for _ in 0..200 {
            advance(&VelocityVerlet, &mut system, 0.01);
        }
        system.reverse_velocities();
        for _ in 0..200 {
            advance(&VelocityVerlet, &mut system, 0.01);
        }
        for (actual, expected) in system
            .positions()
            .iter()
            .flatten()
            .zip(initial.positions().iter().flatten())
        {
            assert!((actual - expected).abs() < 1e-11);
        }
        assert!(
            system
                .velocities()
                .iter()
                .flatten()
                .all(|v| v.abs() < 1e-11)
        );
    }
}
