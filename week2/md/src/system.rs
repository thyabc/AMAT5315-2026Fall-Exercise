use crate::{energy, force};

/// Unit-mass particles in two dimensions, with open boundaries and plain LJ forces.
/// State is private to keep the acceleration cache consistent with positions.
#[derive(Clone, Debug)]
pub struct System {
    pub(crate) positions: Vec<[f64; 2]>,
    pub(crate) velocities: Vec<[f64; 2]>,
    pub(crate) accelerations: Vec<[f64; 2]>,
}

impl System {
    pub fn new(positions: Vec<[f64; 2]>, velocities: Vec<[f64; 2]>) -> Self {
        assert_eq!(positions.len(), velocities.len());
        assert!(
            positions
                .iter()
                .chain(&velocities)
                .flatten()
                .all(|x| x.is_finite())
        );
        let accelerations = vec![[0.0; 2]; positions.len()];
        let mut system = Self {
            positions,
            velocities,
            accelerations,
        };
        system.update_accelerations();
        system
    }

    pub fn n_atoms(&self) -> usize {
        self.positions.len()
    }
    pub fn positions(&self) -> &[[f64; 2]] {
        &self.positions
    }
    pub fn velocities(&self) -> &[[f64; 2]] {
        &self.velocities
    }
    pub fn accelerations(&self) -> &[[f64; 2]] {
        &self.accelerations
    }

    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self
            .velocities
            .iter()
            .map(|v| v[0] * v[0] + v[1] * v[1])
            .sum::<f64>()
    }

    pub fn potential_energy(&self) -> f64 {
        let mut potential = 0.0;
        for i in 0..self.n_atoms() {
            for j in i + 1..self.n_atoms() {
                let dx = self.positions[i][0] - self.positions[j][0];
                let dy = self.positions[i][1] - self.positions[j][1];
                potential += energy(dx.hypot(dy));
            }
        }
        potential
    }

    pub fn total_energy(&self) -> f64 {
        self.kinetic_energy() + self.potential_energy()
    }

    pub fn reverse_velocities(&mut self) {
        for v in &mut self.velocities {
            for component in v {
                *component = -*component;
            }
        }
    }

    pub(crate) fn update_accelerations(&mut self) {
        self.accelerations.fill([0.0; 2]);
        for i in 0..self.n_atoms() {
            for j in i + 1..self.n_atoms() {
                let d = [
                    self.positions[i][0] - self.positions[j][0],
                    self.positions[i][1] - self.positions[j][1],
                ];
                let r = d[0].hypot(d[1]);
                assert!(
                    r > 0.0 && r.is_finite(),
                    "pair separation must be finite and positive"
                );
                let scale = force(r) / r;
                for (axis, displacement) in d.into_iter().enumerate() {
                    let acceleration = scale * displacement;
                    self.accelerations[i][axis] += acceleration;
                    self.accelerations[j][axis] -= acceleration;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_forces_are_equal_opposite_and_along_displacement() {
        let system = System::new(vec![[0.0, 0.0], [0.72, 0.96]], vec![[0.0; 2]; 2]);
        for (axis, direction) in [0.6, 0.8].into_iter().enumerate() {
            assert!((system.accelerations()[0][axis] + force(1.2) * direction).abs() < 1e-12);
            assert_eq!(
                system.accelerations()[0][axis],
                -system.accelerations()[1][axis]
            );
        }
        assert!((system.potential_energy() - energy(1.2)).abs() < 1e-12);
    }

    #[test]
    fn kinetic_energy_includes_both_components_and_particles() {
        let system = System::new(vec![[0.0, 0.0], [1.2, 0.0]], vec![[3.0, 4.0], [1.0, -2.0]]);
        assert_eq!(system.kinetic_energy(), 15.0);
    }
}
