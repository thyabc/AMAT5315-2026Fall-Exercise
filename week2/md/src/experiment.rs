use crate::{Integrator, System, advance};

#[derive(Debug)]
pub struct EnergySample {
    pub step: usize,
    pub time: f64,
    pub kinetic: f64,
    pub potential: f64,
    pub total: f64,
    pub relative_error: f64,
}

/// The Part 3 experiment: identical initial state and measurements for any integrator.
/// Includes step zero and every subsequent step, using open boundaries.
pub fn run_dimer(method: &impl Integrator, dt: f64, steps: usize) -> Vec<EnergySample> {
    assert!(dt.is_finite() && dt > 0.0);
    let mut system = System::new(vec![[0.0, 0.0], [1.2, 0.0]], vec![[0.0; 2]; 2]);
    let initial_energy = system.total_energy();
    let mut samples = Vec::with_capacity(steps + 1);
    for step in 0..=steps {
        if step > 0 {
            advance(method, &mut system, dt);
        }
        let kinetic = system.kinetic_energy();
        let potential = system.potential_energy();
        let total = kinetic + potential;
        samples.push(EnergySample {
            step,
            time: step as f64 * dt,
            kinetic,
            potential,
            total,
            relative_error: (total - initial_energy) / initial_energy.abs(),
        });
    }
    samples
}
