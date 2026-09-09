use crate::{Boundary, FluidConfig, PairPotential, System};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

pub fn initialize_fluid(config: &FluidConfig) -> Result<System, String> {
    config.validate()?;
    let lengths = config.box_lengths()?;
    let side = (config.particles as f64).sqrt() as usize;
    let spacing = lengths[0] / side as f64;
    let mut positions = Vec::with_capacity(config.particles);
    for row in 0..side {
        for col in 0..side {
            positions.push([
                (col as f64 + 0.5 * (row % 2) as f64) * spacing,
                row as f64 * lengths[1] / side as f64,
            ]);
        }
    }
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut velocities: Vec<[f64; 2]> = (0..config.particles)
        .map(|_| {
            [
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
            ]
        })
        .collect();
    for axis in 0..2 {
        let mean = velocities.iter().map(|v| v[axis]).sum::<f64>() / config.particles as f64;
        for v in &mut velocities {
            v[axis] -= mean;
        }
    }
    let mut system = System::with_settings(
        positions,
        velocities,
        Boundary::periodic(lengths)?,
        PairPotential::shifted(config.cutoff)?,
    )?;
    system.rescale_temperature(config.temperature)?;
    Ok(system)
}
