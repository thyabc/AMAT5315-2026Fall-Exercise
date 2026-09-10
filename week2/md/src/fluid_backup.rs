use crate::{System, VelocityVerlet, advance, initialize_fluid};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Equilibration,
    Production,
}

/// Observers borrow the state. Only production frames are marked for recording.
pub fn simulate(
    config: &FluidConfig,
    mut observe: impl FnMut(Phase, usize, &System, bool) -> Result<(), String>,
) -> Result<(), String> {
    let mut system = initialize_fluid(config)?;
    for step in 1..=config.equilibration_steps {
    advance(&VelocityVerlet, &mut system, config.dt);

    
        system.rescale_temperature(config.temperature)?;
    observe(Phase::Equilibration, step, &system, false)?;
}
    for step in 0..=config.production_steps {
        if step > 0 {
            advance(&VelocityVerlet, &mut system, config.dt);
        }
        if !system
            .positions()
            .iter()
            .chain(system.velocities())
            .chain(system.accelerations())
            .flatten()
            .all(|x| x.is_finite())
        {
            return Err(format!(
                "nonfinite production state at step {step}; reduce dt"
            ));
        }
        let frame = step % config.sample_every == 0 || step == config.production_steps;
        observe(Phase::Production, step, &system, frame)?;
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidConfig {
    pub particles: usize,
    pub density: f64,
    pub temperature: f64,
    pub seed: u64,
    pub dt: f64,
    pub equilibration_steps: usize,
    pub production_steps: usize,
    pub sample_every: usize,
    pub cutoff: f64,
}

impl Default for FluidConfig {
    fn default() -> Self {
        Self {
            particles: 100,
            density: 0.8,
            temperature: 0.5,
            seed: 2026,
            dt: 0.01,
            equilibration_steps: 5000,
            production_steps: 10000,
            sample_every: 50,
            cutoff: 2.5,
        }
    }
}

impl FluidConfig {
    pub fn box_lengths(&self) -> Result<[f64; 2], String> {
        // The supported lattice is an even square number of staggered sites.
        let side = (self.particles as f64).sqrt() as usize;
        if side < 2 || side.checked_mul(side) != Some(self.particles) || !side.is_multiple_of(2) {
            return Err("particles must be an even square (e.g. 100 = 10 x 10)".into());
        }
        if !self.density.is_finite() || self.density <= 0.0 {
            return Err("density must be finite and positive".into());
        }
        let spacing = (2.0 / (3.0_f64.sqrt() * self.density)).sqrt();
        let lengths = [
            side as f64 * spacing,
            side as f64 * spacing * 3.0_f64.sqrt() / 2.0,
        ];
        crate::Boundary::periodic(lengths)?;
        Ok(lengths)
    }

    pub fn validate(&self) -> Result<(), String> {
        let lengths = self.box_lengths()?;
        for (name, value) in [("temperature", self.temperature), ("dt", self.dt)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} must be finite and positive"));
            }
        }
        crate::PairPotential::shifted(self.cutoff)?;
        if lengths.iter().any(|l| self.cutoff > 0.5 * l) {
            return Err("cutoff must not exceed half the shortest box length".into());
        }
        if self.production_steps == 0 || self.sample_every == 0 {
            return Err("steps and sample-every must be positive".into());
        }
        if self.production_steps.checked_add(1).is_none()
            || self
                .equilibration_steps
                .checked_add(self.production_steps)
                .is_none()
            || !(self.production_steps as f64 * self.dt).is_finite()
        {
            return Err("run length is too large".into());
        }
        Ok(())
    }
}
