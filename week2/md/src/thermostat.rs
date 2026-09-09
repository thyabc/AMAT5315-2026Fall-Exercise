use crate::System;

impl System {
    /// Reduced temperature with two center-of-mass degrees of freedom removed.
    pub fn temperature(&self) -> f64 {
        if self.n_atoms() < 2 {
            return f64::NAN;
        }
        self.kinetic_energy() / (self.n_atoms() - 1) as f64
    }

    pub fn rescale_temperature(&mut self, target: f64) -> Result<(), String> {
        let current = self.temperature();
        if !target.is_finite() || target <= 0.0 || !current.is_finite() || current <= 0.0 {
            return Err("temperature rescaling requires positive finite temperatures".into());
        }
        let scale = (target / current).sqrt();
        for v in &mut self.velocities {
            for component in v {
                *component *= scale;
            }
        }
        Ok(())
    }
}
