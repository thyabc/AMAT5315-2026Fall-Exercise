/// Plain LJ or energy-shifted LJ. The force is not shifted.
#[derive(Clone, Copy, Debug)]
pub struct PairPotential {
    cutoff: Option<f64>,
    shift: f64,
}

impl PairPotential {
    pub const PLAIN: Self = Self {
        cutoff: None,
        shift: 0.0,
    };

    pub fn shifted(cutoff: f64) -> Result<Self, String> {
        if !cutoff.is_finite() || cutoff <= 0.0 || !crate::energy(cutoff).is_finite() {
            return Err("cutoff must be finite and positive with finite LJ energy".into());
        }
        Ok(Self {
            cutoff: Some(cutoff),
            shift: crate::energy(cutoff),
        })
    }

    pub fn cutoff(self) -> Option<f64> {
        self.cutoff
    }
    pub fn energy(self, r: f64) -> f64 {
        if self.cutoff.is_some_and(|rc| r >= rc) {
            0.0
        } else {
            crate::energy(r) - self.shift
        }
    }
    pub fn force(self, r: f64) -> f64 {
        if self.cutoff.is_some_and(|rc| r >= rc) {
            0.0
        } else {
            crate::force(r)
        }
    }
}
