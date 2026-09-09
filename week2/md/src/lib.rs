/// Returns the program's greeting.
pub fn greeting() -> &'static str {
    "Hello, world!"
}

/// Lennard-Jones pair energy in reduced units, for separation `r > 0`.
pub fn energy(r: f64) -> f64 {
    let inv_r6 = r.recip().powi(6);
    4.0 * (inv_r6 * inv_r6 - inv_r6)
}

/// Radial pair force in reduced units, for separation `r > 0`.
/// Positive values indicate repulsion; negative values indicate attraction.
pub fn force(r: f64) -> f64 {
    let inv_r = r.recip();
    let inv_r6 = inv_r.powi(6);
    24.0 * inv_r * (2.0 * inv_r6 * inv_r6 - inv_r6)
}

#[cfg(test)]
mod tests {
    use super::{energy, force, greeting};

    #[test]
    fn greeting_is_hello_world() {
        assert_eq!(greeting(), "Hello, world!");
    }

    #[test]
    fn well_depth() {
        let r0 = 2.0_f64.powf(1.0 / 6.0);
        assert!((energy(r0) + 1.0).abs() < 1e-12);
        assert!(energy(1.0).abs() < 1e-12);
    }

    #[test]
    fn force_matches_negative_energy_derivative() {
        let r0 = 2.0_f64.powf(1.0 / 6.0);
        let h = 1e-5;
        for r in [0.95, 1.0, r0, 1.3, 2.0] {
            let actual = force(r);
            let numerical = -(energy(r + h) - energy(r - h)) / (2.0 * h);
            let tolerance = 1e-6 * actual.abs().max(1.0);
            assert!(
                (actual - numerical).abs() < tolerance,
                "r={r}: force={actual}, numerical derivative={numerical}, tolerance={tolerance}"
            );
        }
        assert!(force(1.0) > 0.0);
        assert!(force(r0).abs() < 1e-12);
        assert!(force(1.3) < 0.0);
    }
}
