use crate::Observable;
use std::fmt;

pub const ENERGY_LIMIT: f64 = 0.01;
pub const TEMPERATURE_LIMIT: f64 = 0.05;
pub const SPEED_LIMIT: f64 = 0.05;

#[derive(Debug)]
pub struct CheckReport {
    pub max_energy_change: f64,
    pub energy_slope: f64,
    pub mean_temperature: f64,
    pub relative_temperature_error: f64,
    pub block_temperatures: Vec<f64>,
    pub speed_discrepancy: f64,
    pub energy_pass: bool,
    pub temperature_pass: bool,
    pub speed_pass: bool,
}

impl CheckReport {
    pub fn passed(&self) -> bool {
        self.energy_pass && self.temperature_pass && self.speed_pass
    }
}

impl fmt::Display for CheckReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = |pass| if pass { "PASS" } else { "FAIL" };
        writeln!(
            f,
            "energy drift: {}  max |E-E0|/K0={:.8e} < {ENERGY_LIMIT}; slope={:.8e} energy/time",
            status(self.energy_pass),
            self.max_energy_change,
            self.energy_slope
        )?;
        writeln!(
            f,
            "temperature: {}  mean={:.8}; relative error={:.8} <= {TEMPERATURE_LIMIT}",
            status(self.temperature_pass),
            self.mean_temperature,
            self.relative_temperature_error
        )?;
        writeln!(f, "temperature block means: {:?}", self.block_temperatures)?;
        writeln!(
            f,
            "speed distribution: {}  max CDF discrepancy={:.8} < {SPEED_LIMIT} (correlated samples; no p-value)",
            status(self.speed_pass),
            self.speed_discrepancy
        )?;
        write!(f, "overall: {}", status(self.passed()))
    }
}

pub fn assess(
    samples: &[Observable],
    speeds: &[f64],
    particles: usize,
    target: f64,
) -> Result<CheckReport, String> {
    if samples.len() < 2
        || speeds.is_empty()
        || particles < 2
        || !target.is_finite()
        || target <= 0.0
    {
        return Err(
            "checks require measurements, speeds, N>=2, and a positive target temperature".into(),
        );
    }
    if samples.iter().any(|s| {
        ![s.time, s.kinetic, s.potential, s.total, s.temperature]
            .iter()
            .all(|x| x.is_finite())
            || s.kinetic <= 0.0
            || s.temperature <= 0.0
    }) || speeds.iter().any(|s| !s.is_finite() || *s < 0.0)
        || samples.windows(2).any(|s| s[1].time <= s[0].time)
    {
        return Err("invalid measurements for physical checks".into());
    }
    let count = samples.len() as f64;
    let initial = &samples[0];
    let max_energy_change = samples
        .iter()
        .map(|s| (s.total - initial.total).abs() / initial.kinetic)
        .fold(0.0, f64::max);
    let mean_temperature = samples.iter().map(|s| s.temperature).sum::<f64>() / count;
    let relative_temperature_error = (mean_temperature / target - 1.0).abs();
    let time_mean = samples.iter().map(|s| s.time).sum::<f64>() / count;
    let energy_mean = samples.iter().map(|s| s.total).sum::<f64>() / count;
    let energy_slope = samples
        .iter()
        .map(|s| (s.time - time_mean) * (s.total - energy_mean))
        .sum::<f64>()
        / samples
            .iter()
            .map(|s| (s.time - time_mean).powi(2))
            .sum::<f64>();
    let block_count = 10.min(samples.len());
    let block_temperatures = (0..block_count)
        .map(|i| {
            let block =
                &samples[i * samples.len() / block_count..(i + 1) * samples.len() / block_count];
            block.iter().map(|s| s.temperature).sum::<f64>() / block.len() as f64
        })
        .collect();
    // Removing COM motion gives each Cartesian component variance T*(N-1)/N.
    let variance = mean_temperature * (particles - 1) as f64 / particles as f64;
    let mut sorted = speeds.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut speed_discrepancy: f64 = 0.0;
    for (i, speed) in sorted.iter().enumerate() {
        let cdf = -(-speed.powi(2) / (2.0 * variance)).exp_m1();
        let lower = i as f64 / sorted.len() as f64;
        let upper = (i + 1) as f64 / sorted.len() as f64;
        speed_discrepancy = speed_discrepancy
            .max((cdf - lower).abs())
            .max((upper - cdf).abs());
    }
    Ok(CheckReport {
        max_energy_change,
        energy_slope,
        mean_temperature,
        relative_temperature_error,
        block_temperatures,
        speed_discrepancy,
        energy_pass: max_energy_change < ENERGY_LIMIT,
        temperature_pass: relative_temperature_error <= TEMPERATURE_LIMIT,
        speed_pass: speed_discrepancy < SPEED_LIMIT,
    })
}
