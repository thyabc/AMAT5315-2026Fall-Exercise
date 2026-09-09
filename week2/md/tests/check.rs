use md::{Observable, assess};

fn fixture() -> (Vec<Observable>, Vec<f64>) {
    let samples = (0..101).map(|step| Observable { step, time: step as f64 * 0.005,
        kinetic: 99.0, potential: -300.0, total: -201.0, temperature: 1.0 }).collect();
    // Deterministic quantiles of the exact reference CDF, not random data.
    let speeds = (0..1000).map(|i| {
        let p = (i as f64 + 0.5) / 1000.0;
        (-2.0 * 0.99 * (1.0 - p).ln()).sqrt()
    }).collect();
    (samples, speeds)
}

#[test]
fn checker_passes_reference_and_rejects_each_physical_failure() {
    let (samples, speeds) = fixture();
    assert!(assess(&samples, &speeds, 100, 1.0).unwrap().passed());
    let mut drifting = samples.clone();
    for s in &mut drifting { s.total += s.step as f64 * 0.02; s.potential = s.total - s.kinetic; }
    let report = assess(&drifting, &speeds, 100, 1.0).unwrap();
    assert!(!report.energy_pass);
    assert!((report.energy_slope - 4.0).abs() < 1e-10);
    let mut hot = samples.clone();
    for s in &mut hot { s.temperature = 1.2; s.kinetic = 118.8; s.total = s.kinetic + s.potential; }
    assert!(!assess(&hot, &speeds, 100, 1.0).unwrap().temperature_pass);
    assert!(!assess(&samples, &[1.0; 1000], 100, 1.0).unwrap().speed_pass);
}

#[test]
fn checker_rejects_nonfinite_empty_and_zero_kinetic_inputs() {
    let (mut samples, speeds) = fixture();
    assert!(assess(&[], &speeds, 100, 1.0).is_err());
    assert!(assess(&samples, &[], 100, 1.0).is_err());
    assert!(assess(&samples, &[f64::NAN], 100, 1.0).is_err());
    samples[0].kinetic = 0.0;
    assert!(assess(&samples, &speeds, 100, 1.0).is_err());
}

#[test]
#[ignore = "full 12000-step acceptance experiment; run with cargo test --release -- --ignored"]
fn prescribed_equilibrium_fluid_passes_all_checks() {
    let config = md::FluidConfig::default();
    let mut samples = Vec::new();
    let mut speeds = Vec::new();
    let mut frame_count = 0;
    md::simulate(&config, |phase, step, system, frame| {
        if phase == md::Phase::Production {
            samples.push(Observable::measure(step, config.dt, system));
            if frame {
                frame_count += 1;
                speeds.extend(system.velocities().iter().map(|v| v[0].hypot(v[1])));
            }
            for axis in 0..2 {
                assert!(system.velocities().iter().map(|v| v[axis]).sum::<f64>().abs() < 1e-9);
            }
        }
        Ok(())
    }).unwrap();
    assert_eq!(frame_count, 201);
    assert_eq!(samples.len(), 10001);
    let report = assess(&samples, &speeds, 100, 1.0).unwrap();
    println!("{report}");
    assert!(report.passed());
}
