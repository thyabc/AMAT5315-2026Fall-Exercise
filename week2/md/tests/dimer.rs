use md::{EnergySample, Euler, VelocityVerlet, run_dimer};

fn maximum_error(samples: &[EnergySample]) -> f64 {
    for sample in samples {
        assert!(sample.relative_error.is_finite());
    }
    samples
        .iter()
        .map(|s| s.relative_error.abs())
        .fold(0.0, f64::max)
}

#[test]
fn dimer_integrators_share_the_same_experiment() {
    // run_dimer always starts at (0, 0), (1.2, 0), with zero velocities.
    // Only the integrator value changes between these two calls.
    let euler = run_dimer(&Euler, 0.01, 500);
    let verlet = run_dimer(&VelocityVerlet, 0.01, 500);
    for samples in [&euler, &verlet] {
        assert_eq!(samples.len(), 501);
        assert_eq!(samples[0].kinetic, 0.0);
        assert_eq!(samples[0].potential, md::energy(1.2));
        assert_eq!(samples[0].relative_error, 0.0);
        for (step, sample) in samples.iter().enumerate() {
            assert_eq!(sample.step, step);
            assert_eq!(sample.time, step as f64 * 0.01);
        }
        maximum_error(samples);
    }
    let verlet_max = maximum_error(&verlet);
    let euler_final = euler.last().unwrap().relative_error;
    println!("500 steps: Verlet maximum={verlet_max:e}, Euler final={euler_final:e}");
    assert!(verlet_max < 1e-3, "Verlet maximum={verlet_max:e}");
    assert!(euler_final > 0.5, "Euler final={euler_final:e}");
}

#[test]
fn verlet_energy_remains_bounded_for_5000_steps() {
    let samples = run_dimer(&VelocityVerlet, 0.01, 5000);
    assert_eq!(samples.len(), 5001);
    assert_eq!(samples.last().unwrap().time, 50.0);
    let maximum = maximum_error(&samples);
    println!("5000 steps: Verlet maximum={maximum:e}");
    assert!(maximum < 1e-3, "long-run Verlet maximum={maximum:e}");
}
