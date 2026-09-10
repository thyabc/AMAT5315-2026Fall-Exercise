use md::{Boundary, FluidConfig, PairPotential, System, VelocityVerlet, advance, initialize_fluid};

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 0.1, "{a} != {b}");
}

#[test]
fn periodic_geometry_handles_multiple_crossings_and_nearest_images() {
    let boundary = Boundary::periodic([10.0, 8.0]).unwrap();
    assert_eq!(boundary.wrap([-21.0, 25.0]), [9.0, 1.0]);
    assert_eq!(boundary.wrap([10.0, -8.0]), [0.0, 0.0]);
    assert_eq!(
        boundary.displacement([0.2, 1.0], [9.0, 1.0]),
        [1.1999999999999993, 0.0]
    );
    assert!(Boundary::periodic([0.0, 8.0]).is_err());
}

#[test]
fn shifted_cutoff_is_energy_continuous_and_uses_plain_force_inside() {
    let potential = PairPotential::shifted(2.5).unwrap();
    close(potential.energy(1.2), md::energy(1.2) - md::energy(2.5));
    close(potential.force(1.2), md::force(1.2));
    assert!(potential.energy(2.5 - 1e-8).abs() < 1e-8);
    for r in [2.5, 3.0] {
        assert_eq!(potential.energy(r), 0.0);
        assert_eq!(potential.force(r), 0.0);
    }
    let h = 1e-6;
    let numerical = -(potential.energy(1.3 + h) - potential.energy(1.3 - h)) / (2.0 * h);
    assert!((potential.force(1.3) - numerical).abs() < 1e-8);
    assert!(PairPotential::shifted(f64::NAN).is_err());
}

#[test]
fn periodic_pairs_share_geometry_for_energy_and_force() {
    let boundary = Boundary::periodic([10.0, 8.0]).unwrap();
    let potential = PairPotential::shifted(2.5).unwrap();
    let system = System::with_settings(
        vec![[0.2, 1.0], [9.0, 1.0]],
        vec![[0.0; 2]; 2],
        boundary,
        potential,
    )
    .unwrap();
    close(system.potential_energy(), potential.energy(1.2));
    close(system.accelerations()[0][0], potential.force(1.2));
    close(system.accelerations()[0][0], -system.accelerations()[1][0]);
    let translated = System::with_settings(
        vec![[20.2, -7.0], [9.0, 1.0]],
        vec![[0.0; 2]; 2],
        boundary,
        potential,
    )
    .unwrap();
    close(system.potential_energy(), translated.potential_energy());
    assert!(
        System::with_settings(vec![[0.0; 2]; 2], vec![[0.0; 2]; 2], boundary, potential).is_err()
    );
    assert!(
        System::with_settings(
            vec![[0.0; 2]],
            vec![[0.0; 2]],
            Boundary::periodic([4.0, 4.0]).unwrap(),
            potential
        )
        .is_err()
    );
}

#[test]
fn triangular_initialization_has_prescribed_density_temperature_and_seed() {
    let config = FluidConfig::default();
    let a = initialize_fluid(&config).unwrap();
    let b = initialize_fluid(&config).unwrap();
    assert_eq!(a.n_atoms(), 100);
    assert_eq!(a.positions(), b.positions());
    assert_eq!(a.velocities(), b.velocities());
    let lengths = a.boundary().lengths().unwrap();
    close(lengths[0] * lengths[1], 125.0);
    close(a.temperature(), 0.5);
    for axis in 0..2 {
        close(a.velocities().iter().map(|v| v[axis]).sum(), 0.0);
    }
    let spacing = (2.0 / (3.0_f64.sqrt() * 0.8)).sqrt();
    for (i, x) in a.positions().iter().enumerate() {
        let nearest = a
            .positions()
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, y)| {
                let d = a.boundary().displacement(*x, *y);
                d[0].hypot(d[1])
            })
            .fold(f64::INFINITY, f64::min);
        close(nearest, spacing);
        assert!(x[0] >= 0.0 && x[0] < lengths[0] && x[1] >= 0.0 && x[1] < lengths[1]);
    }
    let other = initialize_fluid(&FluidConfig {
        seed: 2027,
        ..config
    })
    .unwrap();
    assert_ne!(a.velocities(), other.velocities());
}

#[test]
fn periodic_free_flight_wraps_without_changing_velocity() {
    let mut system = System::with_settings(
        vec![[9.9, 0.1]],
        vec![[2.0, -2.0]],
        Boundary::periodic([10.0, 8.0]).unwrap(),
        PairPotential::shifted(2.5).unwrap(),
    )
    .unwrap();
    advance(&VelocityVerlet, &mut system, 0.1);
    close(system.positions()[0][0], 0.1);
    close(system.positions()[0][1], 7.9);
    assert_eq!(system.velocities(), &[[2.0, -2.0]]);
}

#[test]
fn phase_driver_thermostats_only_equilibration_and_samples_production() {
    use md::{Phase, simulate};
    let config = FluidConfig {
        equilibration_steps: 100,
        production_steps: 13,
        sample_every: 5,
        ..FluidConfig::default()
    };
    let mut equilibration = 0;
    let mut production = Vec::new();
    let mut frames = Vec::new();
    let mut production_start = None;
    simulate(&config, |phase, step, system, frame| {
        match phase {
            Phase::Equilibration => {
                equilibration += 1;
                close(system.temperature(), config.temperature);
                assert!(!frame);
            }
            Phase::Production => {
                if step == 0 {
                    production_start = Some(system.clone());
                }
                production.push(system.temperature());
                if frame {
                    frames.push(step);
                }
            }
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(equilibration, 100);
    assert_eq!(production.len(), 14);
    assert_eq!(frames, vec![0, 5, 10, 13]);
    let mut unthermostatted = production_start.unwrap();
    for temperature in production.iter().skip(1) {
        advance(&VelocityVerlet, &mut unthermostatted, config.dt);
        close(*temperature, unthermostatted.temperature());
    }
    assert!(
        production
            .iter()
            .any(|t| (t - config.temperature).abs() > 1e-4)
    );
}

#[test]
fn force_cache_is_refreshed_after_periodic_crossing() {
    let boundary = Boundary::periodic([10.0, 8.0]).unwrap();
    let potential = PairPotential::shifted(2.5).unwrap();
    let mut system = System::with_settings(
        vec![[9.999, 1.0], [1.2, 1.0]],
        vec![[1.0, 0.0], [-1.0, 0.0]],
        boundary,
        potential,
    )
    .unwrap();
    advance(&VelocityVerlet, &mut system, 0.005);
    assert!(system.positions()[0][0] < 1.0);
    let refreshed = System::with_settings(
        system.positions().to_vec(),
        system.velocities().to_vec(),
        boundary,
        potential,
    )
    .unwrap();
    assert_eq!(system.accelerations(), refreshed.accelerations());
    close(system.velocities().iter().map(|v| v[0]).sum(), 0.0);
}
