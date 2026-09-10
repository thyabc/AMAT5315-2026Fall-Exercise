use crate::{Boundary, FluidConfig, PairPotential, Phase, System, simulate};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

const OBS_HEADER: &str = "step,time,kinetic,potential,total,temperature";
const FRAME_HEADER: &str = "step,id,x,y,vx,vy";

#[derive(Clone, Debug)]
pub struct Observable {
    pub step: usize,
    pub time: f64,
    pub kinetic: f64,
    pub potential: f64,
    pub total: f64,
    pub temperature: f64,
}

impl Observable {
    pub fn measure(step: usize, dt: f64, system: &System) -> Self {
        let kinetic = system.kinetic_energy();
        let potential = system.potential_energy();
        Self {
            step,
            time: step as f64 * dt,
            kinetic,
            potential,
            total: kinetic + potential,
            temperature: system.temperature(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metadata {
    pub format_version: u32,
    pub complete: bool,
    pub config: FluidConfig,
    pub box_lengths: [f64; 2],
    pub dimensions: usize,
    pub mass: f64,
    pub units: String,
    pub potential: String,
    pub thermostat: String,
    pub rng: String,
}

pub struct RecordedRun {
    pub metadata: Metadata,
    pub observables: Vec<Observable>,
    pub speeds: Vec<f64>,
}

fn io_error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// Metadata is marked complete only after both streamed CSV files are flushed.
pub fn record_run(config: &FluidConfig, output: &Path, overwrite: bool) -> Result<(), String> {
    config.validate()?;
    if output.exists() && !overwrite {
        return Err(format!(
            "{} already exists; choose another output or use --overwrite",
            output.display()
        ));
    }
    fs::create_dir_all(output).map_err(io_error)?;
    let mut metadata = Metadata {
        format_version: 1,
        complete: false,
        config: config.clone(),
        box_lengths: config.box_lengths()?,
        dimensions: 2,
        mass: 1.0,
        units: "LJ reduced: sigma=epsilon=mass=kB=1".into(),
        potential: "energy-shifted LJ".into(),
        thermostat: "velocity-rescaling; equilibration only".into(),
        rng: "ChaCha8Rng / rand_distr StandardNormal; see Cargo.lock".into(),
    };
    let metadata_path = output.join("metadata.json");
    fs::write(
        &metadata_path,
        serde_json::to_vec_pretty(&metadata).map_err(io_error)?,
    )
    .map_err(io_error)?;
    let mut observables =
        BufWriter::new(File::create(output.join("observables.csv")).map_err(io_error)?);
    let mut trajectory =
        BufWriter::new(File::create(output.join("trajectory.csv")).map_err(io_error)?);
    writeln!(observables, "{OBS_HEADER}").map_err(io_error)?;
    writeln!(trajectory, "{FRAME_HEADER}").map_err(io_error)?;
    simulate(config, |phase, step, system, frame| {
        if phase != Phase::Production {
            return Ok(());
        }
        let s = Observable::measure(step, config.dt, system);
        if ![s.kinetic, s.potential, s.total, s.temperature]
            .iter()
            .all(|x| x.is_finite())
        {
            return Err(format!("nonfinite measurement at production step {step}"));
        }
        writeln!(
            observables,
            "{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
            s.step, s.time, s.kinetic, s.potential, s.total, s.temperature
        )
        .map_err(io_error)?;
        if frame {
            for (id, (x, v)) in system
                .positions()
                .iter()
                .zip(system.velocities())
                .enumerate()
            {
                writeln!(
                    trajectory,
                    "{step},{id},{:.17e},{:.17e},{:.17e},{:.17e}",
                    x[0], x[1], v[0], v[1]
                )
                .map_err(io_error)?;
            }
        }
        Ok(())
    })?;
    observables.flush().map_err(io_error)?;
    trajectory.flush().map_err(io_error)?;
    metadata.complete = true;
    fs::write(
        metadata_path,
        serde_json::to_vec_pretty(&metadata).map_err(io_error)?,
    )
    .map_err(io_error)
}

fn read_csv(
    path: &Path,
    header: &str,
) -> Result<std::io::Lines<BufReader<File>>, String> {
    let mut lines =
        BufReader::new(File::open(path).map_err(|e| format!("{}: {e}", path.display()))?).lines();
    if lines.next().transpose().map_err(io_error)?.as_deref() != Some(header) {
        return Err(format!("{}: invalid CSV header", path.display()));
    }
    Ok(lines)
}

fn number(s: &str) -> Result<f64, String> {
    let value = s.parse::<f64>().map_err(io_error)?;
    if !value.is_finite() {
        return Err("nonfinite value in recording".into());
    }
    Ok(value)
}

fn integer(s: &str) -> Result<usize, String> {
    s.parse().map_err(io_error)
}

fn consistent(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

/// Validate schema, completeness, ordering, and frame/observable physical consistency.
pub fn read_run(input: &Path) -> Result<RecordedRun, String> {
    let metadata: Metadata =
        serde_json::from_reader(File::open(input.join("metadata.json")).map_err(io_error)?)
            .map_err(io_error)?;
    metadata.config.validate()?;
    if metadata.format_version != 1
        || !metadata.complete
        || metadata.dimensions != 2
        || metadata.mass != 1.0
        || metadata.potential != "energy-shifted LJ"
    {
        return Err("unsupported or incomplete run metadata".into());
    }
    let config = &metadata.config;
    let lengths = config.box_lengths()?;
    if !lengths
        .iter()
        .zip(metadata.box_lengths)
        .all(|(a, b)| b.is_finite() && consistent(*a, b))
    {
        return Err("box lengths do not match density and particle count".into());
    }
    let mut observables = Vec::new();
    for (step, line) in read_csv(&input.join("observables.csv"), OBS_HEADER)?.enumerate() {
        let line = line.map_err(io_error)?;
        let f: Vec<_> = line.split(',').collect();
        if f.len() != 6 || integer(f[0])? != step || step > config.production_steps {
            return Err("invalid observable row count or step ordering".into());
        }
        let s = Observable {
            step,
            time: number(f[1])?,
            kinetic: number(f[2])?,
            potential: number(f[3])?,
            total: number(f[4])?,
            temperature: number(f[5])?,
        };
        if s.kinetic <= 0.0
            || !consistent(s.time, step as f64 * config.dt)
            || !consistent(s.total, s.kinetic + s.potential)
            || !consistent(s.temperature, s.kinetic / (config.particles - 1) as f64)
        {
            return Err(format!("inconsistent observable at step {step}"));
        }
        observables.push(s);
    }
    if observables.len() != config.production_steps + 1 {
        return Err("incomplete observables".into());
    }
    let trajectory_path = input.join("trajectory.csv");
    let mut lines = read_csv(&trajectory_path, FRAME_HEADER)?;
    let mut speeds = Vec::new();
    for step in (0..=config.production_steps)
        .filter(|s| s % config.sample_every == 0 || *s == config.production_steps)
    {
        let mut positions = Vec::with_capacity(config.particles);
        let mut velocities = Vec::with_capacity(config.particles);
        for id in 0..config.particles {
            let line = lines
                .next()
                .ok_or("incomplete trajectory")?
                .map_err(io_error)?;
            let f: Vec<_> = line.split(',').collect();
            if f.len() != 6 || integer(f[0])? != step || integer(f[1])? != id {
                return Err("invalid trajectory step or particle ordering".into());
            }
            let x = [number(f[2])?, number(f[3])?];
            let v = [number(f[4])?, number(f[5])?];
            if (0..2).any(|axis| x[axis] < 0.0 || x[axis] >= lengths[axis]) {
                return Err("trajectory position outside periodic box".into());
            }
            speeds.push(v[0].hypot(v[1]));
            positions.push(x);
            velocities.push(v);
        }
        let system = System::with_settings(
            positions,
            velocities,
            Boundary::periodic(lengths)?,
            PairPotential::shifted(config.cutoff)?,
        )?;
        let sample = &observables[step];
        if !consistent(system.kinetic_energy(), sample.kinetic)
            || !consistent(system.potential_energy(), sample.potential)
        {
            return Err(format!(
                "trajectory and observables disagree at step {step}"
            ));
        }
        for axis in 0..2 {
            if system
                .velocities()
                .iter()
                .map(|v| v[axis])
                .sum::<f64>()
                .abs()
                > 1e-8 * config.particles as f64
            {
                return Err("trajectory has nonzero center-of-mass momentum".into());
            }
        }
    }
    if lines.next().is_some() {
        return Err("extra trajectory rows".into());
    }
    Ok(RecordedRun {
        metadata,
        observables,
        speeds,
    })
}
