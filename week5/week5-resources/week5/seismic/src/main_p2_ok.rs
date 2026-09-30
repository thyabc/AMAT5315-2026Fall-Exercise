use anyhow::{bail, Context, Result};
use clap::Parser;
use ndarray::Array3;
use ndarray_npy::write_npy;
use serde::Deserialize;
use serde_json::{json, Value};

use std::fs::{self, File};
use std::path::{Path, PathBuf};


#[derive(Parser, Debug)]
#[command(name = "seismic")]
#[command(about = "Week 5 acoustic seismic simulator")]
struct Args {
    #[arg(long)]
    experiment: PathBuf,

    #[arg(long)]
    mode: String,

    #[arg(long)]
    every: Option<usize>,

    #[arg(long)]
    out: PathBuf,
}


#[derive(Debug, Deserialize)]
struct Experiment {
    nx: usize,
    nz: usize,

    dx: f64,
    dt: f64,
    steps: usize,

    source_frequency: f64,
    source_peak_time: f64,

    #[serde(default = "default_source_amplitude")]
    source_amplitude: f64,

    sponge_width: usize,
    sponge_strength: f64,

    shots: Vec<[f64; 2]>,
    receivers: Vec<[usize; 2]>,

    background: Vec<Vec<f64>>,
    perturbation: Vec<Vec<f64>>,

    length_unit_m: f64,
    time_unit_s: f64,
}


fn default_source_amplitude() -> f64 {
    1.0
}


struct ShotResult {
    traces: Vec<f64>,
    frames: Vec<f32>,
    frame_steps: Vec<usize>,
}


fn idx(x: usize, z: usize, nx: usize) -> usize {
    z * nx + x
}


fn flatten_field(
    field: &[Vec<f64>],
    nx: usize,
    nz: usize,
) -> Result<Vec<f64>> {

    if field.len() != nz {
        bail!(
            "field has {} rows, expected {}",
            field.len(),
            nz
        );
    }

    let mut out = Vec::with_capacity(nx * nz);

    for row in field {
        if row.len() != nx {
            bail!(
                "field row has {} entries, expected {}",
                row.len(),
                nx
            );
        }

        out.extend_from_slice(row);
    }

    Ok(out)
}


fn build_sponge(exp: &Experiment) -> Vec<f64> {

    let mut sigma = vec![0.0; exp.nx * exp.nz];

    let width = exp.sponge_width as f64;

    for z in 0..exp.nz {
        for x in 0..exp.nx {

            let distance = x
                .min(z)
                .min(exp.nx - 1 - x)
                .min(exp.nz - 1 - z) as f64;

            let factor = (1.0 - distance / width).max(0.0);

            sigma[idx(x, z, exp.nx)] =
                exp.sponge_strength * factor * factor;
        }
    }

    sigma
}


fn build_source_footprint(
    exp: &Experiment,
    shot: [f64; 2],
) -> Vec<f64> {

    let mut footprint =
        vec![0.0; exp.nx * exp.nz];

    let xs = shot[0];
    let zs = shot[1];

    for z in 0..exp.nz {
        for x in 0..exp.nx {

            let dx = x as f64 - xs;
            let dz = z as f64 - zs;

            footprint[idx(x, z, exp.nx)] =
                (-(dx * dx + dz * dz) / 2.0).exp();
        }
    }

    footprint
}


fn ricker(exp: &Experiment, step: usize) -> f64 {

    let t = step as f64 * exp.dt;

    let theta =
        std::f64::consts::PI
        * exp.source_frequency
        * (t - exp.source_peak_time);

    (1.0 - 2.0 * theta * theta)
        * (-theta * theta).exp()
}


fn run_shot(
    exp: &Experiment,
    velocity: &[f64],
    shot: [f64; 2],
    every: Option<usize>,
) -> ShotResult {

    let ncell = exp.nx * exp.nz;

    // u^{-1}
    let mut prev = vec![0.0_f64; ncell];

    // u^0
    let mut u = vec![0.0_f64; ncell];

    let sigma = build_sponge(exp);

    let footprint =
        build_source_footprint(exp, shot);

    let nr = exp.receivers.len();

    let mut traces =
        vec![0.0_f64; exp.steps * nr];

    let mut frames: Vec<f32> = Vec::new();
    let mut frame_steps = Vec::new();

    let dt2 = exp.dt * exp.dt;
    let dx2 = exp.dx * exp.dx;

    for step in 0..exp.steps {

        // Outer boundary stays exactly zero.
        let mut next =
            vec![0.0_f64; ncell];

        let pulse =
            exp.source_amplitude * ricker(exp, step);

        // Only update interior grid points.
        for z in 1..(exp.nz - 1) {
            for x in 1..(exp.nx - 1) {

                let i = idx(x, z, exp.nx);

                let lap = (
                    u[idx(x + 1, z, exp.nx)]
                    + u[idx(x - 1, z, exp.nx)]
                    + u[idx(x, z + 1, exp.nx)]
                    + u[idx(x, z - 1, exp.nx)]
                    - 4.0 * u[i]
                ) / dx2;

                let q =
                    pulse * footprint[i];

                let s = sigma[i];

                next[i] = (
                    2.0 * u[i]
                    - (1.0 - s * exp.dt) * prev[i]
                    + dt2
                        * (
                            velocity[i]
                                * velocity[i]
                                * lap
                            + q
                        )
                ) / (1.0 + s * exp.dt);
            }
        }

        // Receiver samples AFTER the update.
        for (r, receiver) in
            exp.receivers.iter().enumerate()
        {
            let rx = receiver[0];
            let rz = receiver[1];

            traces[step * nr + r] =
                next[idx(rx, rz, exp.nx)];
        }

        // Save u^(step+1)
        if let Some(interval) = every {
            if (step + 1) % interval == 0 {

                frame_steps.push(step + 1);

                frames.extend(
                    next.iter()
                        .map(|&v| v as f32)
                );
            }
        }

        prev = u;
        u = next;
    }

    ShotResult {
        traces,
        frames,
        frame_steps,
    }
}


fn l2_norm(values: &[f64]) -> f64 {
    values
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt()
}


fn write_json<P: AsRef<Path>>(
    path: P,
    value: &Value,
) -> Result<()> {

    let file = File::create(path)?;

    serde_json::to_writer_pretty(
        file,
        value,
    )?;

    Ok(())
}


fn main() -> Result<()> {

    let args = Args::parse();

    if args.mode != "forward" {
        bail!(
            "Part 2 currently implements only --mode forward"
        );
    }

    let text = fs::read_to_string(
        &args.experiment
    )
    .with_context(|| {
        format!(
            "failed to read {:?}",
            args.experiment
        )
    })?;

    let exp: Experiment =
        serde_json::from_str(&text)?;

    let raw_json: Value =
        serde_json::from_str(&text)?;

    fs::create_dir_all(&args.out)?;

    let background =
        flatten_field(
            &exp.background,
            exp.nx,
            exp.nz,
        )?;

    let perturbation =
        flatten_field(
            &exp.perturbation,
            exp.nx,
            exp.nz,
        )?;

    // CFL safety check.
    let max_c = background
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    let cfl =
        exp.dt * max_c / exp.dx
        * 2.0_f64.sqrt();

    if cfl >= 1.0 {
        bail!(
            "acoustic CFL condition violated: {}",
            cfl
        );
    }

    let nshots = exp.shots.len();
    let nr = exp.receivers.len();

    let mut all_traces =
        vec![
            0.0_f64;
            nshots * exp.steps * nr
        ];

    let mut first_wavefield: Vec<f32> =
        Vec::new();

    let mut recording_steps:
        Vec<usize> = Vec::new();

    println!(
        "shot\tmode\tdata L2 norm"
    );

    // =====================================================
    // Background forward simulation for all shots
    // =====================================================

    for (s, &shot) in
        exp.shots.iter().enumerate()
    {
        let recording =
            if s == 0 {
                args.every
            } else {
                None
            };

        let result =
            run_shot(
                &exp,
                &background,
                shot,
                recording,
            );

        let norm =
            l2_norm(&result.traces);

        println!(
            "{}\tforward\t{:.12}",
            s,
            norm
        );

        for step in 0..exp.steps {
            for r in 0..nr {

                let src =
                    step * nr + r;

                let dst =
                    (s * exp.steps + step)
                    * nr
                    + r;

                all_traces[dst] =
                    result.traces[src];
            }
        }

        if s == 0 {
            first_wavefield =
                result.frames;

            recording_steps =
                result.frame_steps;
        }
    }

    // =====================================================
    // Write receiver traces
    // shape [shot, step, receiver]
    // =====================================================

    let traces_array =
        Array3::from_shape_vec(
            (
                nshots,
                exp.steps,
                nr,
            ),
            all_traces,
        )?;

    write_npy(
        args.out.join("traces.npy"),
        &traces_array,
    )?;

    // =====================================================
    // Recording of first shot
    // =====================================================

    if args.every.is_some() {

        let nframes =
            recording_steps.len();

        let wavefield_array =
            Array3::from_shape_vec(
                (
                    nframes,
                    exp.nz,
                    exp.nx,
                ),
                first_wavefield.clone(),
            )?;

        write_npy(
            args.out.join(
                "wavefield.npy"
            ),
            &wavefield_array,
        )?;

        // -----------------------------------------------
        // Echo:
        // run first shot in background + perturbation
        // and subtract the background recording.
        // -----------------------------------------------

        let perturbed_velocity:
            Vec<f64> = background
            .iter()
            .zip(
                perturbation.iter()
            )
            .map(|(c, m)| c + m)
            .collect();

        let echo_run =
            run_shot(
                &exp,
                &perturbed_velocity,
                exp.shots[0],
                args.every,
            );

        if echo_run.frames.len()
            != first_wavefield.len()
        {
            bail!(
                "echo and background frame counts differ"
            );
        }

        let echo_values:
            Vec<f32> = echo_run
            .frames
            .iter()
            .zip(
                first_wavefield.iter()
            )
            .map(|(a, b)| a - b)
            .collect();

        let echo_array =
            Array3::from_shape_vec(
                (
                    nframes,
                    exp.nz,
                    exp.nx,
                ),
                echo_values,
            )?;

        write_npy(
            args.out.join("echo.npy"),
            &echo_array,
        )?;
    }

    // =====================================================
    // result.json
    // =====================================================

    let result_json = json!({
        "mode": "forward",
        "nx": exp.nx,
        "nz": exp.nz,
        "dx": exp.dx,
        "dt": exp.dt,
        "steps": exp.steps,
        "shots": exp.shots,
        "receivers": exp.receivers
    });

    write_json(
        args.out.join("result.json"),
        &result_json,
    )?;

    // =====================================================
    // run.json
    // =====================================================

    let mut experiment_metadata =
        raw_json.clone();

    if let Some(obj) =
        experiment_metadata.as_object_mut()
    {
        obj.remove("background");
        obj.remove("perturbation");
    }

    let times: Vec<f64> =
        recording_steps
        .iter()
        .map(|&s| s as f64 * exp.dt)
        .collect();

    let run_json = json!({
        "experiment_file":
            args.experiment.to_string_lossy(),
        "experiment":
            experiment_metadata,
        "recording": {
            "every": args.every,
            "steps": recording_steps,
            "times": times
        }
    });

    write_json(
        args.out.join("run.json"),
        &run_json,
    )?;

    // Overall trace norm as an additional check.
    let overall_norm = traces_array
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();

    println!();
    println!(
        "all traces L2 norm = {:.12}",
        overall_norm
    );

    println!(
        "output = {}",
        args.out.display()
    );

    println!(
        "length unit = {} m, time unit = {} s",
        exp.length_unit_m,
        exp.time_unit_s
    );

    Ok(())
}
