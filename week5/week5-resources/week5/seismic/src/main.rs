mod treeverse;

use treeverse::make_treeverse_plan;
use std::collections::HashMap;
use anyhow::{bail, Context, Result};
use clap::Parser;
use ndarray::{Array2, Array3};
use ndarray_npy::{read_npy, write_npy};
use serde::Deserialize;
use serde_json::{json, Value};

use std::fs::{self, File};
use std::path::{Path, PathBuf};
unsafe extern "C" {
fn enzyme_timestep_primal(
    state_ptr: *const f64,
    c_ptr: *const f64,
    sigma_ptr: *const f64,
    source_ptr: *const f64,

    nx: usize,
    nz: usize,
    dx: f64,
    dt: f64,

    out_ptr: *mut f64,
);
fn enzyme_timestep_reverse(
    state_ptr: *const f64,
    astate_ptr: *mut f64,

    c_ptr: *const f64,
    ac_ptr: *mut f64,

    sigma_ptr: *const f64,
    source_ptr: *const f64,

    nx: usize,
    nz: usize,
    dx: f64,
    dt: f64,

    out_ptr: *mut f64,
    aout_ptr: *mut f64,
);
    fn enzyme_timestep_forward(
        state_ptr: *const f64,
        dstate_ptr: *const f64,

        c_ptr: *const f64,
        dc_ptr: *const f64,

        sigma_ptr: *const f64,
        source_ptr: *const f64,

        nx: usize,
        nz: usize,
        dx: f64,
        dt: f64,

        out_ptr: *mut f64,
        dout_ptr: *mut f64,
    );
}

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
#[arg(long)]
data: Option<PathBuf>,
#[arg(
    long,
    default_value = "full"
)]
storage: String,

#[arg(long)]
checkpoints: Option<usize>,
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
fn run_born_mode(
    args: &Args,
    exp: &Experiment,
    raw_json: &Value,
    background: &[f64],
    perturbation: &[f64],
) -> Result<()> {

    let ncell = exp.nx * exp.nz;
    let nshots = exp.shots.len();
    let nr = exp.receivers.len();

    let sigma = build_sponge(exp);

    let mut all_born =
        vec![
            0.0_f64;
            nshots * exp.steps * nr
        ];

    println!(
        "shot\tmode\tdata L2 norm"
    );

    for (shot_index, &shot) in
        exp.shots.iter().enumerate()
    {
        // =================================================
        // Initial primal state:
        // s0 = (u^-1, u^0) = 0
        // =================================================

        let mut state =
            vec![0.0_f64; 2 * ncell];

        // Initial tangent state is fixed:
        // ds0 = 0
        let mut dstate =
            vec![0.0_f64; 2 * ncell];

        let footprint =
            build_source_footprint(
                exp,
                shot
            );

        let mut shot_sum_sq =
            0.0_f64;

        // =================================================
        // Chain one Enzyme JVP per timestep
        // =================================================

        for step in 0..exp.steps {

            let pulse =
                exp.source_amplitude
                * ricker(exp, step);

            // Source is held constant with respect
            // to differentiation.
            let mut source =
                vec![0.0_f64; ncell];

            for i in 0..ncell {
                source[i] =
                    pulse * footprint[i];
            }

            let mut next =
                vec![0.0_f64; 2 * ncell];

            let mut dnext =
                vec![0.0_f64; 2 * ncell];

            unsafe {
                enzyme_timestep_forward(
                    state.as_ptr(),
                    dstate.as_ptr(),

                    background.as_ptr(),

                    // dc = m:
                    // experiment velocity perturbation
                    perturbation.as_ptr(),

                    sigma.as_ptr(),
                    source.as_ptr(),

                    exp.nx,
                    exp.nz,
                    exp.dx,
                    exp.dt,

                    next.as_mut_ptr(),
                    dnext.as_mut_ptr(),
                );
            }

            // =================================================
            // Receiver samples tangent of u^(n+1)
            // =================================================

            for (r, receiver) in
                exp.receivers.iter().enumerate()
            {
                let rx = receiver[0];
                let rz = receiver[1];

                let value =
                    dnext[
                        ncell
                        + idx(
                            rx,
                            rz,
                            exp.nx
                        )
                    ];

                let dst =
                    (shot_index * exp.steps + step)
                    * nr
                    + r;

                all_born[dst] = value;

                shot_sum_sq +=
                    value * value;
            }

            state = next;
            dstate = dnext;
        }

        println!(
            "{}\tborn\t{:.12}",
            shot_index,
            shot_sum_sq.sqrt()
        );
    }

    // =====================================================
    // Write Born data
    // [shot, step, receiver]
    // =====================================================

    let born_array =
        Array3::from_shape_vec(
            (
                nshots,
                exp.steps,
                nr,
            ),
            all_born,
        )?;

    write_npy(
        args.out.join(
            "born_data.npy"
        ),
        &born_array,
    )?;

    let born_norm_sq =
        born_array
            .iter()
            .map(|x| x * x)
            .sum::<f64>();

    println!();

    println!(
        "Born data squared L2 norm = {:.12}",
        born_norm_sq
    );

    // =====================================================
    // result.json
    // =====================================================

    let result_json = json!({
        "mode": "born",
        "nx": exp.nx,
        "nz": exp.nz,
        "dx": exp.dx,
        "dt": exp.dt,
        "steps": exp.steps,
        "shots": exp.shots,
        "receivers": exp.receivers
    });

    write_json(
        args.out.join(
            "result.json"
        ),
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

    let run_json = json!({
        "experiment_file":
            args.experiment
                .to_string_lossy(),
        "experiment":
            experiment_metadata
    });

    write_json(
        args.out.join(
            "run.json"
        ),
        &run_json,
    )?;

    Ok(())
}
fn run_adjoint_mode(
    args: &Args,
    exp: &Experiment,
    raw_json: &Value,
    background: &[f64],
    perturbation: &[f64],
) -> Result<()> {

    let data_path = args
        .data
        .as_ref()
        .context(
            "--mode adjoint requires --data"
        )?;

    // =====================================================
    // Read receiver weights
    // =====================================================

    let weights: Array3<f64> =
        read_npy(data_path)
            .with_context(|| {
                format!(
                    "failed to read {:?}",
                    data_path
                )
            })?;

    let expected_shape = (
        exp.shots.len(),
        exp.steps,
        exp.receivers.len(),
    );

    if weights.dim() != expected_shape {
        bail!(
            "adjoint data has shape {:?}, expected {:?}",
            weights.dim(),
            expected_shape
        );
    }

    let ncell = exp.nx * exp.nz;
    let nshots = exp.shots.len();

    let sigma = build_sponge(exp);

    // Sum B_n^T lambda over all times and shots.
    let mut image =
        vec![0.0_f64; ncell];

    // Recording: first shot only.
    let mut first_frames:
        Vec<f32> = Vec::new();

    let mut recording_steps:
        Vec<usize> = Vec::new();

    // =====================================================
    // Statistics required by the assignment
    // =====================================================

    let mut reverse_calls_total =
        0usize;

    let mut scheduler_forward_calls_total =
        0usize;

    let peak_saved_states =
        exp.steps + 1;

    let bytes_per_state =
        2 * ncell
        * std::mem::size_of::<f64>();

    let peak_saved_bytes =
        peak_saved_states
        * bytes_per_state;

    let mut per_shot_statistics:
        Vec<Value> = Vec::new();

    println!(
        "shot\tforward states\treverse VJPs"
    );

    // =====================================================
    // Each shot independently
    // =====================================================

    for (shot_index, &shot) in
        exp.shots.iter().enumerate()
    {
        let footprint =
            build_source_footprint(
                exp,
                shot
            );

        // -------------------------------------------------
        // 1. Full forward trajectory:
        //
        // trajectory[n] = s_n
        //
        // s_n = [u^(n-1), u^n]
        // -------------------------------------------------

        let mut trajectory:
            Vec<Vec<f64>> =
            Vec::with_capacity(
                exp.steps + 1
            );

        let mut state =
            vec![0.0_f64; 2 * ncell];

        trajectory.push(
            state.clone()
        );

        for step in 0..exp.steps {

            let pulse =
                exp.source_amplitude
                * ricker(exp, step);

            let mut source =
                vec![0.0_f64; ncell];

            for i in 0..ncell {
                source[i] =
                    pulse * footprint[i];
            }

            let mut next =
                vec![0.0_f64; 2 * ncell];

            unsafe {
                enzyme_timestep_primal(
                    state.as_ptr(),
                    background.as_ptr(),
                    sigma.as_ptr(),
                    source.as_ptr(),

                    exp.nx,
                    exp.nz,
                    exp.dx,
                    exp.dt,

                    next.as_mut_ptr(),
                );
            }

            state = next;

            trajectory.push(
                state.clone()
            );
        }

        scheduler_forward_calls_total +=
            exp.steps;

        // -------------------------------------------------
        // 2. Reverse sweep
        //
        // adj_state initially corresponds to s_N
        // -------------------------------------------------

        let mut adj_state =
            vec![0.0_f64; 2 * ncell];

        let mut shot_reverse_calls =
            0usize;

        for step in
            (0..exp.steps).rev()
        {
            // ---------------------------------------------
            // Receiver transpose:
            //
            // data at index step was sampled from
            // u^(step+1), i.e. second half of s_(step+1)
            // ---------------------------------------------

            for (r, receiver) in
                exp.receivers
                    .iter()
                    .enumerate()
            {
                let rx = receiver[0];
                let rz = receiver[1];

                let grid_index =
                    idx(
                        rx,
                        rz,
                        exp.nx
                    );

                adj_state[
                    ncell + grid_index
                ] +=
                    weights[[
                        shot_index,
                        step,
                        r
                    ]];
            }

            // Reconstruct the fixed source q_n.
            let pulse =
                exp.source_amplitude
                * ricker(exp, step);

            let mut source =
                vec![0.0_f64; ncell];

            for i in 0..ncell {
                source[i] =
                    pulse * footprint[i];
            }

            // Input adjoint produced by this VJP.
            let mut astate =
                vec![0.0_f64; 2 * ncell];

            // Velocity adjoint produced by this VJP.
            let mut ac =
                vec![0.0_f64; ncell];

            // Enzyme may recompute the primal output
            // through this buffer.
            let mut reverse_out =
                vec![0.0_f64; 2 * ncell];

            // Output adjoint seed for H_step.
            let mut aout =
                adj_state.clone();

            unsafe {
                enzyme_timestep_reverse(
                    trajectory[step]
                        .as_ptr(),

                    astate.as_mut_ptr(),

                    background.as_ptr(),
                    ac.as_mut_ptr(),

                    sigma.as_ptr(),
                    source.as_ptr(),

                    exp.nx,
                    exp.nz,
                    exp.dx,
                    exp.dt,

                    reverse_out
                        .as_mut_ptr(),

                    aout.as_mut_ptr(),
                );
            }

            // ---------------------------------------------
            // Sum velocity sensitivity
            // ---------------------------------------------

            for i in 0..ncell {
                image[i] += ac[i];
            }

            // Now this is the adjoint of s_step.
            adj_state = astate;

            shot_reverse_calls += 1;
            reverse_calls_total += 1;

            // ---------------------------------------------
            // First-shot adjoint recording.
            //
            // After reversing step n, adj_state is
            // the adjoint of s_n.
            // Its second half is adjoint(u^n).
            // ---------------------------------------------

            if shot_index == 0 {
                if let Some(every) =
                    args.every
                {
                    if step % every == 0 {

                        recording_steps
                            .push(step);

                        first_frames.extend(
                            adj_state[
                                ncell..
                                2 * ncell
                            ]
                            .iter()
                            .map(
                                |&v| v as f32
                            )
                        );
                    }
                }
            }
        }

        println!(
            "{}\t{}\t{}",
            shot_index,
            exp.steps,
            shot_reverse_calls
        );

        per_shot_statistics.push(
            json!({
                "reverse_calls":
                    shot_reverse_calls,

                "scheduler_forward_calls":
                    exp.steps,

                "peak_saved_states":
                    peak_saved_states
            })
        );
    }

    // =====================================================
    // image.npy : [z, x], float64
    // =====================================================

    let image_array =
        Array2::from_shape_vec(
            (
                exp.nz,
                exp.nx
            ),
            image.clone(),
        )?;

    write_npy(
        args.out.join(
            "image.npy"
        ),
        &image_array,
    )?;

    // =====================================================
    // Optional adjoint wavefield recording
    // =====================================================

    if args.every.is_some() {

        let nframes =
            recording_steps.len();

        let wavefield =
            Array3::from_shape_vec(
                (
                    nframes,
                    exp.nz,
                    exp.nx,
                ),
                first_frames,
            )?;

        write_npy(
            args.out.join(
                "wavefield.npy"
            ),
            &wavefield,
        )?;
    }

    // =====================================================
    // Transpose identity
    // =====================================================

    let left =
        weights
            .iter()
            .map(|v| v * v)
            .sum::<f64>();

    let right =
        perturbation
            .iter()
            .zip(image.iter())
            .map(|(m, g)| m * g)
            .sum::<f64>();

    let denominator =
        left.abs()
            .max(right.abs())
            .max(1.0e-30);

    let relative_difference =
        (left - right).abs()
        / denominator;

    println!();
    println!(
        "transpose identity"
    );

    println!(
        "left  = {:.15e}",
        left
    );

    println!(
        "right = {:.15e}",
        right
    );

    println!(
        "relative difference = {:.3e}",
        relative_difference
    );

    // =====================================================
    // Statistics
    // =====================================================

    let statistics = json!({
        "storage": "full",
        "checkpoints": Value::Null,

        "reverse_calls":
            reverse_calls_total,

        "scheduler_forward_calls":
            scheduler_forward_calls_total,

        "peak_saved_states":
            peak_saved_states,

        "peak_saved_bytes":
            peak_saved_bytes,

        "per_shot":
            per_shot_statistics
    });

    let result_json = json!({
        "mode": "adjoint",

        "nx": exp.nx,
        "nz": exp.nz,

        "dx": exp.dx,
        "dt": exp.dt,

        "steps": exp.steps,

        "shots": exp.shots,
        "receivers": exp.receivers,

        "statistics": statistics,

        "transpose_identity": {
            "left": left,
            "right": right,
            "relative_difference":
                relative_difference
        }
    });

    write_json(
        args.out.join(
            "result.json"
        ),
        &result_json,
    )?;

    // =====================================================
    // run.json
    // =====================================================

    let mut experiment_metadata =
        raw_json.clone();

    if let Some(obj) =
        experiment_metadata
            .as_object_mut()
    {
        obj.remove("background");
        obj.remove("perturbation");
    }

    let times:
        Vec<f64> =
        recording_steps
            .iter()
            .map(
                |&step| {
                    step as f64
                    * exp.dt
                }
            )
            .collect();

    let run_json = json!({
        "experiment_file":
            args.experiment
                .to_string_lossy(),

        "experiment":
            experiment_metadata,

        "recording": {
            "every": args.every,
            "steps": recording_steps,
            "times": times
        }
    });

    write_json(
        args.out.join(
            "run.json"
        ),
        &run_json,
    )?;

    Ok(())
}
fn run_adjoint_treeverse_mode(
    args: &Args,
    exp: &Experiment,
    raw_json: &Value,
    background: &[f64],
    perturbation: &[f64],
) -> Result<()> {

    let checkpoints =
        args.checkpoints.context(
            "--storage treeverse requires --checkpoints"
        )?;

    if checkpoints == 0 {
        bail!(
            "--checkpoints must be at least 1"
        );
    }

    let data_path =
        args.data.as_ref().context(
            "--mode adjoint requires --data"
        )?;

    let weights: Array3<f64> =
        read_npy(data_path)?;

    let expected_shape = (
        exp.shots.len(),
        exp.steps,
        exp.receivers.len(),
    );

    if weights.dim() != expected_shape {
        bail!(
            "adjoint data shape {:?}, expected {:?}",
            weights.dim(),
            expected_shape
        );
    }

    let ncell =
        exp.nx * exp.nz;

    let bytes_per_state =
        2 * ncell
        * std::mem::size_of::<f64>();

    let sigma =
        build_sponge(exp);

    let mut image =
        vec![0.0_f64; ncell];

    let mut reverse_calls_total =
        0usize;

    let mut scheduler_forward_calls_total =
        0usize;

    let mut overall_peak_saved_states =
        1usize;

    let mut per_shot_statistics:
        Vec<Value> = Vec::new();

    // ======================================================
    // Each shot
    // ======================================================

    for (shot_index, &shot) in
        exp.shots.iter().enumerate()
    {
        println!(
            "treeverse shot {}",
            shot_index
        );

        let plan =
            make_treeverse_plan(
                exp.steps,
                checkpoints,
            )?;

        let footprint =
            build_source_footprint(
                exp,
                shot
            );

        // --------------------------------------------------
        // Saved checkpoint slots.
        //
        // s0 is permanently saved.
        // --------------------------------------------------

        let s0 =
            vec![0.0_f64; 2 * ncell];

        let mut saved:
            HashMap<usize, Vec<f64>> =
            HashMap::new();

        saved.insert(
            0,
            s0.clone()
        );

        // Replay working state is not counted
        // as a saved checkpoint slot.
        let mut working_state =
            s0;

        let mut working_step =
            0usize;

        // Reverse state:
        // initially adjoint of s_N.
        let mut adj_state =
            vec![0.0_f64; 2 * ncell];

        let mut reverse_calls_shot =
            0usize;

        let mut source =
            vec![0.0_f64; ncell];

        // ==================================================
        // Execute Treeverse actions
        // ==================================================

        for action in &plan.actions {

            match action.action.as_str() {

                // ==========================================
                // restore
                // ==========================================

                "restore" => {

                    let checkpoint =
                        saved.get(
                            &action.step
                        )
                        .with_context(|| {
                            format!(
                                "restore requested missing state s_{}",
                                action.step
                            )
                        })?;

                    working_state
                        .clone_from(
                            checkpoint
                        );

                    working_step =
                        action.step;
                }


                // ==========================================
                // call: s_n -> s_(n+1)
                // ==========================================

                "call" => {

                    if working_step
                        != action.step
                    {
                        bail!(
                            "call step mismatch: working s_{}, action asks step {}",
                            working_step,
                            action.step
                        );
                    }

                    let pulse =
                        exp.source_amplitude
                        * ricker(
                            exp,
                            action.step
                        );

                    for i in 0..ncell {
                        source[i] =
                            pulse
                            * footprint[i];
                    }

                    let mut next =
                        vec![
                            0.0_f64;
                            2 * ncell
                        ];

                    unsafe {
                        enzyme_timestep_primal(
                            working_state
                                .as_ptr(),

                            background
                                .as_ptr(),

                            sigma
                                .as_ptr(),

                            source
                                .as_ptr(),

                            exp.nx,
                            exp.nz,
                            exp.dx,
                            exp.dt,

                            next
                                .as_mut_ptr(),
                        );
                    }

                    working_state =
                        next;

                    working_step += 1;
                }


                // ==========================================
                // store
                // ==========================================

                "store" => {

                    if working_step
                        != action.step
                    {
                        bail!(
                            "store mismatch: working s_{}, expected s_{}",
                            working_step,
                            action.step
                        );
                    }

                    if saved.contains_key(
                        &action.step
                    ) {
                        bail!(
                            "state s_{} already saved",
                            action.step
                        );
                    }

                    saved.insert(
                        action.step,
                        working_state.clone(),
                    );
                }


                // ==========================================
                // grad
                // ==========================================

                "grad" => {

                    let step =
                        action.step;

                    // Treeverse specification:
                    // grad must read s_n from a saved slot.
                    let primal_state =
                        saved.get(&step)
                        .with_context(|| {
                            format!(
                                "grad step {} has no saved primal state",
                                step
                            )
                        })?;

                    // --------------------------------------
                    // Receiver transpose R^T w_n.
                    //
                    // Receiver data after step n belongs to
                    // u^(n+1), the second half of s_(n+1).
                    // --------------------------------------

                    for (
                        receiver_index,
                        receiver
                    ) in exp.receivers
                        .iter()
                        .enumerate()
                    {
                        let rx =
                            receiver[0];

                        let rz =
                            receiver[1];

                        let gi =
                            idx(
                                rx,
                                rz,
                                exp.nx
                            );

                        adj_state[
                            ncell + gi
                        ] += weights[[
                            shot_index,
                            step,
                            receiver_index
                        ]];
                    }

                    // Source q_n is fixed.
                    let pulse =
                        exp.source_amplitude
                        * ricker(
                            exp,
                            step
                        );

                    for i in 0..ncell {
                        source[i] =
                            pulse
                            * footprint[i];
                    }

                    let mut astate =
                        vec![
                            0.0_f64;
                            2 * ncell
                        ];

                    let mut ac =
                        vec![
                            0.0_f64;
                            ncell
                        ];

                    let mut reverse_out =
                        vec![
                            0.0_f64;
                            2 * ncell
                        ];

                    let mut aout =
                        adj_state.clone();

                    unsafe {
                        enzyme_timestep_reverse(
                            primal_state
                                .as_ptr(),

                            astate
                                .as_mut_ptr(),

                            background
                                .as_ptr(),

                            ac
                                .as_mut_ptr(),

                            sigma
                                .as_ptr(),

                            source
                                .as_ptr(),

                            exp.nx,
                            exp.nz,
                            exp.dx,
                            exp.dt,

                            reverse_out
                                .as_mut_ptr(),

                            aout
                                .as_mut_ptr(),
                        );
                    }

                    // Only grad accumulates image.
                    for i in 0..ncell {
                        image[i] += ac[i];
                    }

                    adj_state =
                        astate;

                    reverse_calls_shot += 1;
                    reverse_calls_total += 1;
                }


                // ==========================================
                // fetch: free checkpoint
                // ==========================================

                "fetch" => {

                    if action.step == 0 {
                        bail!(
                            "Treeverse attempted to free s0"
                        );
                    }

                    if saved.remove(
                        &action.step
                    ).is_none()
                    {
                        bail!(
                            "fetch requested missing state s_{}",
                            action.step
                        );
                    }
                }


                other => {
                    bail!(
                        "unknown Treeverse action '{}'",
                        other
                    );
                }
            }

            // =================================================
            // Runtime budget audit
            // =================================================

            if saved.len()
                != action.saved_states
            {
                bail!(
                    "saved-state count mismatch after {} step {}: runtime {}, schedule {}",
                    action.action,
                    action.step,
                    saved.len(),
                    action.saved_states
                );
            }

            if saved.len()
                > checkpoints + 1
            {
                bail!(
                    "checkpoint budget exceeded: {} saved states, limit {}",
                    saved.len(),
                    checkpoints + 1
                );
            }
        }

        // ==================================================
        // End-of-shot audits
        // ==================================================

        if saved.len() != 1
            || !saved.contains_key(&0)
        {
            bail!(
                "Treeverse shot {} did not finish with only s0",
                shot_index
            );
        }

        if reverse_calls_shot
            != exp.steps
        {
            bail!(
                "shot {} reversed {} steps; expected {}",
                shot_index,
                reverse_calls_shot,
                exp.steps
            );
        }

        scheduler_forward_calls_total +=
            plan.forward_calls;

        overall_peak_saved_states =
            overall_peak_saved_states.max(
                plan.peak_saved_states
            );

        // ==================================================
        // Write actions-<shot>.json
        // ==================================================

        let actions_file =
            format!(
                "actions-{}.json",
                shot_index
            );

        let actions_json =
            serde_json::to_value(
                &plan.actions
            )?;

        write_json(
            args.out.join(
                &actions_file
            ),
            &actions_json,
        )?;

        per_shot_statistics.push(
            json!({
                "reverse_calls":
                    reverse_calls_shot,

                "scheduler_forward_calls":
                    plan.forward_calls,

                "peak_saved_states":
                    plan.peak_saved_states,

                "actions_file":
                    actions_file
            })
        );

        println!(
            "  forward calls = {}",
            plan.forward_calls
        );

        println!(
            "  reverse calls = {}",
            reverse_calls_shot
        );

        println!(
            "  peak saved states = {}",
            plan.peak_saved_states
        );
    }


    // ======================================================
    // Save image
    // ======================================================

    let image_array =
        Array2::from_shape_vec(
            (
                exp.nz,
                exp.nx
            ),
            image.clone(),
        )?;

    write_npy(
        args.out.join(
            "image.npy"
        ),
        &image_array,
    )?;


    // ======================================================
    // Transpose identity
    // ======================================================

    let left =
        weights
            .iter()
            .map(
                |v| v * v
            )
            .sum::<f64>();

    let right =
        perturbation
            .iter()
            .zip(
                image.iter()
            )
            .map(
                |(m, g)| m * g
            )
            .sum::<f64>();

    let relative_difference =
        (left - right).abs()
        /
        left.abs()
            .max(right.abs())
            .max(1.0e-30);


    let peak_saved_bytes =
        overall_peak_saved_states
        * bytes_per_state;


    println!();
    println!(
        "Treeverse transpose identity"
    );

    println!(
        "left  = {:.15e}",
        left
    );

    println!(
        "right = {:.15e}",
        right
    );

    println!(
        "relative difference = {:.3e}",
        relative_difference
    );

    println!(
        "peak saved states = {}",
        overall_peak_saved_states
    );

    println!(
        "peak saved bytes = {}",
        peak_saved_bytes
    );


    // ======================================================
    // result.json
    // ======================================================

    let statistics =
        json!({
            "storage":
                "treeverse",

            "checkpoints":
                checkpoints,

            "reverse_calls":
                reverse_calls_total,

            "scheduler_forward_calls":
                scheduler_forward_calls_total,

            "peak_saved_states":
                overall_peak_saved_states,

            "peak_saved_bytes":
                peak_saved_bytes,

            "per_shot":
                per_shot_statistics
        });


    let result_json =
        json!({
            "mode":
                "adjoint",

            "nx":
                exp.nx,

            "nz":
                exp.nz,

            "dx":
                exp.dx,

            "dt":
                exp.dt,

            "steps":
                exp.steps,

            "shots":
                exp.shots,

            "receivers":
                exp.receivers,

            "statistics":
                statistics,

            "transpose_identity": {
                "left":
                    left,

                "right":
                    right,

                "relative_difference":
                    relative_difference
            }
        });


    write_json(
        args.out.join(
            "result.json"
        ),
        &result_json,
    )?;


    // ======================================================
    // run.json
    // ======================================================

    let mut experiment_metadata =
        raw_json.clone();

    if let Some(obj) =
        experiment_metadata
            .as_object_mut()
    {
        obj.remove("background");
        obj.remove("perturbation");
    }

    let run_json =
        json!({
            "experiment_file":
                args.experiment
                    .to_string_lossy(),

            "experiment":
                experiment_metadata,

            "storage":
                "treeverse",

            "checkpoints":
                checkpoints
        });


    write_json(
        args.out.join(
            "run.json"
        ),
        &run_json,
    )?;

    Ok(())
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
  if args.mode == "born" {
    return run_born_mode(
        &args,
        &exp,
        &raw_json,
        &background,
        &perturbation,
    );
}
if args.mode == "adjoint" {

    match args.storage.as_str() {

        "full" => {
            return run_adjoint_mode(
                &args,
                &exp,
                &raw_json,
                &background,
                &perturbation,
            );
        }

        "treeverse" => {
            return run_adjoint_treeverse_mode(
                &args,
                &exp,
                &raw_json,
                &background,
                &perturbation,
            );
        }

        other => {
            bail!(
                "unknown storage '{}'; supported: full, treeverse",
                other
            );
        }
    }
}
if args.mode != "forward" {
    bail!(
        "unknown mode '{}'; supported: forward, born, adjoint",
        args.mode
    );
}
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
