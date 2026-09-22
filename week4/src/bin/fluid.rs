use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs::{File, create_dir_all};
use std::io::{self, BufWriter, Read, Write};
use std::path::PathBuf;

use week4::flow::FlowProblem;
use week4::integrator::{Euler, Integrator, Midpoint, Rk4};

#[derive(Parser)]
#[command(name = "fluid")]
#[command(about = "Integrate a periodic incompressible velocity field")]
struct Cli {
    /// Time integrator: euler, rk2, or rk4.
    #[arg(long)]
    method: String,

    /// Kinematic viscosity.
    #[arg(long)]
    nu: f64,

    /// Fixed time step.
    #[arg(long)]
    dt: f64,

    /// Final integration time.
    #[arg(long = "t-end")]
    t_end: f64,

    /// Requested time between stored snapshots.
    #[arg(long)]
    every: f64,

    /// Output directory.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Debug, Deserialize)]
struct FieldInput {
    case: String,
    n: usize,
    seed: Option<u64>,
    k_band: Option<[usize; 2]>,
    u: Vec<f64>,
    v: Vec<f64>,
}

#[derive(Serialize)]
struct RunMetadata {
    case: String,
    n: usize,
    seed: Option<u64>,
    k_band: Option<[usize; 2]>,

    method: String,
    nu: f64,
    dt: f64,
    t_end: f64,
    snapshot_every: f64,
}

#[derive(Serialize)]
struct Snapshot {
    t: f64,
    step: usize,
    u: Vec<f64>,
    v: Vec<f64>,
    omega: Vec<f64>,
}

fn round6(x: f64) -> f64 {
    (x * 1_000_000.0).round() / 1_000_000.0
}

fn round_vec6(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| round6(v)).collect()
}

fn make_integrator(method: &str) -> Box<dyn Integrator> {
    match method {
        "euler" => Box::new(Euler),
        "rk2" => Box::new(Midpoint),
        "rk4" => Box::new(Rk4),

        _ => panic!("unknown method '{}'; expected euler, rk2, or rk4", method),
    }
}

fn write_snapshot(
    writer: &mut BufWriter<File>,
    flow: &FlowProblem,
    omega: &[f64],
    step: usize,
    dt: f64,
) {
    let (u, v) = flow.velocity_from_vorticity(omega);

    let snapshot = Snapshot {
        t: round6(step as f64 * dt),
        step,
        u: round_vec6(&u),
        v: round_vec6(&v),
        omega: round_vec6(omega),
    };

    serde_json::to_writer(&mut *writer, &snapshot).expect("failed to write fields.jsonl");

    writeln!(writer).expect("failed to finish snapshot line");
}

fn main() {
    let cli = Cli::parse();

    assert!(cli.nu >= 0.0, "nu must be non-negative");

    assert!(cli.dt > 0.0, "dt must be positive");

    assert!(cli.t_end >= 0.0, "t-end must be non-negative");

    assert!(cli.every > 0.0, "every must be positive");

    // ---------------------------------------------------------
    // Read the one JSON object written by `field`.
    // ---------------------------------------------------------
    let mut text = String::new();

    io::stdin()
        .read_to_string(&mut text)
        .expect("failed to read field JSON from stdin");

    let field: FieldInput = serde_json::from_str(&text).expect("invalid field JSON");

    assert_eq!(field.u.len(), field.n * field.n, "u has the wrong size");

    assert_eq!(field.v.len(), field.n * field.n, "v has the wrong size");

    // ---------------------------------------------------------
    // Set up the flow and reconstruct initial vorticity.
    // ---------------------------------------------------------
    let flow = FlowProblem::new(field.n, cli.nu);

    let mut omega = flow.vorticity_from_velocity(&field.u, &field.v);

    // Explicitly enforce the carried 2/3 cutoff.
    omega = flow.project_vorticity(&omega);

    // ---------------------------------------------------------
    // The requested snapshot interval is converted to a whole
    // number of fixed dt steps. No time step is ever shortened
    // merely to land on a snapshot.
    // ---------------------------------------------------------
    let snapshot_steps = (cli.every / cli.dt).round() as usize;

    assert!(snapshot_steps >= 1, "every is too small relative to dt");

    let actual_snapshot_every = snapshot_steps as f64 * cli.dt;

    // ---------------------------------------------------------
    // Prepare output directory and run.json.
    // ---------------------------------------------------------
    create_dir_all(&cli.out).expect("failed to create output directory");

    let metadata = RunMetadata {
        case: field.case.clone(),
        n: field.n,
        seed: field.seed,
        k_band: field.k_band,

        method: cli.method.clone(),
        nu: cli.nu,
        dt: cli.dt,
        t_end: cli.t_end,
        snapshot_every: actual_snapshot_every,
    };

    let run_file = File::create(cli.out.join("run.json")).expect("failed to create run.json");

    serde_json::to_writer_pretty(BufWriter::new(run_file), &metadata)
        .expect("failed to write run.json");

    let fields_file =
        File::create(cli.out.join("fields.jsonl")).expect("failed to create fields.jsonl");

    let mut fields_writer = BufWriter::new(fields_file);

    // ---------------------------------------------------------
    // Choose the time integrator.
    // ---------------------------------------------------------
    let integrator = make_integrator(&cli.method);

    // ---------------------------------------------------------
    // stdout contract:
    //
    // header followed by tab-separated t, E, Z.
    // ---------------------------------------------------------
    println!("t\tE\tZ");

    // Initial state is always a stored snapshot.
    let e0 = flow.energy(&omega);
    let z0 = flow.enstrophy(&omega);

    println!("{:.6}\t{:.6}\t{:.6}", 0.0, e0, z0);

    write_snapshot(&mut fields_writer, &flow, &omega, 0, cli.dt);

    // ---------------------------------------------------------
    // Fixed-step integration.
    //
    // We never shorten dt to hit either a snapshot or t_end.
    // The last full step is taken only if it does not pass t_end.
    // ---------------------------------------------------------
    let mut step: usize = 0;

    while (step + 1) as f64 * cli.dt <= cli.t_end + 1.0e-12 {
        let t = step as f64 * cli.dt;

        let next = integrator.step(t, &omega, cli.dt, &|time, state| flow.rhs(time, state));

        step += 1;

        // Keep only the modes that the solver is supposed
        // to carry.
        omega = flow.project_vorticity(&next);

        let time = step as f64 * cli.dt;

        let energy = flow.energy(&omega);

        let enstrophy = flow.enstrophy(&omega);

        // -----------------------------------------------------
        // Blow-up contract:
        //
        // stop immediately when energy becomes non-finite.
        // Print that line, but do NOT store a non-finite field.
        // -----------------------------------------------------
        if !energy.is_finite() || !enstrophy.is_finite() {
            println!("{:.6}\t{}\t{}", time, energy, enstrophy);

            fields_writer.flush().expect("failed to flush fields.jsonl");

            std::process::exit(1);
        }

        // -----------------------------------------------------
        // Store only at the rounded snapshot interval.
        // -----------------------------------------------------
        if step % snapshot_steps == 0 {
            println!("{:.6}\t{:.6}\t{:.6}", time, energy, enstrophy);

            write_snapshot(&mut fields_writer, &flow, &omega, step, cli.dt);
        }
    }

    fields_writer.flush().expect("failed to flush fields.jsonl");
}
