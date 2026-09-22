use clap::Parser;
use serde::{Deserialize, Serialize};
use std::io::{self, Read};

use week4::flow::FlowProblem;
use week4::spectral::coordinate;

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    nu: f64,

    #[arg(long, default_value_t = 3)]
    kx: usize,

    #[arg(long, default_value_t = 2)]
    ky: usize,

    #[arg(long, default_value_t = 1.0e-5)]
    relative: f64,
}

#[derive(Debug, Deserialize, Serialize)]
struct FieldData {
    case: String,
    n: usize,
    seed: Option<u64>,
    k_band: Option<[usize; 2]>,
    u: Vec<f64>,
    v: Vec<f64>,
}

fn main() {
    let args = Cli::parse();

    let mut text = String::new();

    io::stdin()
        .read_to_string(&mut text)
        .expect("failed to read stdin");

    let mut field: FieldData = serde_json::from_str(&text).expect("invalid field JSON");

    let flow = FlowProblem::new(field.n, args.nu);

    let mut omega = flow.vorticity_from_velocity(&field.u, &field.v);

    // Largest speed in the original field.
    let umax = field
        .u
        .iter()
        .zip(field.v.iter())
        .map(|(&u, &v)| (u * u + v * v).sqrt())
        .fold(0.0_f64, f64::max);

    let epsilon = args.relative * umax;

    /*
     * Add one real Fourier mode.
     *
     * k = (3,2) has |k| = sqrt(13),
     * which lies in the required range 2 <= |k| <= 6.
     */
    for iy in 0..field.n {
        let y = coordinate(field.n, iy);

        for ix in 0..field.n {
            let x = coordinate(field.n, ix);

            let idx = iy * field.n + ix;

            omega[idx] += epsilon * (args.kx as f64 * x + args.ky as f64 * y).cos();
        }
    }

    // Recover a divergence-free velocity field.
    let (u, v) = flow.velocity_from_vorticity(&omega);

    field.u = u;
    field.v = v;

    serde_json::to_writer(io::stdout(), &field).expect("failed to write JSON");

    println!();
}
