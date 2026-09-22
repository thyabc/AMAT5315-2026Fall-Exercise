use clap::{Parser, Subcommand};
use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rustfft::num_complex::Complex;
use serde::Serialize;
use std::f64::consts::PI;

use week4::spectral::{Spectral2D, coordinate};

#[derive(Parser)]
#[command(name = "field")]
#[command(about = "Generate a periodic velocity field")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Taylor-Green exact velocity field.
    TaylorGreen {
        /// Grid points per side.
        #[arg(long)]
        n: usize,

        /// Time of the exact solution.
        #[arg(long, default_value_t = 0.0)]
        t: f64,

        /// Kinematic viscosity; required when t > 0.
        #[arg(long)]
        nu: Option<f64>,
    },

    /// Seeded random vorticity field.
    Random {
        /// Grid points per side.
        #[arg(long)]
        n: usize,

        /// Random seed.
        #[arg(long)]
        seed: u64,

        /// Smallest retained |k|.
        #[arg(long = "k-min")]
        k_min: usize,

        /// Largest retained |k|.
        #[arg(long = "k-max")]
        k_max: usize,
    },
}

#[derive(Serialize)]
struct FieldOutput {
    case: String,
    n: usize,

    // Taylor-Green has no random seed.
    seed: Option<u64>,

    // Random case writes [k_min, k_max].
    k_band: Option<[usize; 2]>,

    u: Vec<f64>,
    v: Vec<f64>,
}

fn check_n(n: usize) {
    assert!(
        n >= 4 && n.is_power_of_two(),
        "n must be a power of two >= 4"
    );
}

/// Convert an integer Fourier wavenumber to rustfft storage index.
///
/// Examples for n=8:
///
/// k =  0 -> 0
/// k =  1 -> 1
/// k = -1 -> 7
/// k = -2 -> 6
fn mode_index(k: isize, n: usize) -> usize {
    if k >= 0 {
        k as usize
    } else {
        (n as isize + k) as usize
    }
}

fn taylor_green(n: usize, t: f64, nu: Option<f64>) -> FieldOutput {
    check_n(n);

    assert!(t >= 0.0, "t must be non-negative");

    let viscosity = if t > 0.0 {
        nu.expect("--nu is required when --t > 0")
    } else {
        nu.unwrap_or(0.0)
    };

    assert!(viscosity >= 0.0, "nu must be non-negative");

    let decay = (-2.0 * viscosity * t).exp();

    let mut u = vec![0.0; n * n];
    let mut v = vec![0.0; n * n];

    for iy in 0..n {
        let y = coordinate(n, iy);

        for ix in 0..n {
            let x = coordinate(n, ix);
            let idx = iy * n + ix;

            u[idx] = x.cos() * y.sin() * decay;

            v[idx] = -x.sin() * y.cos() * decay;
        }
    }

    FieldOutput {
        case: "taylor-green".to_string(),
        n,
        seed: None,
        k_band: None,
        u,
        v,
    }
}

/// Generate the random case from equal-amplitude vorticity modes.
///
/// Every independent conjugate pair receives one phase drawn uniformly
/// from [0, 2*pi).  The order of these phase draws depends only on the
/// integer wavevectors and k-band, not on n.
fn random_field(n: usize, seed: u64, k_min: usize, k_max: usize) -> FieldOutput {
    check_n(n);

    assert!(k_min >= 1, "k-min must be at least 1");

    assert!(k_min <= k_max, "k-min must be <= k-max");

    // This also avoids self-conjugate Nyquist modes.
    assert!(k_max < n / 2, "k-max must be below the Nyquist wavenumber");

    let spectral = Spectral2D::new(n);

    let mut omega_hat = vec![Complex::new(0.0, 0.0); n * n];

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let kmin2 = (k_min * k_min) as isize;
    let kmax2 = (k_max * k_max) as isize;

    /*
     * Visit one member of every conjugate pair.
     *
     * Canonical half-plane:
     *
     *     ky > 0
     *
     * or
     *
     *     ky == 0 and kx > 0.
     *
     * The loops run only over the requested physical k-band,
     * so phase-draw order is independent of the grid size n.
     */
    for ky in -(k_max as isize)..=(k_max as isize) {
        for kx in -(k_max as isize)..=(k_max as isize) {
            let k2 = kx * kx + ky * ky;

            if k2 < kmin2 || k2 > kmax2 {
                continue;
            }

            if kx == 0 && ky == 0 {
                continue;
            }

            let canonical = ky > 0 || (ky == 0 && kx > 0);

            if !canonical {
                continue;
            }

            let phase = rng.random_range(0.0..(2.0 * PI));

            /*
             * The overall coefficient magnitude is arbitrary here,
             * because all modes are rescaled together below so that
             * E(0) = 0.5.
             *
             * n^2 keeps real-space magnitudes conveniently O(1)
             * before that final scaling.
             */
            let amplitude = (n * n) as f64;

            let coefficient = Complex::from_polar(amplitude, phase);

            let ix = mode_index(kx, n);

            let iy = mode_index(ky, n);

            let ix_conj = mode_index(-kx, n);

            let iy_conj = mode_index(-ky, n);

            omega_hat[iy * n + ix] = coefficient;

            omega_hat[iy_conj * n + ix_conj] = coefficient.conj();
        }
    }

    /*
     * The requested initial band is far below the 2/3 cutoff for
     * the contract run, but applying the filter here makes that
     * convention explicit.
     */
    spectral.apply_two_thirds_hat(&mut omega_hat);

    let omega = spectral.inverse_real(omega_hat);

    let (mut u, mut v) = spectral.velocity_from_vorticity(&omega);

    /*
     * Scale the entire field to
     *
     *     E(0) = 1/2 <u^2 + v^2> = 0.5.
     */
    let size = (n * n) as f64;

    let energy = 0.5
        * u.iter()
            .zip(v.iter())
            .map(|(&ui, &vi)| ui * ui + vi * vi)
            .sum::<f64>()
        / size;

    assert!(
        energy.is_finite() && energy > 0.0,
        "random field has invalid initial energy"
    );

    let scale = (0.5 / energy).sqrt();

    for value in &mut u {
        *value *= scale;
    }

    for value in &mut v {
        *value *= scale;
    }

    FieldOutput {
        case: "random".to_string(),
        n,
        seed: Some(seed),
        k_band: Some([k_min, k_max]),
        u,
        v,
    }
}

fn main() {
    let cli = Cli::parse();

    let output = match cli.command {
        Command::TaylorGreen { n, t, nu } => taylor_green(n, t, nu),

        Command::Random {
            n,
            seed,
            k_min,
            k_max,
        } => random_field(n, seed, k_min, k_max),
    };

    /*
     * field.design.toml requires one JSON object on stdout.
     *
     * Do not print diagnostic text to stdout here: later this
     * output is piped directly into `fluid`.
     */
    serde_json::to_writer(std::io::stdout(), &output).expect("failed to write field JSON");

    println!();
}
