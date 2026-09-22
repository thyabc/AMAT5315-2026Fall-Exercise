use std::f64::consts::PI;
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};

use week4::integrator::{Integrator, Rk4};
use week4::line::{DerivativeMethod, LineProblem, periodic_grid};

fn complex_growth(integrator: &dyn Integrator, re_lambda: f64, im_lambda: f64) -> f64 {
    // Represent y = a + ib as [a, b].
    // y' = lambda y gives
    // a' = Re(lambda)*a - Im(lambda)*b
    // b' = Im(lambda)*a + Re(lambda)*b
    let y0 = vec![1.0, 0.0];

    let rhs = |_t: f64, y: &[f64]| -> Vec<f64> {
        vec![
            re_lambda * y[0] - im_lambda * y[1],
            im_lambda * y[0] + re_lambda * y[1],
        ]
    };

    let y1 = integrator.step(0.0, &y0, 1.0, &rhs);

    (y1[0] * y1[0] + y1[1] * y1[1]).sqrt()
}

fn rk4_stability_function(re: f64, im: f64) -> f64 {
    let z = rustfft::num_complex::Complex::new(re, im);

    let one = rustfft::num_complex::Complex::new(1.0, 0.0);

    let r = one + z + z * z / 2.0 + z * z * z / 6.0 + z * z * z * z / 24.0;

    r.norm()
}

fn stability_function(method: &str, re: f64, im: f64) -> f64 {
    let z = rustfft::num_complex::Complex::new(re, im);

    let one = rustfft::num_complex::Complex::new(1.0, 0.0);

    match method {
        "euler" => (one + z).norm(),

        "rk2" => (one + z + z * z / 2.0).norm(),

        "rk4" => (one + z + z * z / 2.0 + z * z * z / 6.0 + z * z * z * z / 24.0).norm(),

        _ => panic!("unknown method"),
    }
}

fn periodic_gaussian(n: usize, center: f64, sigma: f64) -> Vec<f64> {
    let x = periodic_grid(n);

    x.iter()
        .map(|&xj| {
            // Add several periodic images.
            (-2..=2)
                .map(|m| {
                    let shift = xj - center + 2.0 * PI * m as f64;

                    (-shift * shift / (2.0 * sigma * sigma)).exp()
                })
                .sum()
        })
        .collect()
}

fn write_pulse_history(path: &str, dt: f64) {
    let n = 64;
    let c = 1.0;
    let nu = 0.05;
    let t_end = 6.0;

    let problem = LineProblem::new(n, c, nu, DerivativeMethod::Fourier);

    let rk4 = Rk4;

    let mut u = periodic_gaussian(n, PI / 2.0, 0.35);

    let mut writer = BufWriter::new(File::create(path).unwrap());

    writeln!(writer, "t,x,u").unwrap();

    let steps = (t_end / dt).ceil() as usize;

    let x = periodic_grid(n);

    for step in 0..=steps {
        let t = step as f64 * dt;

        if t > t_end + 1.0e-12 {
            break;
        }

        for j in 0..n {
            writeln!(writer, "{:.15},{:.15},{:.15}", t, x[j], u[j]).unwrap();
        }

        if step < steps {
            u = rk4.step(t, &u, dt, &|time, state| problem.rhs(time, state));
        }
    }
}

fn main() {
    create_dir_all("artifacts").unwrap();

    // ---------------------------------------------------------
    // 1. Measured growth map using the actual RK4 implementation.
    // ---------------------------------------------------------
    let rk4 = Rk4;

    let mut map = BufWriter::new(File::create("artifacts/line-stability-map.csv").unwrap());

    writeln!(map, "re,im,growth").unwrap();

    let n_grid = 321;
    let re_min = -4.0;
    let re_max = 1.0;
    let im_min = -4.0;
    let im_max = 4.0;

    for iy in 0..n_grid {
        let im = im_min + (im_max - im_min) * iy as f64 / (n_grid - 1) as f64;

        for ix in 0..n_grid {
            let re = re_min + (re_max - re_min) * ix as f64 / (n_grid - 1) as f64;

            let growth = complex_growth(&rk4, re, im);

            writeln!(map, "{:.15},{:.15},{:.15}", re, im, growth).unwrap();
        }
    }

    // ---------------------------------------------------------
    // 2. Analytic stability boundaries sampled on the same plane.
    // ---------------------------------------------------------
    let mut boundary = BufWriter::new(File::create("artifacts/stability-boundaries.csv").unwrap());

    writeln!(boundary, "re,im,euler,rk2,rk4").unwrap();

    for iy in 0..n_grid {
        let im = im_min + (im_max - im_min) * iy as f64 / (n_grid - 1) as f64;

        for ix in 0..n_grid {
            let re = re_min + (re_max - re_min) * ix as f64 / (n_grid - 1) as f64;

            writeln!(
                boundary,
                "{:.15},{:.15},{:.15},{:.15},{:.15}",
                re,
                im,
                stability_function("euler", re, im),
                stability_function("rk2", re, im),
                rk4_stability_function(re, im),
            )
            .unwrap();
        }
    }

    // ---------------------------------------------------------
    // 3. Spectrum lambda_k * dt for dt = 0.045 and 0.056.
    // lambda_k = -nu*k^2 - i*c*k.
    // Nyquist first derivative is zero.
    // ---------------------------------------------------------
    let mut modes = BufWriter::new(File::create("artifacts/line-modes.csv").unwrap());

    writeln!(modes, "dt,k,re,im").unwrap();

    let n = 64;
    let nu = 0.05;
    let c = 1.0;

    for &dt in &[0.045_f64, 0.056_f64] {
        for j in 0..n {
            let k: isize = if j < n / 2 {
                j as isize
            } else {
                j as isize - n as isize
            };

            let re = -nu * (k * k) as f64 * dt;

            let im = if j == n / 2 {
                // Nyquist odd derivative = 0.
                0.0
            } else {
                -c * k as f64 * dt
            };

            writeln!(modes, "{:.15},{},{:.15},{:.15}", dt, k, re, im).unwrap();
        }
    }

    // ---------------------------------------------------------
    // 4. Pulse runs below and above the stability boundary.
    // ---------------------------------------------------------
    write_pulse_history("artifacts/pulse-dt-0.045.csv", 0.045);

    write_pulse_history("artifacts/pulse-dt-0.056.csv", 0.056);

    println!("Wrote line stability data.");
}
