use std::f64::consts::PI;
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};

use week4::integrator::{Euler, Integrator, Midpoint, Rk4};
use week4::line::{DerivativeMethod, LineProblem, max_abs_error, periodic_grid};

fn periodic_gaussian_exact(n: usize, x0: f64, sigma: f64, c: f64, nu: f64, t: f64) -> Vec<f64> {
    let x = periodic_grid(n);

    let sigma2 = sigma * sigma + 2.0 * nu * t;
    let amplitude = sigma / sigma2.sqrt();
    let center = x0 + c * t;

    x.iter()
        .map(|&xj| {
            (-6..=6)
                .map(|m| {
                    let d = xj - center + 2.0 * PI * m as f64;

                    amplitude * (-d * d / (2.0 * sigma2)).exp()
                })
                .sum()
        })
        .collect()
}

fn integrate_to_time(
    problem: &LineProblem,
    integrator: &dyn Integrator,
    initial: &[f64],
    dt: f64,
    t_end: f64,
) -> Vec<f64> {
    let mut u = initial.to_vec();
    let mut t = 0.0;

    while t < t_end - 1.0e-14 {
        // Part 1 wants the solution exactly at t_end.
        // The final line-problem step may therefore be shorter.
        let h = (t_end - t).min(dt);

        u = integrator.step(t, &u, h, &|time, state| problem.rhs(time, state));

        t += h;
    }

    u
}

fn broken_rk4_step(problem: &LineProblem, t: f64, u: &[f64], dt: f64) -> Vec<f64> {
    let k1 = problem.rhs(t, u);

    let y2: Vec<f64> = u
        .iter()
        .zip(k1.iter())
        .map(|(&ui, &ki)| ui + 0.5 * dt * ki)
        .collect();

    let k2 = problem.rhs(t + 0.5 * dt, &y2);

    let y3: Vec<f64> = u
        .iter()
        .zip(k2.iter())
        .map(|(&ui, &ki)| ui + 0.5 * dt * ki)
        .collect();

    let k3 = problem.rhs(t + 0.5 * dt, &y3);

    let y4: Vec<f64> = u
        .iter()
        .zip(k3.iter())
        .map(|(&ui, &ki)| ui + dt * ki)
        .collect();

    let k4 = problem.rhs(t + dt, &y4);

    // Deliberately wrong final RK4 weights:
    // 1/4, 1/4, 1/4, 1/4.
    u.iter()
        .zip(k1.iter())
        .zip(k2.iter())
        .zip(k3.iter())
        .zip(k4.iter())
        .map(|((((&ui, &a), &b), &c), &d)| ui + dt * (a + b + c + d) / 4.0)
        .collect()
}

fn integrate_broken_rk4(problem: &LineProblem, initial: &[f64], dt: f64, t_end: f64) -> Vec<f64> {
    let mut u = initial.to_vec();
    let mut t = 0.0;

    while t < t_end - 1.0e-14 {
        let h = (t_end - t).min(dt);

        u = broken_rk4_step(problem, t, &u, h);

        t += h;
    }

    u
}

fn main() {
    create_dir_all("artifacts").unwrap();

    // =========================================================
    // Panel 1: pulse race after one lap
    // =========================================================
    let n = 64;
    let c = 1.0;
    let nu = 0.002;
    let sigma = 0.25;
    let x0 = PI / 2.0;
    let t_end = 2.0 * PI;

    let fourier = LineProblem::new(n, c, nu, DerivativeMethod::Fourier);

    let centered = LineProblem::new(n, c, nu, DerivativeMethod::Centered);

    let initial = periodic_gaussian_exact(n, x0, sigma, c, nu, 0.0);

    let exact = periodic_gaussian_exact(n, x0, sigma, c, nu, t_end);

    let rk4 = Rk4;
    let euler = Euler;

    let fourier_rk4 = integrate_to_time(&fourier, &rk4, &initial, 0.02, t_end);

    let centered_rk4 = integrate_to_time(&centered, &rk4, &initial, 0.02, t_end);

    let fourier_euler = integrate_to_time(&fourier, &euler, &initial, 0.005, t_end);

    println!(
        "race Fourier + RK4 max error  = {:.8e}",
        max_abs_error(&fourier_rk4, &exact)
    );

    println!(
        "race centered + RK4 max error = {:.8e}",
        max_abs_error(&centered_rk4, &exact)
    );

    println!(
        "race Fourier + Euler max error = {:.8e}",
        max_abs_error(&fourier_euler, &exact)
    );

    let x = periodic_grid(n);

    let mut race = BufWriter::new(File::create("artifacts/line-race.csv").unwrap());

    writeln!(race, "x,exact,fourier_rk4,centered_rk4,fourier_euler").unwrap();

    for j in 0..n {
        writeln!(
            race,
            "{:.15},{:.15},{:.15},{:.15},{:.15}",
            x[j], exact[j], fourier_rk4[j], centered_rk4[j], fourier_euler[j]
        )
        .unwrap();
    }

    // =========================================================
    // Panel 2: order study
    // =========================================================
    let n = 64;
    let c = 1.0;
    let nu = 0.05;
    let sigma = 0.35;
    let x0 = PI / 2.0;
    let t_end = 1.0;

    let problem = LineProblem::new(n, c, nu, DerivativeMethod::Fourier);

    let initial = periodic_gaussian_exact(n, x0, sigma, c, nu, 0.0);

    let exact = periodic_gaussian_exact(n, x0, sigma, c, nu, t_end);

    let euler = Euler;
    let midpoint = Midpoint;
    let rk4 = Rk4;

    let dts = [0.02, 0.01, 0.005, 0.0025];

    let mut order = BufWriter::new(File::create("artifacts/line-order.csv").unwrap());

    writeln!(order, "method,dt,error").unwrap();

    for &dt in &dts {
        let numerical = integrate_to_time(&problem, &euler, &initial, dt, t_end);

        let error = max_abs_error(&numerical, &exact);

        writeln!(order, "euler,{:.15},{:.15e}", dt, error).unwrap();
    }

    for &dt in &dts {
        let numerical = integrate_to_time(&problem, &midpoint, &initial, dt, t_end);

        let error = max_abs_error(&numerical, &exact);

        writeln!(order, "rk2,{:.15},{:.15e}", dt, error).unwrap();
    }

    for &dt in &dts {
        let numerical = integrate_to_time(&problem, &rk4, &initial, dt, t_end);

        let error = max_abs_error(&numerical, &exact);

        writeln!(order, "rk4,{:.15},{:.15e}", dt, error).unwrap();
    }

    for &dt in &dts {
        let numerical = integrate_broken_rk4(&problem, &initial, dt, t_end);

        let error = max_abs_error(&numerical, &exact);

        writeln!(order, "rk4-broken,{:.15},{:.15e}", dt, error).unwrap();
    }

    println!("Wrote line accuracy data.");
}
