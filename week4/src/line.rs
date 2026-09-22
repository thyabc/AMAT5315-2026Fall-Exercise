use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

use crate::integrator::Integrator;

/// Choice of spatial derivative for the 1-D advection-diffusion problem.
#[derive(Debug, Clone, Copy)]
pub enum DerivativeMethod {
    Fourier,
    Centered,
}

/// FFT operators for a periodic one-dimensional grid.
pub struct Fourier1D {
    n: usize,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
}

impl Fourier1D {
    pub fn new(n: usize) -> Self {
        assert!(n >= 2);

        let mut planner = FftPlanner::<f64>::new();

        let forward = planner.plan_fft_forward(n);
        let inverse = planner.plan_fft_inverse(n);

        Self {
            n,
            forward,
            inverse,
        }
    }

    /// Fourier wavenumber in FFT storage order:
    ///
    /// 0, 1, ..., n/2-1, -n/2, ..., -1  for even n.
    pub fn wavenumber(&self, j: usize) -> isize {
        if j < self.n.div_ceil(2) {
            j as isize
        } else {
            j as isize - self.n as isize
        }
    }

    fn forward_real(&self, u: &[f64]) -> Vec<Complex<f64>> {
        assert_eq!(u.len(), self.n);

        let mut data: Vec<Complex<f64>> = u.iter().map(|&x| Complex::new(x, 0.0)).collect();

        self.forward.process(&mut data);

        data
    }

    fn inverse_real(&self, mut data: Vec<Complex<f64>>) -> Vec<f64> {
        self.inverse.process(&mut data);

        let scale = 1.0 / self.n as f64;

        data.iter().map(|z| z.re * scale).collect()
    }

    /// First derivative using the Fourier multiplier i*k.
    ///
    /// For even n, the Nyquist mode k = -n/2 has its odd
    /// derivative set to zero.
    pub fn first_derivative(&self, u: &[f64]) -> Vec<f64> {
        let mut u_hat = self.forward_real(u);

        for (j, value) in u_hat.iter_mut().enumerate() {
            if self.n % 2 == 0 && j == self.n / 2 {
                // Nyquist mode: odd derivative is zero.
                *value = Complex::new(0.0, 0.0);
                continue;
            }

            let k = self.wavenumber(j) as f64;

            *value *= Complex::new(0.0, k);
        }

        self.inverse_real(u_hat)
    }

    /// Second derivative using the Fourier multiplier -k^2.
    pub fn second_derivative(&self, u: &[f64]) -> Vec<f64> {
        let mut u_hat = self.forward_real(u);

        for (j, value) in u_hat.iter_mut().enumerate() {
            let k = self.wavenumber(j) as f64;

            *value *= -k * k;
        }

        self.inverse_real(u_hat)
    }
}

/// Periodic grid x_j = 2*pi*j/n.
pub fn periodic_grid(n: usize) -> Vec<f64> {
    let dx = 2.0 * PI / n as f64;

    (0..n).map(|j| j as f64 * dx).collect()
}

/// First derivative using the centred finite difference.
pub fn centered_first_derivative(u: &[f64]) -> Vec<f64> {
    let n = u.len();
    let dx = 2.0 * PI / n as f64;

    let mut du = vec![0.0; n];

    for j in 0..n {
        let jp = (j + 1) % n;
        let jm = (j + n - 1) % n;

        du[j] = (u[jp] - u[jm]) / (2.0 * dx);
    }

    du
}

/// Second derivative using the centred finite difference.
pub fn centered_second_derivative(u: &[f64]) -> Vec<f64> {
    let n = u.len();
    let dx = 2.0 * PI / n as f64;

    let mut d2u = vec![0.0; n];

    for j in 0..n {
        let jp = (j + 1) % n;
        let jm = (j + n - 1) % n;

        d2u[j] = (u[jp] - 2.0 * u[j] + u[jm]) / (dx * dx);
    }

    d2u
}

/// One-dimensional periodic advection-diffusion problem:
///
/// u_t + c u_x = nu u_xx.
pub struct LineProblem {
    pub n: usize,
    pub c: f64,
    pub nu: f64,
    pub derivative: DerivativeMethod,
    fourier: Fourier1D,
}

impl LineProblem {
    pub fn new(n: usize, c: f64, nu: f64, derivative: DerivativeMethod) -> Self {
        Self {
            n,
            c,
            nu,
            derivative,
            fourier: Fourier1D::new(n),
        }
    }

    /// Rate function:
    ///
    /// du/dt = -c*u_x + nu*u_xx.
    pub fn rhs(&self, _t: f64, u: &[f64]) -> Vec<f64> {
        assert_eq!(u.len(), self.n);

        let (ux, uxx) = match self.derivative {
            DerivativeMethod::Fourier => (
                self.fourier.first_derivative(u),
                self.fourier.second_derivative(u),
            ),

            DerivativeMethod::Centered => {
                (centered_first_derivative(u), centered_second_derivative(u))
            }
        };

        ux.iter()
            .zip(uxx.iter())
            .map(|(&uxj, &uxxj)| -self.c * uxj + self.nu * uxxj)
            .collect()
    }

    /// Exact solution for the initial condition sin(k*x):
    ///
    /// u(x,t) = exp(-nu*k^2*t) * sin(k*(x-c*t)).
    pub fn exact_single_wave(&self, k: usize, t: f64) -> Vec<f64> {
        let amplitude = (-self.nu * (k * k) as f64 * t).exp();

        periodic_grid(self.n)
            .iter()
            .map(|&x| amplitude * ((k as f64) * (x - self.c * t)).sin())
            .collect()
    }

    /// Integrate from t = 0 to t_end using a fixed time step.
    pub fn integrate(
        &self,
        integrator: &dyn Integrator,
        initial: &[f64],
        dt: f64,
        t_end: f64,
    ) -> Vec<f64> {
        assert_eq!(initial.len(), self.n);

        let steps_f = t_end / dt;
        let steps = steps_f.round() as usize;

        assert!(
            (steps as f64 * dt - t_end).abs() < 1.0e-12,
            "t_end must be a whole number of time steps"
        );

        let mut u = initial.to_vec();
        let mut t = 0.0;

        for _ in 0..steps {
            u = integrator.step(t, &u, dt, &|time, state| self.rhs(time, state));

            t += dt;
        }

        u
    }
}

/// Maximum absolute difference between two vectors.
pub fn max_abs_error(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());

    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrator::{Euler, Midpoint, Rk4};

    #[test]
    fn fourier_first_derivative_is_exact_for_wave() {
        let n = 64;
        let fft = Fourier1D::new(n);
        let x = periodic_grid(n);

        let u: Vec<f64> = x.iter().map(|&xj| (3.0 * xj).sin()).collect();

        let exact: Vec<f64> = x.iter().map(|&xj| 3.0 * (3.0 * xj).cos()).collect();

        let numerical = fft.first_derivative(&u);

        assert!(max_abs_error(&numerical, &exact) < 1.0e-11);
    }

    #[test]
    fn fourier_second_derivative_is_exact_for_wave() {
        let n = 64;
        let fft = Fourier1D::new(n);
        let x = periodic_grid(n);

        let u: Vec<f64> = x.iter().map(|&xj| (3.0 * xj).sin()).collect();

        let exact: Vec<f64> = u.iter().map(|&uj| -9.0 * uj).collect();

        let numerical = fft.second_derivative(&u);

        assert!(max_abs_error(&numerical, &exact) < 1.0e-10);
    }

    #[test]
    fn nyquist_first_derivative_is_zero() {
        let n = 64;
        let fft = Fourier1D::new(n);

        let u: Vec<f64> = (0..n)
            .map(|j| if j % 2 == 0 { 1.0 } else { -1.0 })
            .collect();

        let du = fft.first_derivative(&u);

        let largest = du.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);

        assert!(largest < 1.0e-12);
    }

    #[test]
    fn centered_difference_improves_when_grid_is_refined() {
        fn error(n: usize) -> f64 {
            let x = periodic_grid(n);

            let u: Vec<f64> = x.iter().map(|&xj| (3.0 * xj).sin()).collect();

            let exact: Vec<f64> = x.iter().map(|&xj| 3.0 * (3.0 * xj).cos()).collect();

            let numerical = centered_first_derivative(&u);

            max_abs_error(&numerical, &exact)
        }

        let e32 = error(32);
        let e64 = error(64);

        // Second-order finite differences should improve by
        // approximately a factor of four when dx is halved.
        let ratio = e32 / e64;

        assert!(ratio > 3.5 && ratio < 4.5);
    }

    #[test]
    fn every_integrator_matches_single_wave() {
        let n = 64;
        let c = 1.0;
        let nu = 0.01;
        let k = 1;

        let problem = LineProblem::new(n, c, nu, DerivativeMethod::Fourier);

        let initial: Vec<f64> = periodic_grid(n)
            .iter()
            .map(|&x| (k as f64 * x).sin())
            .collect();

        let dt = 0.001;
        let t_end = 0.1;

        let exact = problem.exact_single_wave(k, t_end);

        let euler = problem.integrate(&Euler, &initial, dt, t_end);

        let midpoint = problem.integrate(&Midpoint, &initial, dt, t_end);

        let rk4 = problem.integrate(&Rk4, &initial, dt, t_end);

        assert!(max_abs_error(&euler, &exact) < 1.0e-4);
        assert!(max_abs_error(&midpoint, &exact) < 1.0e-6);
        assert!(max_abs_error(&rk4, &exact) < 1.0e-10);
    }
}
