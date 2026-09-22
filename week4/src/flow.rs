use rustfft::num_complex::Complex;

use crate::spectral::Spectral2D;

/// Two-dimensional incompressible flow in vorticity form:
///
/// d(omega)/dt = -(u * omega_x + v * omega_y)
///                 + nu * laplacian(omega)
///
/// on the periodic box [0, 2*pi)^2.
pub struct FlowProblem {
    pub n: usize,
    pub nu: f64,
    spectral: Spectral2D,
}

impl FlowProblem {
    pub fn new(n: usize, nu: f64) -> Self {
        Self {
            n,
            nu,
            spectral: Spectral2D::new(n),
        }
    }

    fn index(&self, ix: usize, iy: usize) -> usize {
        iy * self.n + ix
    }

    /// Project a vorticity field onto the modes retained by
    /// the two-thirds dealiasing rule.
    pub fn project_vorticity(&self, omega: &[f64]) -> Vec<f64> {
        assert_eq!(omega.len(), self.n * self.n);

        let mut omega_hat = self.spectral.forward_real(omega);

        self.spectral.apply_two_thirds_hat(&mut omega_hat);

        self.spectral.inverse_real(omega_hat)
    }

    /// Recover velocity from vorticity:
    ///
    /// -laplacian(psi) = omega
    ///
    /// u = dpsi/dy
    /// v = -dpsi/dx
    pub fn velocity_from_vorticity(&self, omega: &[f64]) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(omega.len(), self.n * self.n);

        let mut omega_hat = self.spectral.forward_real(omega);

        self.spectral.apply_two_thirds_hat(&mut omega_hat);

        let psi_hat = self
            .spectral
            .streamfunction_hat_from_vorticity_hat(&omega_hat);

        let mut u_hat = vec![Complex::new(0.0, 0.0); self.n * self.n];

        let mut v_hat = vec![Complex::new(0.0, 0.0); self.n * self.n];

        for iy in 0..self.n {
            let ky = self.spectral.wavenumber(iy) as f64;

            for ix in 0..self.n {
                let kx = self.spectral.wavenumber(ix) as f64;

                let idx = self.index(ix, iy);

                // u = dpsi/dy
                u_hat[idx] = psi_hat[idx] * Complex::new(0.0, ky);

                // v = -dpsi/dx
                v_hat[idx] = psi_hat[idx] * Complex::new(0.0, -kx);
            }
        }

        (
            self.spectral.inverse_real(u_hat),
            self.spectral.inverse_real(v_hat),
        )
    }

    /// Compute vorticity from velocity:
    ///
    /// omega = dv/dx - du/dy.
    ///
    /// This is used later by the `fluid` command when it reads
    /// the u,v field written by `field`.
    pub fn vorticity_from_velocity(&self, u: &[f64], v: &[f64]) -> Vec<f64> {
        assert_eq!(u.len(), self.n * self.n);
        assert_eq!(v.len(), self.n * self.n);

        let vx = self.spectral.dx(v);
        let uy = self.spectral.dy(u);

        let omega: Vec<f64> = vx.iter().zip(uy.iter()).map(|(&a, &b)| a - b).collect();

        self.project_vorticity(&omega)
    }

    /// The vorticity-equation rate function.
    ///
    /// Every evaluation performs:
    ///
    /// 1. FFT omega and apply the two-thirds cutoff.
    /// 2. Recover psi and velocity.
    /// 3. Compute omega_x and omega_y spectrally.
    /// 4. Form u*omega_x + v*omega_y in real space.
    /// 5. FFT and dealias the nonlinear product.
    /// 6. Add viscous diffusion.
    pub fn rhs(&self, _t: f64, omega: &[f64]) -> Vec<f64> {
        assert_eq!(omega.len(), self.n * self.n);

        // -----------------------------------------------------
        // Carry only the retained vorticity modes.
        // -----------------------------------------------------
        let mut omega_hat = self.spectral.forward_real(omega);

        self.spectral.apply_two_thirds_hat(&mut omega_hat);

        // -----------------------------------------------------
        // Poisson solve:
        //
        // -laplacian(psi) = omega.
        // -----------------------------------------------------
        let psi_hat = self
            .spectral
            .streamfunction_hat_from_vorticity_hat(&omega_hat);

        // -----------------------------------------------------
        // Fourier-space velocity and vorticity gradients.
        // -----------------------------------------------------
        let mut u_hat = vec![Complex::new(0.0, 0.0); self.n * self.n];

        let mut v_hat = vec![Complex::new(0.0, 0.0); self.n * self.n];

        let mut omega_x_hat = vec![Complex::new(0.0, 0.0); self.n * self.n];

        let mut omega_y_hat = vec![Complex::new(0.0, 0.0); self.n * self.n];

        for iy in 0..self.n {
            let ky = self.spectral.wavenumber(iy) as f64;

            for ix in 0..self.n {
                let kx = self.spectral.wavenumber(ix) as f64;

                let idx = self.index(ix, iy);

                // u = dpsi/dy
                u_hat[idx] = psi_hat[idx] * Complex::new(0.0, ky);

                // v = -dpsi/dx
                v_hat[idx] = psi_hat[idx] * Complex::new(0.0, -kx);

                omega_x_hat[idx] = omega_hat[idx] * Complex::new(0.0, kx);

                omega_y_hat[idx] = omega_hat[idx] * Complex::new(0.0, ky);
            }
        }

        // -----------------------------------------------------
        // Return to the grid.
        // -----------------------------------------------------
        let u = self.spectral.inverse_real(u_hat);
        let v = self.spectral.inverse_real(v_hat);

        let omega_x = self.spectral.inverse_real(omega_x_hat);

        let omega_y = self.spectral.inverse_real(omega_y_hat);

        // -----------------------------------------------------
        // Nonlinear advection on the grid.
        //
        // A = u*omega_x + v*omega_y.
        // -----------------------------------------------------
        let advection: Vec<f64> = (0..self.n * self.n)
            .map(|i| u[i] * omega_x[i] + v[i] * omega_y[i])
            .collect();

        // -----------------------------------------------------
        // Transform the product and dealias it.
        // -----------------------------------------------------
        let mut advection_hat = self.spectral.forward_real(&advection);

        self.spectral.apply_two_thirds_hat(&mut advection_hat);

        // -----------------------------------------------------
        // RHS in Fourier space:
        //
        // omega_t =
        //     -advection
        //     + nu * laplacian(omega)
        //
        // laplacian multiplier = -|k|^2.
        // -----------------------------------------------------
        let mut rhs_hat = vec![Complex::new(0.0, 0.0); self.n * self.n];

        for iy in 0..self.n {
            let ky = self.spectral.wavenumber(iy) as f64;

            for ix in 0..self.n {
                let kx = self.spectral.wavenumber(ix) as f64;

                let idx = self.index(ix, iy);

                let k2 = kx * kx + ky * ky;

                rhs_hat[idx] = -advection_hat[idx] - self.nu * k2 * omega_hat[idx];
            }
        }

        // Numerical roundoff must not reintroduce discarded modes.
        self.spectral.apply_two_thirds_hat(&mut rhs_hat);

        self.spectral.inverse_real(rhs_hat)
    }

    /// Kinetic energy per unit mass:
    ///
    /// E = 1/2 <u^2 + v^2>.
    pub fn energy(&self, omega: &[f64]) -> f64 {
        let (u, v) = self.velocity_from_vorticity(omega);

        let sum: f64 = u
            .iter()
            .zip(v.iter())
            .map(|(&ui, &vi)| ui * ui + vi * vi)
            .sum();

        0.5 * sum / (self.n * self.n) as f64
    }

    /// Enstrophy:
    ///
    /// Z = 1/2 <omega^2>.
    pub fn enstrophy(&self, omega: &[f64]) -> f64 {
        let omega = self.project_vorticity(omega);

        let sum: f64 = omega.iter().map(|&w| w * w).sum();

        0.5 * sum / (self.n * self.n) as f64
    }
}

/// Relative Euclidean error:
///
/// ||a-b|| / ||b||.
pub fn relative_l2_error(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());

    let numerator: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum();

    let denominator: f64 = b.iter().map(|&x| x * x).sum();

    (numerator / denominator).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral::coordinate;

    fn taylor_green(n: usize, nu: f64, t: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let decay = (-2.0 * nu * t).exp();

        let mut u = vec![0.0; n * n];
        let mut v = vec![0.0; n * n];
        let mut omega = vec![0.0; n * n];

        for iy in 0..n {
            let y = coordinate(n, iy);

            for ix in 0..n {
                let x = coordinate(n, ix);
                let idx = iy * n + ix;

                u[idx] = x.cos() * y.sin() * decay;

                v[idx] = -x.sin() * y.cos() * decay;

                omega[idx] = -2.0 * x.cos() * y.cos() * decay;
            }
        }

        (u, v, omega)
    }

    #[test]
    fn taylor_green_velocity_is_recovered() {
        let n = 32;
        let nu = 0.1;

        let flow = FlowProblem::new(n, nu);

        let (u_exact, v_exact, omega) = taylor_green(n, nu, 0.0);

        let (u, v) = flow.velocity_from_vorticity(&omega);

        assert!(relative_l2_error(&u, &u_exact) < 1.0e-10);

        assert!(relative_l2_error(&v, &v_exact) < 1.0e-10);
    }

    #[test]
    fn velocity_to_vorticity_round_trip() {
        let n = 32;
        let nu = 0.1;

        let flow = FlowProblem::new(n, nu);

        let (u, v, omega_exact) = taylor_green(n, nu, 0.0);

        let omega = flow.vorticity_from_velocity(&u, &v);

        assert!(relative_l2_error(&omega, &omega_exact) < 1.0e-10);
    }

    #[test]
    fn taylor_green_rhs_is_exact_decay() {
        let n = 32;
        let nu = 0.1;

        let flow = FlowProblem::new(n, nu);

        let (_, _, omega) = taylor_green(n, nu, 0.0);

        let rhs = flow.rhs(0.0, &omega);

        // Taylor-Green has zero nonlinear advection.
        //
        // omega(t) = omega(0) exp(-2 nu t),
        //
        // so omega_t = -2 nu omega.
        let exact: Vec<f64> = omega.iter().map(|&w| -2.0 * nu * w).collect();

        assert!(relative_l2_error(&rhs, &exact) < 1.0e-10);
    }

    #[test]
    fn taylor_green_energy_and_enstrophy_are_correct() {
        let n = 32;
        let nu = 0.1;

        let flow = FlowProblem::new(n, nu);

        let (_, _, omega) = taylor_green(n, nu, 0.0);

        let e = flow.energy(&omega);
        let z = flow.enstrophy(&omega);

        assert!((e - 0.25).abs() < 1.0e-12);
        assert!((z - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn rhs_contains_no_modes_beyond_cutoff() {
        let n = 32;
        let nu = 0.01;

        let flow = FlowProblem::new(n, nu);

        let mut omega = vec![0.0; n * n];

        // A deliberately nonlinear low-mode field.
        for iy in 0..n {
            let y = coordinate(n, iy);

            for ix in 0..n {
                let x = coordinate(n, ix);
                let idx = iy * n + ix;

                omega[idx] =
                    (4.0 * x).sin() + 0.7 * (5.0 * y).cos() + 0.3 * (3.0 * x + 2.0 * y).sin();
            }
        }

        let rhs = flow.rhs(0.0, &omega);

        let spectral = Spectral2D::new(n);

        let rhs_hat = spectral.forward_real(&rhs);

        let cutoff = (n / 3) as isize;

        let mut outside_max = 0.0_f64;

        for iy in 0..n {
            let ky = spectral.wavenumber(iy);

            for ix in 0..n {
                let kx = spectral.wavenumber(ix);

                if kx.abs() > cutoff || ky.abs() > cutoff {
                    let idx = iy * n + ix;

                    outside_max = outside_max.max(rhs_hat[idx].norm());
                }
            }
        }

        assert!(outside_max < 1.0e-9);
    }
}
