use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

/// Two-dimensional Fourier operators on the periodic box
/// [0, 2*pi)^2.
///
/// Real-space storage is row-major:
///
///```text
///     index = iy * n + ix
///```
pub struct Spectral2D {
    pub n: usize,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
}

impl Spectral2D {
    pub fn new(n: usize) -> Self {
        assert!(n >= 4);

        let mut planner = FftPlanner::<f64>::new();

        let forward = planner.plan_fft_forward(n);
        let inverse = planner.plan_fft_inverse(n);

        Self {
            n,
            forward,
            inverse,
        }
    }

    /// FFT wavenumber:
    ///
    /// even n:
    /// 0, 1, ..., n/2-1, -n/2, ..., -1
    pub fn wavenumber(&self, j: usize) -> isize {
        if j <= (self.n - 1) / 2 {
            j as isize
        } else {
            j as isize - self.n as isize
        }
    }

    fn index(&self, ix: usize, iy: usize) -> usize {
        iy * self.n + ix
    }

    /// Forward 2-D FFT.
    ///
    /// The forward transform is unnormalised.
    pub fn forward_real(&self, field: &[f64]) -> Vec<Complex<f64>> {
        assert_eq!(field.len(), self.n * self.n);

        let mut data: Vec<Complex<f64>> = field.iter().map(|&x| Complex::new(x, 0.0)).collect();

        self.forward_complex_in_place(&mut data);

        data
    }

    /// Forward transform of already-complex data.
    pub fn forward_complex_in_place(&self, data: &mut [Complex<f64>]) {
        assert_eq!(data.len(), self.n * self.n);

        // FFT along x: rows are contiguous.
        for iy in 0..self.n {
            let start = iy * self.n;
            let end = start + self.n;

            self.forward.process(&mut data[start..end]);
        }

        // FFT along y: columns must be gathered.
        let mut column = vec![Complex::<f64>::new(0.0, 0.0); self.n];

        for ix in 0..self.n {
            for iy in 0..self.n {
                column[iy] = data[self.index(ix, iy)];
            }

            self.forward.process(&mut column);

            for iy in 0..self.n {
                data[self.index(ix, iy)] = column[iy];
            }
        }
    }

    /// Inverse 2-D FFT, returning the real part.
    ///
    /// rustfft leaves inverse transforms unnormalised,
    /// so a 2-D inverse needs 1 / n^2.
    pub fn inverse_real(&self, mut data: Vec<Complex<f64>>) -> Vec<f64> {
        assert_eq!(data.len(), self.n * self.n);

        // Inverse along x.
        for iy in 0..self.n {
            let start = iy * self.n;
            let end = start + self.n;

            self.inverse.process(&mut data[start..end]);
        }

        // Inverse along y.
        let mut column = vec![Complex::<f64>::new(0.0, 0.0); self.n];

        for ix in 0..self.n {
            for iy in 0..self.n {
                column[iy] = data[self.index(ix, iy)];
            }

            self.inverse.process(&mut column);

            for iy in 0..self.n {
                data[self.index(ix, iy)] = column[iy];
            }
        }

        let scale = 1.0 / (self.n * self.n) as f64;

        data.iter().map(|z| z.re * scale).collect()
    }

    /// Inverse 2-D FFT retaining complex values.
    pub fn inverse_complex(&self, mut data: Vec<Complex<f64>>) -> Vec<Complex<f64>> {
        assert_eq!(data.len(), self.n * self.n);

        for iy in 0..self.n {
            let start = iy * self.n;
            let end = start + self.n;

            self.inverse.process(&mut data[start..end]);
        }

        let mut column = vec![Complex::<f64>::new(0.0, 0.0); self.n];

        for ix in 0..self.n {
            for iy in 0..self.n {
                column[iy] = data[self.index(ix, iy)];
            }

            self.inverse.process(&mut column);

            for iy in 0..self.n {
                data[self.index(ix, iy)] = column[iy];
            }
        }

        let scale = 1.0 / (self.n * self.n) as f64;

        for z in &mut data {
            *z *= scale;
        }

        data
    }

    /// Fourier first derivative in x.
    pub fn dx(&self, field: &[f64]) -> Vec<f64> {
        let mut hat = self.forward_real(field);

        for iy in 0..self.n {
            for ix in 0..self.n {
                let idx = self.index(ix, iy);
                let kx = self.wavenumber(ix);

                // Odd derivative of the Nyquist mode is zero.
                if self.n % 2 == 0 && kx == -(self.n as isize) / 2 {
                    hat[idx] = Complex::new(0.0, 0.0);
                } else {
                    hat[idx] *= Complex::new(0.0, kx as f64);
                }
            }
        }

        self.inverse_real(hat)
    }

    /// Fourier first derivative in y.
    pub fn dy(&self, field: &[f64]) -> Vec<f64> {
        let mut hat = self.forward_real(field);

        for iy in 0..self.n {
            let ky = self.wavenumber(iy);

            for ix in 0..self.n {
                let idx = self.index(ix, iy);

                if self.n % 2 == 0 && ky == -(self.n as isize) / 2 {
                    hat[idx] = Complex::new(0.0, 0.0);
                } else {
                    hat[idx] *= Complex::new(0.0, ky as f64);
                }
            }
        }

        self.inverse_real(hat)
    }

    pub fn dxx(&self, field: &[f64]) -> Vec<f64> {
        let mut hat = self.forward_real(field);

        for iy in 0..self.n {
            for ix in 0..self.n {
                let idx = self.index(ix, iy);

                let kx = self.wavenumber(ix) as f64;

                hat[idx] *= -kx * kx;
            }
        }

        self.inverse_real(hat)
    }

    pub fn dyy(&self, field: &[f64]) -> Vec<f64> {
        let mut hat = self.forward_real(field);

        for iy in 0..self.n {
            let ky = self.wavenumber(iy) as f64;

            for ix in 0..self.n {
                let idx = self.index(ix, iy);

                hat[idx] *= -ky * ky;
            }
        }

        self.inverse_real(hat)
    }

    pub fn dxdy(&self, field: &[f64]) -> Vec<f64> {
        let mut hat = self.forward_real(field);

        for iy in 0..self.n {
            let ky_i = self.wavenumber(iy);

            for ix in 0..self.n {
                let idx = self.index(ix, iy);
                let kx_i = self.wavenumber(ix);

                let x_nyquist = self.n % 2 == 0 && kx_i == -(self.n as isize) / 2;

                let y_nyquist = self.n % 2 == 0 && ky_i == -(self.n as isize) / 2;

                if x_nyquist || y_nyquist {
                    hat[idx] = Complex::new(0.0, 0.0);
                } else {
                    let kx = kx_i as f64;
                    let ky = ky_i as f64;

                    hat[idx] *= -kx * ky;
                }
            }
        }

        self.inverse_real(hat)
    }

    pub fn laplacian(&self, field: &[f64]) -> Vec<f64> {
        let mut hat = self.forward_real(field);

        for iy in 0..self.n {
            let ky = self.wavenumber(iy) as f64;

            for ix in 0..self.n {
                let idx = self.index(ix, iy);
                let kx = self.wavenumber(ix) as f64;

                let k2 = kx * kx + ky * ky;

                hat[idx] *= -k2;
            }
        }

        self.inverse_real(hat)
    }

    /// Apply the two-thirds rule in Fourier space.
    ///
    /// Keep only
    ///
    /// |kx| <= floor(n/3)
    /// |ky| <= floor(n/3).
    pub fn apply_two_thirds_hat(&self, field_hat: &mut [Complex<f64>]) {
        assert_eq!(field_hat.len(), self.n * self.n);

        let cutoff = (self.n / 3) as isize;

        for iy in 0..self.n {
            let ky = self.wavenumber(iy);

            for ix in 0..self.n {
                let kx = self.wavenumber(ix);
                let idx = self.index(ix, iy);

                if kx.abs() > cutoff || ky.abs() > cutoff {
                    field_hat[idx] = Complex::new(0.0, 0.0);
                }
            }
        }
    }

    /// Filter a real field using the two-thirds rule.
    pub fn apply_two_thirds_real(&self, field: &[f64]) -> Vec<f64> {
        let mut hat = self.forward_real(field);

        self.apply_two_thirds_hat(&mut hat);

        self.inverse_real(hat)
    }

    /// Recover streamfunction from vorticity:
    ///
    ///```text
    ///     -laplacian(psi) = omega
    ///```
    ///
    /// hence
    ///
    ///```text
    ///     psi_hat = omega_hat / (kx^2 + ky^2)
    ///```
    ///
    /// with psi_hat(0,0) = 0.
    pub fn streamfunction_hat_from_vorticity_hat(
        &self,
        omega_hat: &[Complex<f64>],
    ) -> Vec<Complex<f64>> {
        assert_eq!(omega_hat.len(), self.n * self.n);

        let mut psi_hat = vec![Complex::new(0.0, 0.0); self.n * self.n];

        for iy in 0..self.n {
            let ky = self.wavenumber(iy) as f64;

            for ix in 0..self.n {
                let idx = self.index(ix, iy);
                let kx = self.wavenumber(ix) as f64;

                let k2 = kx * kx + ky * ky;

                if k2 == 0.0 {
                    psi_hat[idx] = Complex::new(0.0, 0.0);
                } else {
                    psi_hat[idx] = omega_hat[idx] / k2;
                }
            }
        }

        psi_hat
    }

    /// Recover velocity from vorticity.
    ///
    /// u = dpsi/dy
    /// v = -dpsi/dx
    pub fn velocity_from_vorticity(&self, omega: &[f64]) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(omega.len(), self.n * self.n);

        let mut omega_hat = self.forward_real(omega);

        // The carried field is also dealiased.
        self.apply_two_thirds_hat(&mut omega_hat);

        let psi_hat = self.streamfunction_hat_from_vorticity_hat(&omega_hat);

        let mut u_hat = vec![Complex::new(0.0, 0.0); self.n * self.n];

        let mut v_hat = vec![Complex::new(0.0, 0.0); self.n * self.n];

        for iy in 0..self.n {
            let ky_i = self.wavenumber(iy);

            for ix in 0..self.n {
                let idx = self.index(ix, iy);
                let kx_i = self.wavenumber(ix);

                let x_nyquist = self.n % 2 == 0 && kx_i == -(self.n as isize) / 2;

                let y_nyquist = self.n % 2 == 0 && ky_i == -(self.n as isize) / 2;

                if !y_nyquist {
                    u_hat[idx] = psi_hat[idx] * Complex::new(0.0, ky_i as f64);
                }

                if !x_nyquist {
                    v_hat[idx] = psi_hat[idx] * Complex::new(0.0, -(kx_i as f64));
                }
            }
        }

        (self.inverse_real(u_hat), self.inverse_real(v_hat))
    }
}

/// Periodic coordinate:
///
/// x_j = 2*pi*j/n.
pub fn coordinate(n: usize, j: usize) -> f64 {
    2.0 * PI * j as f64 / n as f64
}

/// Second-order centred x derivative.
pub fn centered_dx(field: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(field.len(), n * n);

    let dx = 2.0 * PI / n as f64;

    let mut result = vec![0.0; n * n];

    for iy in 0..n {
        for ix in 0..n {
            let ip = (ix + 1) % n;
            let im = (ix + n - 1) % n;

            result[iy * n + ix] = (field[iy * n + ip] - field[iy * n + im]) / (2.0 * dx);
        }
    }

    result
}

pub fn centered_dxx(field: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(field.len(), n * n);

    let dx = 2.0 * PI / n as f64;

    let mut result = vec![0.0; n * n];

    for iy in 0..n {
        for ix in 0..n {
            let ip = (ix + 1) % n;
            let im = (ix + n - 1) % n;

            result[iy * n + ix] =
                (field[iy * n + ip] - 2.0 * field[iy * n + ix] + field[iy * n + im]) / (dx * dx);
        }
    }

    result
}

pub fn centered_dy(field: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(field.len(), n * n);

    let dx = 2.0 * PI / n as f64;

    let mut result = vec![0.0; n * n];

    for iy in 0..n {
        let jp = (iy + 1) % n;
        let jm = (iy + n - 1) % n;

        for ix in 0..n {
            result[iy * n + ix] = (field[jp * n + ix] - field[jm * n + ix]) / (2.0 * dx);
        }
    }

    result
}

pub fn centered_dyy(field: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(field.len(), n * n);

    let dx = 2.0 * PI / n as f64;

    let mut result = vec![0.0; n * n];

    for iy in 0..n {
        let jp = (iy + 1) % n;
        let jm = (iy + n - 1) % n;

        for ix in 0..n {
            result[iy * n + ix] =
                (field[jp * n + ix] - 2.0 * field[iy * n + ix] + field[jm * n + ix]) / (dx * dx);
        }
    }

    result
}

pub fn centered_dxdy(field: &[f64], n: usize) -> Vec<f64> {
    // Apply the first derivative in x, then in y,
    // exactly as described in the learning sheet.
    let dx_field = centered_dx(field, n);

    centered_dy(&dx_field, n)
}

pub fn centered_laplacian(field: &[f64], n: usize) -> Vec<f64> {
    let xx = centered_dxx(field, n);
    let yy = centered_dyy(field, n);

    xx.iter().zip(yy.iter()).map(|(&a, &b)| a + b).collect()
}

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

    fn make_g(n: usize) -> Vec<f64> {
        let mut g = vec![0.0; n * n];

        for iy in 0..n {
            let y = coordinate(n, iy);

            for ix in 0..n {
                let x = coordinate(n, ix);

                g[iy * n + ix] = (3.0 * x).sin() * (2.0 * y).cos();
            }
        }

        g
    }

    #[test]
    fn fourier_derivatives_match_exact_wave() {
        let n = 32;

        let spectral = Spectral2D::new(n);

        let g = make_g(n);

        let mut exact_dx = vec![0.0; n * n];
        let mut exact_dxx = vec![0.0; n * n];
        let mut exact_dxdy = vec![0.0; n * n];
        let mut exact_lap = vec![0.0; n * n];

        for iy in 0..n {
            let y = coordinate(n, iy);

            for ix in 0..n {
                let x = coordinate(n, ix);
                let idx = iy * n + ix;

                exact_dx[idx] = 3.0 * (3.0 * x).cos() * (2.0 * y).cos();

                exact_dxx[idx] = -9.0 * g[idx];

                exact_dxdy[idx] = -6.0 * (3.0 * x).cos() * (2.0 * y).sin();

                exact_lap[idx] = -13.0 * g[idx];
            }
        }

        let dx = spectral.dx(&g);
        let dxx = spectral.dxx(&g);
        let dxdy = spectral.dxdy(&g);
        let lap = spectral.laplacian(&g);

        assert!(max_abs_error(&dx, &exact_dx) < 1.0e-10);

        assert!(max_abs_error(&dxx, &exact_dxx) < 1.0e-10);

        assert!(max_abs_error(&dxdy, &exact_dxdy) < 1.0e-10);

        assert!(max_abs_error(&lap, &exact_lap) < 1.0e-10);
    }

    #[test]
    fn finite_difference_errors_reduce_by_four() {
        fn errors(n: usize) -> [f64; 4] {
            let g = make_g(n);

            let mut exact_dx = vec![0.0; n * n];
            let mut exact_dxx = vec![0.0; n * n];
            let mut exact_dxdy = vec![0.0; n * n];
            let mut exact_lap = vec![0.0; n * n];

            for iy in 0..n {
                let y = coordinate(n, iy);

                for ix in 0..n {
                    let x = coordinate(n, ix);
                    let idx = iy * n + ix;

                    exact_dx[idx] = 3.0 * (3.0 * x).cos() * (2.0 * y).cos();

                    exact_dxx[idx] = -9.0 * g[idx];

                    exact_dxdy[idx] = -6.0 * (3.0 * x).cos() * (2.0 * y).sin();

                    exact_lap[idx] = -13.0 * g[idx];
                }
            }

            [
                max_abs_error(&centered_dx(&g, n), &exact_dx),
                max_abs_error(&centered_dxx(&g, n), &exact_dxx),
                max_abs_error(&centered_dxdy(&g, n), &exact_dxdy),
                max_abs_error(&centered_laplacian(&g, n), &exact_lap),
            ]
        }

        let e32 = errors(32);
        let e64 = errors(64);

        for i in 0..4 {
            let ratio = e32[i] / e64[i];

            assert!(ratio > 3.5 && ratio < 4.5, "ratio = {ratio}");
        }
    }

    #[test]
    fn two_thirds_filter_removes_high_mode() {
        let n = 32;
        let spectral = Spectral2D::new(n);

        let mut field = vec![0.0; n * n];

        // kx = 14 is above floor(32/3) = 10.
        for iy in 0..n {
            for ix in 0..n {
                let x = coordinate(n, ix);

                field[iy * n + ix] = (14.0 * x).sin();
            }
        }

        let filtered = spectral.apply_two_thirds_real(&field);

        let max_value = filtered.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);

        assert!(max_value < 1.0e-10);
    }

    #[test]
    fn velocity_recovery_is_divergence_free() {
        let n = 32;
        let spectral = Spectral2D::new(n);

        let mut omega = vec![0.0; n * n];

        for iy in 0..n {
            let y = coordinate(n, iy);

            for ix in 0..n {
                let x = coordinate(n, ix);

                omega[iy * n + ix] = (3.0 * x).sin() * (2.0 * y).cos();
            }
        }

        let (u, v) = spectral.velocity_from_vorticity(&omega);

        let ux = spectral.dx(&u);
        let vy = spectral.dy(&v);

        let divergence: Vec<f64> = ux.iter().zip(vy.iter()).map(|(&a, &b)| a + b).collect();

        let max_div = divergence.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);

        assert!(max_div < 1.0e-10);
    }
}
