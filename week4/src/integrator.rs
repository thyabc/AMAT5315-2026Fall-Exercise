/// Common interface for the explicit time integrators used in Week 4.
///
/// The state is a vector of real values.  The rate function receives
/// the current time and state and returns dy/dt with the same length.
pub trait Integrator {
    fn name(&self) -> &'static str;

    fn step(&self, t: f64, y: &[f64], dt: f64, rhs: &dyn Fn(f64, &[f64]) -> Vec<f64>) -> Vec<f64>;
}

/// Forward Euler:
///
/// y_{n+1} = y_n + dt * f(t_n, y_n)
#[derive(Debug, Clone, Copy, Default)]
pub struct Euler;

impl Integrator for Euler {
    fn name(&self) -> &'static str {
        "euler"
    }

    fn step(&self, t: f64, y: &[f64], dt: f64, rhs: &dyn Fn(f64, &[f64]) -> Vec<f64>) -> Vec<f64> {
        let k1 = rhs(t, y);

        assert_eq!(
            y.len(),
            k1.len(),
            "rhs returned a vector with the wrong length"
        );

        y.iter()
            .zip(k1.iter())
            .map(|(&yi, &k1i)| yi + dt * k1i)
            .collect()
    }
}

/// Explicit midpoint rule / RK2:
///
/// k1 = f(t_n, y_n)
/// k2 = f(t_n + dt/2, y_n + dt*k1/2)
/// y_{n+1} = y_n + dt*k2
#[derive(Debug, Clone, Copy, Default)]
pub struct Midpoint;

impl Integrator for Midpoint {
    fn name(&self) -> &'static str {
        "rk2"
    }

    fn step(&self, t: f64, y: &[f64], dt: f64, rhs: &dyn Fn(f64, &[f64]) -> Vec<f64>) -> Vec<f64> {
        let k1 = rhs(t, y);

        assert_eq!(
            y.len(),
            k1.len(),
            "rhs returned a vector with the wrong length"
        );

        let y_mid: Vec<f64> = y
            .iter()
            .zip(k1.iter())
            .map(|(&yi, &k1i)| yi + 0.5 * dt * k1i)
            .collect();

        let k2 = rhs(t + 0.5 * dt, &y_mid);

        assert_eq!(
            y.len(),
            k2.len(),
            "rhs returned a vector with the wrong length"
        );

        y.iter()
            .zip(k2.iter())
            .map(|(&yi, &k2i)| yi + dt * k2i)
            .collect()
    }
}

/// Classical fourth-order Runge-Kutta:
///
/// k1 = f(t_n, y_n)
/// k2 = f(t_n + dt/2, y_n + dt*k1/2)
/// k3 = f(t_n + dt/2, y_n + dt*k2/2)
/// k4 = f(t_n + dt,   y_n + dt*k3)
///
/// y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
#[derive(Debug, Clone, Copy, Default)]
pub struct Rk4;

impl Integrator for Rk4 {
    fn name(&self) -> &'static str {
        "rk4"
    }

    fn step(&self, t: f64, y: &[f64], dt: f64, rhs: &dyn Fn(f64, &[f64]) -> Vec<f64>) -> Vec<f64> {
        let k1 = rhs(t, y);

        assert_eq!(
            y.len(),
            k1.len(),
            "rhs returned a vector with the wrong length"
        );

        let y2: Vec<f64> = y
            .iter()
            .zip(k1.iter())
            .map(|(&yi, &k1i)| yi + 0.5 * dt * k1i)
            .collect();

        let k2 = rhs(t + 0.5 * dt, &y2);

        assert_eq!(
            y.len(),
            k2.len(),
            "rhs returned a vector with the wrong length"
        );

        let y3: Vec<f64> = y
            .iter()
            .zip(k2.iter())
            .map(|(&yi, &k2i)| yi + 0.5 * dt * k2i)
            .collect();

        let k3 = rhs(t + 0.5 * dt, &y3);

        assert_eq!(
            y.len(),
            k3.len(),
            "rhs returned a vector with the wrong length"
        );

        let y4: Vec<f64> = y
            .iter()
            .zip(k3.iter())
            .map(|(&yi, &k3i)| yi + dt * k3i)
            .collect();

        let k4 = rhs(t + dt, &y4);

        assert_eq!(
            y.len(),
            k4.len(),
            "rhs returned a vector with the wrong length"
        );

        y.iter()
            .zip(k1.iter())
            .zip(k2.iter())
            .zip(k3.iter())
            .zip(k4.iter())
            .map(|((((&yi, &k1i), &k2i), &k3i), &k4i)| {
                yi + dt * (k1i + 2.0 * k2i + 2.0 * k3i + k4i) / 6.0
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exponential_rhs(_t: f64, y: &[f64]) -> Vec<f64> {
        y.to_vec()
    }

    #[test]
    fn euler_one_step() {
        let method = Euler;
        let y0 = vec![1.0];

        let y1 = method.step(0.0, &y0, 0.1, &exponential_rhs);

        assert!((y1[0] - 1.1).abs() < 1.0e-14);
    }

    #[test]
    fn midpoint_one_step() {
        let method = Midpoint;
        let y0 = vec![1.0];

        let y1 = method.step(0.0, &y0, 0.1, &exponential_rhs);

        assert!((y1[0] - 1.105).abs() < 1.0e-14);
    }

    #[test]
    fn rk4_one_step() {
        let method = Rk4;
        let y0 = vec![1.0];

        let y1 = method.step(0.0, &y0, 0.1, &exponential_rhs);

        let expected = 1.1051708333333332;

        assert!((y1[0] - expected).abs() < 1.0e-14);
    }

    #[test]
    fn integrators_work_on_vectors() {
        let rhs = |_t: f64, y: &[f64]| -> Vec<f64> { vec![y[0], -2.0 * y[1]] };

        let y0 = vec![1.0, 2.0];

        let euler = Euler.step(0.0, &y0, 0.1, &rhs);
        let midpoint = Midpoint.step(0.0, &y0, 0.1, &rhs);
        let rk4 = Rk4.step(0.0, &y0, 0.1, &rhs);

        assert_eq!(euler.len(), 2);
        assert_eq!(midpoint.len(), 2);
        assert_eq!(rk4.len(), 2);
    }
}
