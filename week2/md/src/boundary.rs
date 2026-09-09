/// Pair geometry in two dimensions. Periodic lengths are validated at construction.
#[derive(Clone, Copy, Debug)]
pub struct Boundary(Option<[f64; 2]>);

impl Boundary {
    pub const OPEN: Self = Self(None);

    pub fn periodic(lengths: [f64; 2]) -> Result<Self, String> {
        if lengths.iter().any(|x| !x.is_finite() || *x <= 0.0) {
            return Err("box lengths must be finite and positive".into());
        }
        Ok(Self(Some(lengths)))
    }

    pub fn lengths(self) -> Option<[f64; 2]> {
        self.0
    }

    pub fn wrap(self, mut x: [f64; 2]) -> [f64; 2] {
        if let Some(lengths) = self.0 {
            for axis in 0..2 {
                x[axis] = x[axis].rem_euclid(lengths[axis]);
                // rem_euclid can round a tiny negative coordinate up to L.
                if x[axis] == lengths[axis] {
                    x[axis] = 0.0;
                }
            }
        }
        x
    }

    pub fn displacement(self, x: [f64; 2], y: [f64; 2]) -> [f64; 2] {
        let mut d = [x[0] - y[0], x[1] - y[1]];
        if let Some(lengths) = self.0 {
            for axis in 0..2 {
                d[axis] -= lengths[axis] * (d[axis] / lengths[axis]).round();
            }
        }
        d
    }
}
