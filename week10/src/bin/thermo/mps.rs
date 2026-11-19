use nalgebra::DMatrix;

// A three-leg MPS core.
// Axis order: left bond, physical index, right bond.
#[derive(Clone, Debug)]
pub struct Core {
    pub dl: usize,
    pub phys: usize,
    pub dr: usize,
    pub data: Vec<f64>,
}

impl Core {
    pub fn new(dl: usize, phys: usize, dr: usize) -> Self {
        Self {
            dl,
            phys,
            dr,
            data: vec![0.0; dl * phys * dr],
        }
    }

    fn index(&self, a: usize, p: usize, b: usize) -> usize {
        (a * self.phys + p) * self.dr + b
    }

    pub fn get(&self, a: usize, p: usize, b: usize) -> f64 {
        self.data[self.index(a, p, b)]
    }

    pub fn set(&mut self, a: usize, p: usize, b: usize, value: f64) {
        let idx = self.index(a, p, b);
        self.data[idx] = value;
    }
}

#[derive(Clone, Debug)]
pub struct Mps {
    pub cores: Vec<Core>,
}

impl Mps {
    pub fn new(cores: Vec<Core>) -> Result<Self, String> {
        if cores.is_empty() {
            return Err("MPS must contain at least one core".into());
        }

        if cores[0].dl != 1 {
            return Err("First MPS left bond must have dimension 1".into());
        }

        if cores.last().unwrap().dr != 1 {
            return Err("Last MPS right bond must have dimension 1".into());
        }

        for i in 0..cores.len() - 1 {
            if cores[i].dr != cores[i + 1].dl {
                return Err(format!("Bond mismatch between cores {} and {}", i, i + 1));
            }
        }

        Ok(Self { cores })
    }

    pub fn len(&self) -> usize {
        self.cores.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cores.is_empty()
    }

    // Convert the MPS to a dense vector.
    // This is intended for small-system verification.
    pub fn to_dense(&self) -> Vec<f64> {
        let l = self.len();

        let mut result = Vec::new();

        for state in 0..(1usize << l) {
            let mut current = vec![1.0];

            for i in 0..l {
                let p = (state >> (l - 1 - i)) & 1;
                let core = &self.cores[i];

                let mut next = vec![0.0; core.dr];

                for a in 0..core.dl {
                    for b in 0..core.dr {
                        next[b] += current[a] * core.get(a, p, b);
                    }
                }

                current = next;
            }

            result.push(current[0]);
        }

        result
    }

    // Algorithm 3:
    // 1. Right-to-left exact SVD sweep.
    // 2. Left-to-right truncated SVD sweep.
    //
    // After compression, the first L-1 cores are left-isometric.
    pub fn compress(&mut self, chi: usize) -> Result<(), String> {
        if chi == 0 {
            return Err("Bond dimension chi must be >= 1".into());
        }

        let l = self.len();

        if l <= 1 {
            return Ok(());
        }

        // -------------------------------------------------
        // Sweep 1: right to left, without truncation.
        // -------------------------------------------------
        for i in (1..l).rev() {
            let core = self.cores[i].clone();

            let dl = core.dl;
            let phys = core.phys;
            let dr = core.dr;

            // Reshape B[a,p,b] into M[a,(p,b)].
            let mut matrix = DMatrix::<f64>::zeros(dl, phys * dr);

            for a in 0..dl {
                for p in 0..phys {
                    for b in 0..dr {
                        matrix[(a, p * dr + b)] = core.get(a, p, b);
                    }
                }
            }

            let svd = matrix.svd(true, true);

            let u = svd.u.ok_or("Missing U in right sweep")?;
            let vt = svd.v_t.ok_or("Missing Vt in right sweep")?;

            let singular = svd.singular_values;
            let rank = singular.len();

            // The current core becomes Vt.
            let mut new_core = Core::new(rank, phys, dr);

            for q in 0..rank {
                for p in 0..phys {
                    for b in 0..dr {
                        new_core.set(q, p, b, vt[(q, p * dr + b)]);
                    }
                }
            }

            // Pass U*S to the previous core.
            let previous = self.cores[i - 1].clone();

            let mut new_previous = Core::new(previous.dl, previous.phys, rank);

            for a in 0..previous.dl {
                for p in 0..previous.phys {
                    for q in 0..rank {
                        let mut value = 0.0;

                        for gamma in 0..dl {
                            value += previous.get(a, p, gamma) * u[(gamma, q)] * singular[q];
                        }

                        new_previous.set(a, p, q, value);
                    }
                }
            }

            self.cores[i] = new_core;
            self.cores[i - 1] = new_previous;
        }

        // -------------------------------------------------
        // Sweep 2: left to right, truncating to chi.
        // -------------------------------------------------
        for i in 0..l - 1 {
            let core = self.cores[i].clone();

            let dl = core.dl;
            let phys = core.phys;
            let dr = core.dr;

            // Reshape B[a,p,b] into M[(a,p),b].
            let mut matrix = DMatrix::<f64>::zeros(dl * phys, dr);

            for a in 0..dl {
                for p in 0..phys {
                    for b in 0..dr {
                        matrix[(a * phys + p, b)] = core.get(a, p, b);
                    }
                }
            }

            let svd = matrix.svd(true, true);

            let u = svd.u.ok_or("Missing U in left sweep")?;
            let vt = svd.v_t.ok_or("Missing Vt in left sweep")?;

            let singular = svd.singular_values;

            let rank = chi.min(singular.len());

            // Current core becomes the retained columns of U.
            let mut new_core = Core::new(dl, phys, rank);

            for a in 0..dl {
                for p in 0..phys {
                    for q in 0..rank {
                        new_core.set(a, p, q, u[(a * phys + p, q)]);
                    }
                }
            }

            // Pass S*Vt to the next core.
            let next = self.cores[i + 1].clone();

            let mut new_next = Core::new(rank, next.phys, next.dr);

            for q in 0..rank {
                for p in 0..next.phys {
                    for b in 0..next.dr {
                        let mut value = 0.0;

                        for gamma in 0..dr {
                            value += singular[q] * vt[(q, gamma)] * next.get(gamma, p, b);
                        }

                        new_next.set(q, p, b, value);
                    }
                }
            }

            self.cores[i] = new_core;
            self.cores[i + 1] = new_next;
        }

        Ok(())
    }

    // Verify left-isometry of the first L-1 cores.
    pub fn max_left_isometry_error(&self) -> f64 {
        let mut max_error: f64 = 0.0;

        if self.len() <= 1 {
            return max_error;
        }

        for core in self.cores.iter().take(self.len() - 1) {
            let rows = core.dl * core.phys;
            let cols = core.dr;

            for i in 0..cols {
                for j in 0..cols {
                    let mut value = 0.0;

                    for row in 0..rows {
                        let a = row / core.phys;
                        let p = row % core.phys;

                        value += core.get(a, p, i) * core.get(a, p, j);
                    }

                    let expected = if i == j { 1.0 } else { 0.0 };

                    max_error = max_error.max((value - expected).abs());
                }
            }
        }

        max_error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_mps() -> Mps {
        let mut a = Core::new(1, 2, 3);
        let mut b = Core::new(3, 2, 2);
        let mut c = Core::new(2, 2, 1);

        for x in 0..a.dl {
            for p in 0..a.phys {
                for y in 0..a.dr {
                    a.set(x, p, y, ((1 + p + 2 * y) as f64) * 0.17);
                }
            }
        }

        for x in 0..b.dl {
            for p in 0..b.phys {
                for y in 0..b.dr {
                    b.set(x, p, y, ((2 + 3 * x + p + y) as f64) * 0.11);
                }
            }
        }

        for x in 0..c.dl {
            for p in 0..c.phys {
                for y in 0..c.dr {
                    c.set(x, p, y, ((1 + 2 * x + p) as f64) * 0.13);
                }
            }
        }

        Mps::new(vec![a, b, c]).unwrap()
    }

    fn relative_dense_error(a: &[f64], b: &[f64]) -> f64 {
        let numerator = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt();

        let denominator = a.iter().map(|x| x * x).sum::<f64>().sqrt();

        numerator / denominator
    }

    #[test]
    fn test_dense_conversion() {
        let mps = sample_mps();
        let dense = mps.to_dense();

        assert_eq!(dense.len(), 8);
        assert!(dense.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_exact_compression() {
        let mut mps = sample_mps();

        let before = mps.to_dense();

        mps.compress(16).unwrap();

        let after = mps.to_dense();

        assert!(relative_dense_error(&before, &after) < 1e-10);
    }

    #[test]
    fn test_bond_dimension_cap() {
        let mut mps = sample_mps();

        mps.compress(1).unwrap();

        for core in &mps.cores {
            assert!(core.dr <= 1);
        }
    }

    #[test]
    fn test_left_isometry() {
        let mut mps = sample_mps();

        mps.compress(16).unwrap();

        assert!(mps.max_left_isometry_error() < 1e-10);
    }
}
