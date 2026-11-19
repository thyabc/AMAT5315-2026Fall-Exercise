#[path = "thermo/boundary.rs"]
mod boundary;
#[path = "thermo/mps.rs"]
mod mps;
#[path = "thermo/trg.rs"]
mod trg;
use std::env;
use std::error::Error;

const D: usize = 2;

// Four-leg Ising site tensor.
// Axis order: r, u, l, d.
#[derive(Clone, Debug)]
struct SiteTensor {
    data: Vec<f64>,
}

impl SiteTensor {
    fn new() -> Self {
        Self {
            data: vec![0.0; D * D * D * D],
        }
    }

    fn index(r: usize, u: usize, l: usize, d: usize) -> usize {
        (((r * D + u) * D + l) * D) + d
    }

    fn get(&self, r: usize, u: usize, l: usize, d: usize) -> f64 {
        self.data[Self::index(r, u, l, d)]
    }

    fn set(&mut self, r: usize, u: usize, l: usize, d: usize, value: f64) {
        let idx = Self::index(r, u, l, d);
        self.data[idx] = value;
    }
}

// Construct W from the square-root SVD of the bond matrix.
fn make_w(k: f64) -> [[f64; D]; D] {
    let c = k.cosh().sqrt();
    let s = k.sinh().sqrt();

    [[c, s], [c, -s]]
}

// Construct the open-boundary vector.
fn make_boundary(k: f64) -> [f64; D] {
    [1.0 / k.cosh().sqrt(), 0.0]
}

// Construct the four-leg site tensor:
//
// A[r,u,l,d] = sum_s W[s,r] W[s,u] W[s,l] W[s,d]
//
fn make_site_tensor(k: f64) -> SiteTensor {
    let w = make_w(k);
    let mut a = SiteTensor::new();

    for r in 0..D {
        for u in 0..D {
            for l in 0..D {
                for d in 0..D {
                    let mut value = 0.0;

                    for spin in 0..D {
                        value += w[spin][r] * w[spin][u] * w[spin][l] * w[spin][d];
                    }

                    a.set(r, u, l, d, value);
                }
            }
        }
    }

    a
}

// Verify that W W^T reproduces the original Ising bond weight.
fn verify_bond(k: f64) -> f64 {
    let w = make_w(k);

    let mut max_error: f64 = 0.0;

    for s in 0..D {
        for t in 0..D {
            let mut reconstructed = 0.0;

            for a in 0..D {
                reconstructed += w[s][a] * w[t][a];
            }

            let spin_s = if s == 0 { 1.0 } else { -1.0 };
            let spin_t = if t == 0 { 1.0 } else { -1.0 };

            let exact = (k * spin_s * spin_t).exp();

            let error = (reconstructed - exact).abs();

            max_error = max_error.max(error);
        }
    }

    max_error
}

// Verify W b = [1,1]^T for a missing boundary bond.
fn verify_boundary(k: f64) -> f64 {
    let w = make_w(k);
    let b = make_boundary(k);

    let mut max_error: f64 = 0.0;

    for s in 0..D {
        let value = w[s][0] * b[0] + w[s][1] * b[1];

        max_error = max_error.max((value - 1.0).abs());
    }

    max_error
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli_args: Vec<String> = std::env::args().collect();

    if let Some(pos) = cli_args.iter().position(|arg| arg == "--method") {
        let method = cli_args
            .get(pos + 1)
            .ok_or("Missing value after --method")?;

        return match method.as_str() {
            "boundary" => boundary::run_cli(),
            "trg" => trg::run_cli(),
            other => Err(format!("Unsupported method: {}", other).into()),
        };
    }

    let args: Vec<String> = env::args().collect();

    let mut k: f64 = 0.45;

    if let Some(pos) = args.iter().position(|x| x == "--k") {
        if pos + 1 >= args.len() {
            return Err("Missing value after --k".into());
        }

        k = args[pos + 1].parse()?;
    }

    if !k.is_finite() || k < 0.0 {
        return Err("K must be finite and non-negative".into());
    }

    println!("======================================");
    println!("Week 10 - Part 2: Boundary MPS");
    println!("======================================");

    println!("\n[1] Ising model parameters");
    println!("K = {:.6}", k);
    println!("T = {:.6}", 1.0 / k);

    println!("\n[2] Constructing bond matrix W");

    let w = make_w(k);

    for row in w {
        println!("[{:12.8}, {:12.8}]", row[0], row[1]);
    }

    println!("\n[3] Constructing site tensor A");

    let a = make_site_tensor(k);

    println!("Tensor dimensions: 2 x 2 x 2 x 2");
    println!("Number of entries: {}", a.data.len());

    println!("\nSelected tensor entries:");

    println!("A[0,0,0,0] = {:.12}", a.get(0, 0, 0, 0));

    println!("A[0,0,0,1] = {:.12}", a.get(0, 0, 0, 1));

    println!("A[0,1,0,1] = {:.12}", a.get(0, 1, 0, 1));

    println!("\n[4] Verifying bond factorization");

    let bond_error = verify_bond(k);

    println!("Maximum bond reconstruction error = {:.4e}", bond_error);

    println!("\n[5] Verifying open boundary vector");

    let boundary_error = verify_boundary(k);

    println!("Maximum boundary error = {:.4e}", boundary_error);

    if bond_error > 1e-12 {
        return Err("Bond factorization verification failed".into());
    }

    if boundary_error > 1e-12 {
        return Err("Boundary vector verification failed".into());
    }

    println!("\n======================================");
    println!("Local tensor construction passed!");
    println!("======================================");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bond_factorization() {
        for k in [0.0, 0.2, 0.45, 0.7] {
            assert!(verify_bond(k) < 1e-12);
        }
    }

    #[test]
    fn test_boundary_vector() {
        for k in [0.0, 0.2, 0.45, 0.7] {
            assert!(verify_boundary(k) < 1e-12);
        }
    }

    #[test]
    fn test_site_tensor_dimensions() {
        let a = make_site_tensor(0.45);

        assert_eq!(a.data.len(), 16);

        assert!(a.data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_site_tensor_leg_permutation_symmetry() {
        let a = make_site_tensor(0.45);

        for r in 0..D {
            for u in 0..D {
                for l in 0..D {
                    for d in 0..D {
                        let original = a.get(r, u, l, d);

                        // Swapping any adjacent pair of legs
                        // must leave the tensor unchanged.
                        let swap_ru = a.get(u, r, l, d);
                        let swap_ul = a.get(r, l, u, d);
                        let swap_ld = a.get(r, u, d, l);

                        assert!((original - swap_ru).abs() < 1e-12);
                        assert!((original - swap_ul).abs() < 1e-12);
                        assert!((original - swap_ld).abs() < 1e-12);
                    }
                }
            }
        }
    }
}
