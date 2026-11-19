use super::mps::{Core, Mps};
use super::{make_boundary, make_site_tensor};
use std::error::Error;
use std::fs;
use std::time::Instant;

const PHYS: usize = 2;

#[derive(Clone)]
struct RowSite {
    left: usize,
    right: usize,
    data: Vec<f64>,
}

impl RowSite {
    fn new(left: usize, right: usize) -> Self {
        Self {
            left,
            right,
            data: vec![0.0; left * PHYS * PHYS * right],
        }
    }

    fn index(&self, l: usize, u: usize, d: usize, r: usize) -> usize {
        (((l * PHYS + u) * PHYS + d) * self.right) + r
    }

    fn get(&self, l: usize, u: usize, d: usize, r: usize) -> f64 {
        self.data[self.index(l, u, d, r)]
    }

    fn set(&mut self, l: usize, u: usize, d: usize, r: usize, value: f64) {
        let i = self.index(l, u, d, r);
        self.data[i] = value;
    }
}

// Construct one row tensor with open left/right boundaries.
// Axis order: left, up, down, right.
fn make_row_site(a: &super::SiteTensor, b: &[f64; 2], x: usize, l: usize) -> RowSite {
    let left_dim = if x == 0 { 1 } else { 2 };
    let right_dim = if x + 1 == l { 1 } else { 2 };

    let mut t = RowSite::new(left_dim, right_dim);

    for left in 0..left_dim {
        for up in 0..PHYS {
            for down in 0..PHYS {
                for right in 0..right_dim {
                    let mut value = 0.0;

                    for old_left in 0..2 {
                        for old_right in 0..2 {
                            let wl = if x == 0 {
                                b[old_left]
                            } else if left == old_left {
                                1.0
                            } else {
                                0.0
                            };

                            let wr = if x + 1 == l {
                                b[old_right]
                            } else if right == old_right {
                                1.0
                            } else {
                                0.0
                            };

                            value += wl * a.get(old_right, up, old_left, down) * wr;
                        }
                    }

                    t.set(left, up, down, right, value);
                }
            }
        }
    }

    t
}

// Initialize the top boundary MPS.
// Every core is reshape(b, (1,2,1)).
fn initialize_boundary(l: usize, b: &[f64; 2]) -> Mps {
    let mut cores = Vec::with_capacity(l);

    for _ in 0..l {
        let mut core = Core::new(1, 2, 1);

        for p in 0..2 {
            core.set(0, p, 0, b[p]);
        }

        cores.push(core);
    }

    Mps::new(cores).expect("Invalid initial MPS")
}

// Contract one row into the boundary MPS.
fn absorb_row(mps: &mut Mps, a: &super::SiteTensor, b: &[f64; 2], l: usize) -> Result<(), String> {
    let mut new_cores = Vec::with_capacity(l);

    for x in 0..l {
        let old = &mps.cores[x];

        let t = make_row_site(a, b, x, l);

        let mut next = Core::new(old.dl * t.left, PHYS, old.dr * t.right);

        for alpha in 0..old.dl {
            for left in 0..t.left {
                for down in 0..PHYS {
                    for beta in 0..old.dr {
                        for right in 0..t.right {
                            let mut value = 0.0;

                            for up in 0..PHYS {
                                value += old.get(alpha, up, beta) * t.get(left, up, down, right);
                            }

                            next.set(alpha * t.left + left, down, beta * t.right + right, value);
                        }
                    }
                }
            }
        }

        new_cores.push(next);
    }

    mps.cores = new_cores;

    Ok(())
}

// Normalize only the last core.
// Return the normalization factor.
fn normalize_last_core(mps: &mut Mps) -> Result<f64, String> {
    let last = mps.cores.last_mut().ok_or("Empty MPS")?;

    let norm = last.data.iter().map(|x| x * x).sum::<f64>().sqrt();

    if !norm.is_finite() || norm <= 0.0 {
        return Err("Invalid MPS normalization factor".into());
    }

    for value in &mut last.data {
        *value /= norm;
    }

    Ok(norm)
}

// Close the bottom edge by contracting every physical index with b.
fn close_bottom(mps: &Mps, b: &[f64; 2]) -> Result<f64, String> {
    let mut v = vec![1.0];

    for core in &mps.cores {
        if v.len() != core.dl {
            return Err("Bottom contraction bond mismatch".into());
        }

        let mut next = vec![0.0; core.dr];

        for alpha in 0..core.dl {
            for p in 0..PHYS {
                for beta in 0..core.dr {
                    next[beta] += v[alpha] * core.get(alpha, p, beta) * b[p];
                }
            }
        }

        v = next;
    }

    if v.len() != 1 {
        return Err("Final MPS boundary is not scalar".into());
    }

    let z = v[0];

    if !z.is_finite() || z <= 0.0 {
        return Err(format!("Invalid final partition scalar: {}", z));
    }

    Ok(z)
}

struct BoundaryResult {
    log_z: f64,
    g: f64,
    scales: Vec<f64>,
    diagnostics: Vec<(usize, usize, usize, usize)>,
    mps: Mps,
    elapsed: f64,
}

// Algorithm 4: Boundary-MPS contraction of an open square.
fn contract_boundary(l: usize, k: f64, chi: usize) -> Result<BoundaryResult, Box<dyn Error>> {
    if l < 2 {
        return Err("L must be at least 2".into());
    }

    if chi == 0 {
        return Err("chi must be at least 1".into());
    }

    if !k.is_finite() || k < 0.0 {
        return Err("K must be finite and non-negative".into());
    }

    let start = Instant::now();

    let a = make_site_tensor(k);
    let b = make_boundary(k);

    let mut mps = initialize_boundary(l, &b);

    let mut log_scale = 0.0;
    let mut scales = Vec::with_capacity(l);

    // (row, bond, dimension_before, dimension_after)
    let mut diagnostics = Vec::new();

    for row in 0..l {
        // Absorb one complete row.
        absorb_row(&mut mps, &a, &b, l)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Record enlarged bond dimensions before compression.
        let before: Vec<usize> = mps.cores.iter().map(|c| c.dr).collect();

        // Canonical SVD compression.
        mps.compress(chi).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("MPS compression failed: {}", e),
            )
        })?;

        let after: Vec<usize> = mps.cores.iter().map(|c| c.dr).collect();

        for bond in 0..l - 1 {
            diagnostics.push((row + 1, bond + 1, before[bond], after[bond]));
        }

        // Normalize the last core only.
        let c = normalize_last_core(&mut mps)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        scales.push(c);
        log_scale += c.ln();
    }

    // Close the bottom boundary.
    let final_scalar =
        close_bottom(&mps, &b).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    let log_z = log_scale + final_scalar.ln();

    let g = log_z / ((l * l) as f64);

    let elapsed = start.elapsed().as_secs_f64();

    Ok(BoundaryResult {
        log_z,
        g,
        scales,
        diagnostics,
        mps,
        elapsed,
    })
}

// Independent exact enumeration for small open squares.
fn exact_enumeration(l: usize, k: f64) -> f64 {
    let n = l * l;
    let total_states = 1usize << n;

    let mut exponents = Vec::with_capacity(total_states);

    for state in 0..total_states {
        let spin = |x: usize| -> f64 {
            if (state >> x) & 1 == 0 {
                1.0
            } else {
                -1.0
            }
        };

        let mut energy_sum = 0.0;

        for y in 0..l {
            for x in 0..l {
                let i = y * l + x;

                if x + 1 < l {
                    energy_sum += spin(i) * spin(i + 1);
                }

                if y + 1 < l {
                    energy_sum += spin(i) * spin(i + l);
                }
            }
        }

        exponents.push(k * energy_sum);
    }

    let max_value = exponents.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let sum = exponents.iter().map(|x| (x - max_value).exp()).sum::<f64>();

    max_value + sum.ln()
}

// Parse a required CLI argument.
fn get_arg(args: &[String], name: &str) -> Result<String, Box<dyn Error>> {
    let pos = args.iter().position(|x| x == name).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Missing required argument {}", name),
        )
    })?;

    let value = args.get(pos + 1).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Missing value for {}", name),
        )
    })?;

    Ok(value.clone())
}

// Save the complete numerical evidence.
fn save_evidence(
    out: &str,
    l: usize,
    k: f64,
    chi: usize,
    result: &BoundaryResult,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(out)?;

    let summary = format!(
        "Model: 2D zero-field Ising model\n\
         Geometry: open square\n\
         L: {}\n\
         K: {:.17e}\n\
         chi: {}\n\
         log_Z: {:.17e}\n\
         g: {:.17e}\n\
         solver_seconds: {:.9}\n",
        l, k, chi, result.log_z, result.g, result.elapsed
    );

    fs::write(format!("{}/summary.txt", out), summary)?;

    let mut scale_text = String::from("row,normalization_factor,log_factor\n");

    for (i, c) in result.scales.iter().enumerate() {
        scale_text.push_str(&format!("{},{:.17e},{:.17e}\n", i + 1, c, c.ln()));
    }

    fs::write(format!("{}/normalization_scales.csv", out), scale_text)?;

    let mut diagnostic_text = String::from("row,bond,dimension_before,dimension_after\n");

    for (row, bond, before, after) in &result.diagnostics {
        diagnostic_text.push_str(&format!("{},{},{},{}\n", row, bond, before, after));
    }

    fs::write(
        format!("{}/truncation_diagnostics.csv", out),
        diagnostic_text,
    )?;

    let mut tensor_text = String::new();

    for (i, core) in result.mps.cores.iter().enumerate() {
        tensor_text.push_str(&format!(
            "CORE {}\nDL={} PHYS={} DR={}\n",
            i + 1,
            core.dl,
            core.phys,
            core.dr
        ));

        for value in &core.data {
            tensor_text.push_str(&format!("{:.17e}\n", value));
        }
    }

    fs::write(format!("{}/final_mps.txt", out), tensor_text)?;

    Ok(())
}

// CLI entry point.
pub fn run_cli() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();

    let method = get_arg(&args, "--method")?;

    if method != "boundary" {
        return Err("Only method=boundary is currently supported".into());
    }

    let l: usize = get_arg(&args, "--l")?.parse()?;
    let k: f64 = get_arg(&args, "--k")?.parse()?;
    let chi: usize = get_arg(&args, "--chi")?.parse()?;
    let out = get_arg(&args, "--out")?;

    println!("==========================================");
    println!("Week 10 - Boundary MPS Contraction");
    println!("==========================================");

    println!("Boundary condition: open");
    println!("L = {}", l);
    println!("K = {:.8}", k);
    println!("T = {:.8}", 1.0 / k);
    println!("chi = {}", chi);
    println!("Output = {}", out);

    println!("\nStarting row-by-row contraction...");

    let result = contract_boundary(l, k, chi)?;

    println!("\n==========================================");
    println!("Boundary MPS result");
    println!("==========================================");

    println!("log(Z) = {:.15e}", result.log_z);
    println!("g      = {:.15e}", result.g);
    println!("Time   = {:.6} seconds", result.elapsed);

    println!("\nNormalization factors:");

    for (i, c) in result.scales.iter().enumerate() {
        println!("Row {:>3}: {:.12e}", i + 1, c);
    }

    println!("\nSaving evidence...");

    save_evidence(&out, l, k, chi, &result)?;

    println!("Evidence saved to {}", out);

    println!("\nBoundary MPS completed successfully.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_enumeration_at_zero() {
        let log_z = exact_enumeration(4, 0.0);

        let expected = 16.0 * 2.0_f64.ln();

        assert!((log_z - expected).abs() < 1e-12);
    }

    #[test]
    fn test_boundary_mps_against_enumeration() {
        for k in [0.0, 0.44, 0.7] {
            let result = contract_boundary(4, k, 8).unwrap();

            let exact_log_z = exact_enumeration(4, k);
            let exact_g = exact_log_z / 16.0;

            let error = (result.g - exact_g).abs();

            println!(
                "K={:.3}, MPS={:.12e}, exact={:.12e}, error={:.4e}",
                k, result.g, exact_g, error
            );

            assert!(
                error < 1e-8,
                "Boundary MPS disagrees with exact enumeration at K={}",
                k
            );
        }
    }

    #[test]
    fn test_open_boundary_k_zero() {
        let result = contract_boundary(4, 0.0, 8).unwrap();

        let expected = 2.0_f64.ln();

        assert!((result.g - expected).abs() < 1e-10);
    }

    #[test]
    fn test_final_bond_dimensions() {
        let result = contract_boundary(4, 0.44, 2).unwrap();

        for core in &result.mps.cores {
            assert!(core.dl <= 2);
            assert!(core.dr <= 2);
        }
    }
}
