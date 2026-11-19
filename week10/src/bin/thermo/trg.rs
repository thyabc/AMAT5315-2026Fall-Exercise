use super::{make_site_tensor, SiteTensor};
use nalgebra::DMatrix;
use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind, Write};
use std::time::Instant;

#[derive(Clone, Debug)]
struct Tensor4 {
    d: usize,
    data: Vec<f64>,
}

impl Tensor4 {
    fn from_site_tensor(a: &SiteTensor) -> Self {
        Self {
            d: 2,
            data: a.data.clone(),
        }
    }

    fn zeros(d: usize) -> Self {
        Self {
            d,
            data: vec![0.0; d * d * d * d],
        }
    }

    #[inline]
    fn index(&self, r: usize, u: usize, l: usize, down: usize) -> usize {
        (((r * self.d + u) * self.d + l) * self.d) + down
    }

    #[inline]
    fn get(&self, r: usize, u: usize, l: usize, down: usize) -> f64 {
        self.data[self.index(r, u, l, down)]
    }

    #[inline]
    fn set(&mut self, r: usize, u: usize, l: usize, down: usize, value: f64) {
        let i = self.index(r, u, l, down);
        self.data[i] = value;
    }

    fn max_abs(&self) -> f64 {
        self.data.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
    }
}

struct SplitFactors {
    rank: usize,
    // S1(l,u,q), S2(d,l,q), S3(q,r,d), S4(q,u,r)
    s1: Vec<f64>,
    s2: Vec<f64>,
    s3: Vec<f64>,
    s4: Vec<f64>,
    singular_1: Vec<f64>,
    singular_2: Vec<f64>,
}

impl SplitFactors {
    #[inline]
    fn s1(&self, d: usize, l: usize, u: usize, q: usize) -> f64 {
        self.s1[(l * d + u) * self.rank + q]
    }

    #[inline]
    fn s2(&self, d: usize, down: usize, l: usize, q: usize) -> f64 {
        self.s2[(down * d + l) * self.rank + q]
    }

    #[inline]
    fn s3(&self, d: usize, q: usize, r: usize, down: usize) -> f64 {
        self.s3[(q * d + r) * d + down]
    }

    #[inline]
    fn s4(&self, d: usize, q: usize, u: usize, r: usize) -> f64 {
        self.s4[(q * d + u) * d + r]
    }
}

// Split A across (l,u)|(r,d), retaining at most chi singular directions.
// The singular values are divided symmetrically as sqrt(S) on both sides.
fn split_first(a: &Tensor4, chi: usize) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), Box<dyn Error>> {
    let d = a.d;
    let mut matrix = DMatrix::<f64>::zeros(d * d, d * d);

    for l in 0..d {
        for u in 0..d {
            for r in 0..d {
                for down in 0..d {
                    matrix[(l * d + u, r * d + down)] = a.get(r, u, l, down);
                }
            }
        }
    }

    let svd = matrix.svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| IoError::new(ErrorKind::Other, "TRG first split did not produce U"))?;
    let vt = svd
        .v_t
        .ok_or_else(|| IoError::new(ErrorKind::Other, "TRG first split did not produce Vt"))?;

    let singular_all: Vec<f64> = svd.singular_values.iter().copied().collect();
    let rank = chi.min(singular_all.len());

    let mut s1 = vec![0.0; d * d * rank];
    let mut s3 = vec![0.0; rank * d * d];

    for l in 0..d {
        for up in 0..d {
            let row = l * d + up;
            for q in 0..rank {
                s1[(l * d + up) * rank + q] = u[(row, q)] * singular_all[q].sqrt();
            }
        }
    }

    for q in 0..rank {
        let root = singular_all[q].sqrt();
        for r in 0..d {
            for down in 0..d {
                let col = r * d + down;
                s3[(q * d + r) * d + down] = root * vt[(q, col)];
            }
        }
    }

    Ok((s1, s3, singular_all))
}

// Split A across (d,l)|(u,r), retaining at most chi singular directions.
fn split_second(a: &Tensor4, chi: usize) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), Box<dyn Error>> {
    let d = a.d;
    let mut matrix = DMatrix::<f64>::zeros(d * d, d * d);

    for down in 0..d {
        for l in 0..d {
            for u in 0..d {
                for r in 0..d {
                    matrix[(down * d + l, u * d + r)] = a.get(r, u, l, down);
                }
            }
        }
    }

    let svd = matrix.svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| IoError::new(ErrorKind::Other, "TRG second split did not produce U"))?;
    let vt = svd
        .v_t
        .ok_or_else(|| IoError::new(ErrorKind::Other, "TRG second split did not produce Vt"))?;

    let singular_all: Vec<f64> = svd.singular_values.iter().copied().collect();
    let rank = chi.min(singular_all.len());

    let mut s2 = vec![0.0; d * d * rank];
    let mut s4 = vec![0.0; rank * d * d];

    for down in 0..d {
        for l in 0..d {
            let row = down * d + l;
            for q in 0..rank {
                s2[(down * d + l) * rank + q] = u[(row, q)] * singular_all[q].sqrt();
            }
        }
    }

    for q in 0..rank {
        let root = singular_all[q].sqrt();
        for up in 0..d {
            for r in 0..d {
                let col = up * d + r;
                s4[(q * d + up) * d + r] = root * vt[(q, col)];
            }
        }
    }

    Ok((s2, s4, singular_all))
}

fn split_tensor(a: &Tensor4, chi: usize) -> Result<SplitFactors, Box<dyn Error>> {
    let (s1, s3, singular_1) = split_first(a, chi)?;
    let (s2, s4, singular_2) = split_second(a, chi)?;

    let rank_1 = chi.min(singular_1.len());
    let rank_2 = chi.min(singular_2.len());

    if rank_1 != rank_2 {
        return Err("The two TRG diagonal splits returned different ranks".into());
    }

    Ok(SplitFactors {
        rank: rank_1,
        s1,
        s2,
        s3,
        s4,
        singular_1,
        singular_2,
    })
}

// Regroup four half-tensors according to Algorithm 5:
//
// A'[r,u,l,d] = sum_{w,a,b,gamma}
//   S1(w,a,r) S2(a,b,u) S3(l,b,gamma) S4(d,gamma,w).
//
// The contraction is evaluated as X * Y, where the matrix dimensions
// are rank^2 x old_dim^2 and old_dim^2 x rank^2.
fn regroup(a: &Tensor4, f: &SplitFactors) -> Result<Tensor4, Box<dyn Error>> {
    let d = a.d;
    let q = f.rank;

    let mut x = DMatrix::<f64>::zeros(q * q, d * d);
    for r in 0..q {
        for u in 0..q {
            for w in 0..d {
                for b in 0..d {
                    let mut value = 0.0;
                    for aa in 0..d {
                        value += f.s1(d, w, aa, r) * f.s2(d, aa, b, u);
                    }
                    x[(r * q + u, w * d + b)] = value;
                }
            }
        }
    }

    let mut y = DMatrix::<f64>::zeros(d * d, q * q);
    for w in 0..d {
        for b in 0..d {
            for l in 0..q {
                for down in 0..q {
                    let mut value = 0.0;
                    for gamma in 0..d {
                        value += f.s3(d, l, b, gamma) * f.s4(d, down, gamma, w);
                    }
                    y[(w * d + b, l * q + down)] = value;
                }
            }
        }
    }

    let product = x * y;
    let mut next = Tensor4::zeros(q);

    for r in 0..q {
        for u in 0..q {
            for l in 0..q {
                for down in 0..q {
                    next.set(r, u, l, down, product[(r * q + u, l * q + down)]);
                }
            }
        }
    }

    if next.data.iter().any(|x| !x.is_finite()) {
        return Err("Non-finite value generated during TRG regrouping".into());
    }

    Ok(next)
}

struct TrgResult {
    log_z: f64,
    g: f64,
    z: f64,
    elapsed: f64,
    steps: usize,
    final_dim: usize,
}

// Algorithm 5: periodic square-lattice TRG contraction.
fn contract_trg(l: usize, k: f64, chi: usize, out: &str) -> Result<TrgResult, Box<dyn Error>> {
    if l < 2 || !l.is_power_of_two() {
        return Err("TRG requires L >= 2 and a power-of-two side length".into());
    }
    if chi == 0 {
        return Err("chi must be at least 1".into());
    }
    if !k.is_finite() || k < 0.0 {
        return Err("K must be finite and non-negative".into());
    }

    let start = Instant::now();
    fs::create_dir_all(out)?;

    let site = make_site_tensor(k);
    let mut a = Tensor4::from_site_tensor(&site);

    let mut n_sites = l * l;
    let steps = 2 * l.trailing_zeros() as usize;
    let mut ell = 0.0_f64;

    let mut step_summary =
        String::from("step,sites_before,bond_dim_before,rank_after,scale,log_scale_contribution\n");

    for step in 0..steps {
        let scale = a.max_abs();
        if !scale.is_finite() || scale <= 0.0 {
            return Err(format!("Invalid normalization scale at TRG step {}", step + 1).into());
        }

        for value in &mut a.data {
            *value /= scale;
        }

        let log_contribution = (n_sites as f64) * scale.ln();
        ell += log_contribution;

        let factors = split_tensor(&a, chi)?;
        let old_dim = a.d;
        let rank = factors.rank;

        let step_dir = format!("{}/step_{:02}", out, step + 1);
        fs::create_dir_all(&step_dir)?;

        // Save all four split factors in a documented binary order:
        // S1, then S2, then S3, then S4; each f64 is little-endian.
        let mut factor_writer = fs::File::create(format!("{}/split_factors.bin", step_dir))?;
        for values in [&factors.s1, &factors.s2, &factors.s3, &factors.s4] {
            for value in values {
                factor_writer.write_all(&value.to_le_bytes())?;
            }
        }
        factor_writer.flush()?;

        let mut singular_1 = String::from("index,singular_value\n");
        for (i, value) in factors.singular_1.iter().enumerate() {
            singular_1.push_str(&format!("{},{:.17e}\n", i, value));
        }
        fs::write(format!("{}/singular_values_cut1.csv", step_dir), singular_1)?;

        let mut singular_2 = String::from("index,singular_value\n");
        for (i, value) in factors.singular_2.iter().enumerate() {
            singular_2.push_str(&format!("{},{:.17e}\n", i, value));
        }
        fs::write(format!("{}/singular_values_cut2.csv", step_dir), singular_2)?;

        let factor_meta = format!(
            "step: {}\n\
             sites_before: {}\n\
             old_bond_dimension: {}\n\
             retained_rank: {}\n\
             S1_shape: [{},{},{}] axis_order=(l,u,q)\n\
             S2_shape: [{},{},{}] axis_order=(d,l,q)\n\
             S3_shape: [{},{},{}] axis_order=(q,r,d)\n\
             S4_shape: [{},{},{}] axis_order=(q,u,r)\n\
             split_factors.bin_order: S1,S2,S3,S4; little-endian f64\n\
             normalization_scale: {:.17e}\n\
             log_scale_contribution: {:.17e}\n",
            step + 1,
            n_sites,
            old_dim,
            rank,
            old_dim,
            old_dim,
            rank,
            old_dim,
            old_dim,
            rank,
            rank,
            old_dim,
            old_dim,
            rank,
            old_dim,
            old_dim,
            scale,
            log_contribution
        );
        fs::write(format!("{}/metadata.txt", step_dir), factor_meta)?;

        println!(
            "TRG step {:>2}/{:>2}: sites={}, D={} -> chi={}, scale={:.6e}",
            step + 1,
            steps,
            n_sites,
            old_dim,
            rank,
            scale
        );

        a = regroup(&a, &factors)?;
        n_sites /= 2;

        step_summary.push_str(&format!(
            "{},{},{},{},{:.17e},{:.17e}\n",
            step + 1,
            n_sites * 2,
            old_dim,
            rank,
            scale,
            log_contribution
        ));
    }

    if n_sites != 1 {
        return Err(format!("TRG ended with {} tensors instead of one", n_sites).into());
    }

    // Periodic closure: z = sum_{i,j} A[i,j,i,j].
    // The two sums are independent; do not set all four indices equal.
    let mut z = 0.0_f64;
    for i in 0..a.d {
        for j in 0..a.d {
            z += a.get(i, j, i, j);
        }
    }

    if !z.is_finite() || z <= 0.0 {
        return Err(format!("Invalid periodic closure scalar z={}", z).into());
    }

    let log_z = ell + z.ln();
    let g = log_z / ((l * l) as f64);
    let elapsed = start.elapsed().as_secs_f64();

    fs::write(format!("{}/step_scales.csv", out), step_summary)?;

    let mut final_writer = fs::File::create(format!("{}/final_tensor.bin", out))?;
    for value in &a.data {
        final_writer.write_all(&value.to_le_bytes())?;
    }
    final_writer.flush()?;

    let summary = format!(
        "Model: 2D zero-field Ising model\n\
         Geometry: periodic square (torus)\n\
         L: {}\n\
         K: {:.17e}\n\
         T: {:.17e}\n\
         chi: {}\n\
         steps: {}\n\
         final_tensor_dimension: {}\n\
         normalization_log_scale: {:.17e}\n\
         periodic_closure_z: {:.17e}\n\
         log_Z: {:.17e}\n\
         g: {:.17e}\n\
         solver_seconds: {:.9}\n",
        l,
        k,
        1.0 / k,
        chi,
        steps,
        a.d,
        ell,
        z,
        log_z,
        g,
        elapsed
    );
    fs::write(format!("{}/summary.txt", out), summary)?;

    Ok(TrgResult {
        log_z,
        g,
        z,
        elapsed,
        steps,
        final_dim: a.d,
    })
}

fn get_arg(args: &[String], name: &str) -> Result<String, Box<dyn Error>> {
    let pos = args.iter().position(|x| x == name).ok_or_else(|| {
        IoError::new(
            ErrorKind::InvalidInput,
            format!("Missing required argument {}", name),
        )
    })?;

    args.get(pos + 1).cloned().ok_or_else(|| {
        IoError::new(
            ErrorKind::InvalidInput,
            format!("Missing value for {}", name),
        )
        .into()
    })
}

// Periodic exact enumeration. Each site contributes one bond in each
// positive direction; for L=2 this deliberately counts parallel bonds.
fn exact_periodic_log_z(l: usize, k: f64) -> f64 {
    let n = l * l;
    let total = 1usize << n;
    let mut exponents = Vec::with_capacity(total);

    for state in 0..total {
        let spin = |i: usize| -> f64 {
            if (state >> i) & 1 == 0 {
                1.0
            } else {
                -1.0
            }
        };

        let mut bond_sum = 0.0;
        for y in 0..l {
            for x in 0..l {
                let i = y * l + x;
                let right = y * l + (x + 1) % l;
                let down = ((y + 1) % l) * l + x;

                bond_sum += spin(i) * spin(right);
                bond_sum += spin(i) * spin(down);
            }
        }
        exponents.push(k * bond_sum);
    }

    let max_value = exponents.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    max_value
        + exponents
            .iter()
            .map(|x| (x - max_value).exp())
            .sum::<f64>()
            .ln()
}

fn exact_periodic_cv(l: usize, k: f64) -> f64 {
    let n = l * l;
    let total = 1usize << n;
    let mut bond_sums = Vec::with_capacity(total);

    for state in 0..total {
        let spin = |i: usize| -> f64 {
            if (state >> i) & 1 == 0 {
                1.0
            } else {
                -1.0
            }
        };

        let mut bond_sum = 0.0;
        for y in 0..l {
            for x in 0..l {
                let i = y * l + x;
                let right = y * l + (x + 1) % l;
                let down = ((y + 1) % l) * l + x;
                bond_sum += spin(i) * spin(right);
                bond_sum += spin(i) * spin(down);
            }
        }
        bond_sums.push(bond_sum);
    }

    let max_value = bond_sums
        .iter()
        .map(|b| k * b)
        .fold(f64::NEG_INFINITY, f64::max);

    let weights: Vec<f64> = bond_sums
        .iter()
        .map(|b| (k * b - max_value).exp())
        .collect();

    let z = weights.iter().sum::<f64>();
    let mean = weights
        .iter()
        .zip(bond_sums.iter())
        .map(|(w, b)| w * b)
        .sum::<f64>()
        / z;
    let mean2 = weights
        .iter()
        .zip(bond_sums.iter())
        .map(|(w, b)| w * b * b)
        .sum::<f64>()
        / z;

    k * k * (mean2 - mean * mean) / (n as f64)
}

pub fn run_cli() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let method = get_arg(&args, "--method")?;
    if method != "trg" {
        return Err("TRG CLI requires --method trg".into());
    }

    let l: usize = get_arg(&args, "--l")?.parse()?;
    let k: f64 = get_arg(&args, "--k")?.parse()?;
    let chi: usize = get_arg(&args, "--chi")?.parse()?;
    let out = get_arg(&args, "--out")?;

    println!("==========================================");
    println!("Week 10 - Periodic Tensor Renormalization Group");
    println!("==========================================");
    println!("Boundary condition: periodic");
    println!("L = {}", l);
    println!("K = {:.8}", k);
    println!("T = {:.8}", 1.0 / k);
    println!("chi = {}", chi);
    println!("Output = {}", out);
    println!();

    let result = contract_trg(l, k, chi, &out)?;

    println!("\n==========================================");
    println!("Periodic TRG result");
    println!("==========================================");
    println!("log(Z) = {:.15e}", result.log_z);
    println!("g      = {:.15e}", result.g);
    println!("z      = {:.15e}", result.z);
    println!("Steps  = {}", result.steps);
    println!("Final tensor dimension = {}", result.final_dim);
    println!("Time   = {:.6} seconds", result.elapsed);
    println!("Evidence saved to {}", out);
    println!("Periodic TRG completed successfully.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_periodic_l2_exact_enumeration() {
        let k = 0.44;
        let result = contract_trg(2, k, 16, "target/trg_test_l2").unwrap();
        let exact = exact_periodic_log_z(2, k) / 4.0;
        assert!(
            (result.g - exact).abs() < 1e-8,
            "TRG={}, exact={}, error={}",
            result.g,
            exact,
            (result.g - exact).abs()
        );
    }

    #[test]
    fn test_periodic_l4_chi8_against_enumeration() {
        let k = 0.44;
        let result = contract_trg(4, k, 8, "target/trg_test_l4").unwrap();
        let exact = exact_periodic_log_z(4, k) / 16.0;
        assert!(
            (result.g - exact).abs() < 2e-3,
            "TRG={}, exact={}, error={}",
            result.g,
            exact,
            (result.g - exact).abs()
        );
    }

    #[test]
    fn test_periodic_l2_specific_heat() {
        let k = 0.44;
        let h = 0.0025;
        let gm = contract_trg(2, k - h, 16, "target/trg_test_cv_m")
            .unwrap()
            .g;
        let g0 = contract_trg(2, k, 16, "target/trg_test_cv_0").unwrap().g;
        let gp = contract_trg(2, k + h, 16, "target/trg_test_cv_p")
            .unwrap()
            .g;

        let cv_trg = k * k * (gp - 2.0 * g0 + gm) / (h * h);
        let cv_exact = exact_periodic_cv(2, k);

        assert!(
            (cv_trg - cv_exact).abs() < 1e-3,
            "TRG cV={}, exact cV={}, error={}",
            cv_trg,
            cv_exact,
            (cv_trg - cv_exact).abs()
        );
    }
}
