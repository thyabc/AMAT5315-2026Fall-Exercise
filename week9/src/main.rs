use anyhow::{anyhow, ensure, Context, Result};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;
use std::str::SplitWhitespace;

use tenferro_cpu::CpuBackend;
use tenferro_einsum::TensorEinsumExt;
use tenferro_tensor::{BackendSessionHost, Tensor};

struct Factor {
    scope: Vec<usize>,
    values: Vec<f64>,
}

struct Model {
    dims: Vec<usize>,
    factors: Vec<Factor>,
}

#[derive(Serialize)]
struct Inference {
    evidence_mass: f64,
    marginals: Vec<[f64; 2]>,
    joint_tl: [[f64; 2]; 2],
}

#[derive(Serialize)]
struct RunRecord {
    label: String,
    observations: BTreeMap<String, usize>,
    evidence_mass_enumeration: f64,
    evidence_mass_contraction: f64,
    relative_evidence_error: f64,
    marginals_enumeration: Vec<[f64; 2]>,
    marginals_contraction: Vec<[f64; 2]>,
    joint_tl_enumeration: [[f64; 2]; 2],
    joint_tl_contraction: [[f64; 2]; 2],
    max_marginal_error: f64,
    max_joint_error: f64,
}

fn token<'a>(it: &mut SplitWhitespace<'a>, name: &str) -> Result<&'a str> {
    it.next()
        .ok_or_else(|| anyhow!("Missing UAI field: {}", name))
}

// Parse the standard MARKOV UAI format.
// UAI table values are stored in row-major order.
fn load_uai(path: &str) -> Result<Model> {
    let content = fs::read_to_string(path).with_context(|| format!("Cannot read {}", path))?;

    let mut it = content.split_whitespace();

    ensure!(
        token(&mut it, "model type")? == "MARKOV",
        "Expected MARKOV model"
    );

    let n_vars: usize = token(&mut it, "variable count")?.parse()?;

    let mut dims = Vec::with_capacity(n_vars);

    for _ in 0..n_vars {
        dims.push(token(&mut it, "variable dimension")?.parse()?);
    }

    let n_factors: usize = token(&mut it, "factor count")?.parse()?;

    let mut scopes = Vec::with_capacity(n_factors);

    // Read all factor scopes first.
    for _ in 0..n_factors {
        let rank: usize = token(&mut it, "factor rank")?.parse()?;
        let mut scope = Vec::with_capacity(rank);

        for _ in 0..rank {
            let v: usize = token(&mut it, "scope variable")?.parse()?;
            ensure!(v < n_vars, "Invalid variable ID {}", v);
            scope.push(v);
        }

        scopes.push(scope);
    }

    let mut factors = Vec::with_capacity(n_factors);

    // Then read all factor tables.
    for scope in scopes {
        let count: usize = token(&mut it, "table entry count")?.parse()?;

        let expected: usize = scope.iter().map(|&v| dims[v]).product();

        ensure!(
            count == expected,
            "Factor {:?}: expected {} entries, got {}",
            scope,
            expected,
            count
        );

        let mut values = Vec::with_capacity(count);

        for _ in 0..count {
            values.push(token(&mut it, "factor value")?.parse::<f64>()?);
        }

        factors.push(Factor { scope, values });
    }

    ensure!(it.next().is_none(), "Unexpected trailing UAI data");

    Ok(Model { dims, factors })
}

// Convert UAI row-major factor values to tenferro column-major storage.
fn row_to_col(shape: &[usize], row: &[f64]) -> Vec<f64> {
    let total = row.len();
    let mut col = vec![0.0; total];

    for col_index in 0..total {
        let mut remainder = col_index;
        let mut coordinates = vec![0; shape.len()];

        // Decode column-major index.
        for axis in 0..shape.len() {
            coordinates[axis] = remainder % shape[axis];
            remainder /= shape[axis];
        }

        // Encode the same coordinates in row-major order.
        let mut row_index = 0;

        for axis in 0..shape.len() {
            row_index = row_index * shape[axis] + coordinates[axis];
        }

        col[col_index] = row[row_index];
    }

    col
}

// Direct enumeration oracle.
fn enumerate(model: &Model, evidence: &BTreeMap<usize, usize>) -> Result<Inference> {
    let n = model.dims.len();
    let total: usize = model.dims.iter().product();

    let mut mass = 0.0;
    let mut marginals = vec![[0.0_f64; 2]; n];
    let mut joint = [[0.0_f64; 2]; 2];

    for linear in 0..total {
        let mut assignment = vec![0; n];
        let mut remainder = linear;

        // Last variable changes fastest.
        for v in (0..n).rev() {
            assignment[v] = remainder % model.dims[v];
            remainder /= model.dims[v];
        }

        // Apply observations.
        if evidence.iter().any(|(&v, &s)| assignment[v] != s) {
            continue;
        }

        let mut weight = 1.0;

        for factor in &model.factors {
            let mut index = 0;

            for &v in &factor.scope {
                index = index * model.dims[v] + assignment[v];
            }

            weight *= factor.values[index];
        }

        mass += weight;

        for v in 0..n {
            marginals[v][assignment[v]] += weight;
        }

        // T = variable 1; L = variable 3.
        joint[assignment[1]][assignment[3]] += weight;
    }

    ensure!(mass > 0.0, "Evidence has zero probability");

    for marginal in &mut marginals {
        for value in marginal {
            *value /= mass;
        }
    }

    for row in &mut joint {
        for value in row {
            *value /= mass;
        }
    }

    Ok(Inference {
        evidence_mass: mass,
        marginals,
        joint_tl: joint,
    })
}

// Build tensor factors and append unary evidence indicator factors.
fn build_tensors(
    model: &Model,
    evidence: &BTreeMap<usize, usize>,
) -> Result<(Vec<Tensor>, Vec<Vec<usize>>)> {
    let mut tensors = Vec::new();
    let mut scopes = Vec::new();

    for factor in &model.factors {
        let shape: Vec<usize> = factor.scope.iter().map(|&v| model.dims[v]).collect();

        let data = row_to_col(&shape, &factor.values);

        let tensor = Tensor::from_vec_col_major(shape, data)?;

        tensors.push(tensor);
        scopes.push(factor.scope.clone());
    }

    // Evidence: add a unary delta tensor for each observed variable.
    for (&v, &state) in evidence {
        let mut values = vec![0.0, 0.0];
        values[state] = 1.0;

        tensors.push(Tensor::from_vec_col_major(vec![2], values)?);

        scopes.push(vec![v]);
    }

    Ok((tensors, scopes))
}

// Variable ID 0..7 maps to einsum labels a..h.
fn variable_label(v: usize) -> char {
    (b'a' + v as u8) as char
}

// Contract an N-ary tensor network with specified open indices.
fn contract(
    tensors: &[Tensor],
    scopes: &[Vec<usize>],
    output_vars: &[usize],
    backend: &mut CpuBackend,
) -> Result<Vec<f64>> {
    let input_labels: Vec<String> = scopes
        .iter()
        .map(|scope| scope.iter().map(|&v| variable_label(v)).collect())
        .collect();

    let output_labels: String = output_vars.iter().map(|&v| variable_label(v)).collect();

    let equation = format!("{}->{}", input_labels.join(","), output_labels);

    let refs: Vec<&Tensor> = tensors.iter().collect();

    let result =
        backend.with_backend_session(|session| refs.as_slice().einsum(&equation, session))?;

    Ok(result.as_slice::<f64>()?.to_vec())
}

fn tensor_inference(
    model: &Model,
    evidence: &BTreeMap<usize, usize>,
    backend: &mut CpuBackend,
) -> Result<Inference> {
    let (tensors, scopes) = build_tensors(model, evidence)?;

    // Closed network: evidence mass.
    let mass = contract(&tensors, &scopes, &[], backend)?[0];

    ensure!(mass > 0.0, "Tensor contraction returned zero mass");

    let mut marginals = vec![[0.0_f64; 2]; model.dims.len()];

    // Keep one variable open at a time.
    for v in 0..model.dims.len() {
        let raw = contract(&tensors, &scopes, &[v], backend)?;

        marginals[v][0] = raw[0] / mass;
        marginals[v][1] = raw[1] / mass;
    }

    // Keep T and L open simultaneously.
    let raw_joint = contract(&tensors, &scopes, &[1, 3], backend)?;

    let mut joint = [[0.0_f64; 2]; 2];

    // Output axes are [T, L], in column-major order.
    for t in 0..2 {
        for l in 0..2 {
            joint[t][l] = raw_joint[t + 2 * l] / mass;
        }
    }

    Ok(Inference {
        evidence_mass: mass,
        marginals,
        joint_tl: joint,
    })
}

fn run_matrix_test(backend: &mut CpuBackend) -> Result<()> {
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])?;

    let b = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 1.0, 0.0, 2.0])?;

    let result = backend.with_backend_session(|session| [&a, &b].einsum("ij,jk->ik", session))?;

    let values = result.as_slice::<f64>()?;
    let expected = [4.0, 10.0, 4.0, 8.0];

    let error = values
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    let trace = values[0] + values[3];

    ensure!(error < 1e-12, "Matrix contraction failed");
    ensure!((trace - 12.0).abs() < 1e-12, "Trace failed");

    println!("\n===== Matrix validation =====");
    println!("AB = [[4, 4], [10, 8]]");
    println!("tr(AB) = {:.15}", trace);
    println!("Max error = {:.3e}", error);
    println!("PASS");

    Ok(())
}

fn run_ising_test(backend: &mut CpuBackend) -> Result<()> {
    let beta = 0.5_f64;
    let j = 1.0_f64;
    let spins = [-1.0_f64, 1.0_f64];

    let mut z_enum = 0.0;

    for mask in 0..8 {
        let s = [
            spins[mask & 1],
            spins[(mask >> 1) & 1],
            spins[(mask >> 2) & 1],
        ];

        let bond_sum = s[0] * s[1] + s[1] * s[2] + s[2] * s[0];

        z_enum += (beta * j * bond_sum).exp();
    }

    let mut bond_row = Vec::new();

    for &s1 in &spins {
        for &s2 in &spins {
            bond_row.push((beta * j * s1 * s2).exp());
        }
    }

    let bond_data = row_to_col(&[2, 2], &bond_row);

    let bond = Tensor::from_vec_col_major(vec![2, 2], bond_data)?;

    let result = backend
        .with_backend_session(|session| [&bond, &bond, &bond].einsum("ab,bc,ca->", session))?;

    let z_tensor = result.as_slice::<f64>()?[0];

    let expected = 12.60256209895193_f64;

    println!("\n===== Ising ring validation =====");
    println!("Enumeration Z = {:.15}", z_enum);
    println!("Tensor Z      = {:.15}", z_tensor);
    println!("Reference Z   = {:.15}", expected);
    println!("Tensor error  = {:.3e}", (z_tensor - z_enum).abs());

    ensure!(
        (z_enum - expected).abs() < 1e-12,
        "Ising enumeration mismatch"
    );

    ensure!(
        (z_tensor - z_enum).abs() < 1e-12,
        "Ising tensor contraction mismatch"
    );

    println!("PASS");

    Ok(())
}

fn main() -> Result<()> {
    println!("==================================");
    println!(" WEEK 9: ASIA EXACT INFERENCE");
    println!("==================================");

    let model = load_uai("asia.uai")?;

    ensure!(model.dims.len() == 8, "ASIA should have 8 variables");
    ensure!(model.factors.len() == 8, "ASIA should have 8 factors");

    ensure!(
        model.dims.iter().all(|&d| d == 2),
        "Expected binary variables"
    );

    println!("Model: MARKOV");
    println!("Variables: {}", model.dims.len());
    println!("Factors: {}", model.factors.len());
    println!("UAI parser: PASS");

    let mut backend = CpuBackend::new();

    run_matrix_test(&mut backend)?;
    run_ising_test(&mut backend)?;

    let cases: [(&str, Vec<(usize, usize)>, f64); 3] = [
        ("prior", vec![], 1.0),
        ("AD", vec![(0, 0), (7, 0)], 0.004501375),
        ("ADX", vec![(0, 0), (7, 0), (6, 0)], 0.00098822675),
    ];

    let variable_names = ["A", "T", "S", "L", "B", "E", "X", "D"];

    for (label, observations, reference_mass) in cases {
        println!("\n==================================");
        println!("CASE: {}", label);
        println!("==================================");

        let evidence: BTreeMap<usize, usize> = observations.iter().copied().collect();

        let enum_result = enumerate(&model, &evidence)?;

        let tensor_result = tensor_inference(&model, &evidence, &mut backend)?;

        let relative_error = (enum_result.evidence_mass - tensor_result.evidence_mass).abs()
            / enum_result.evidence_mass;

        let reference_error = (enum_result.evidence_mass - reference_mass).abs() / reference_mass;

        let max_marginal_error = enum_result
            .marginals
            .iter()
            .zip(tensor_result.marginals.iter())
            .flat_map(|(a, b)| (0..2).map(move |s| (a[s] - b[s]).abs()))
            .fold(0.0_f64, f64::max);

        let max_joint_error = enum_result
            .joint_tl
            .iter()
            .flatten()
            .zip(tensor_result.joint_tl.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        println!("Evidence mass (enum)  = {:.12}", enum_result.evidence_mass);
        println!(
            "Evidence mass (tensor)= {:.12}",
            tensor_result.evidence_mass
        );
        println!("Reference mass error  = {:.3e}", reference_error);
        println!("Relative mass error   = {:.3e}", relative_error);
        println!("Max marginal error    = {:.3e}", max_marginal_error);
        println!("Max joint error       = {:.3e}", max_joint_error);

        println!("\nDisease probabilities (state 0):");

        for v in [1, 3, 4] {
            println!(
                "P({}=0): enum={:.10}, tensor={:.10}",
                variable_names[v], enum_result.marginals[v][0], tensor_result.marginals[v][0]
            );
        }

        println!("\nT-L joint marginal (enumeration):");

        for t in 0..2 {
            println!(
                "{:.10} {:.10}",
                enum_result.joint_tl[t][0], enum_result.joint_tl[t][1]
            );
        }

        ensure!(
            reference_error < 1e-10,
            "{} reference evidence mass mismatch",
            label
        );

        ensure!(relative_error < 1e-10, "{} evidence mass mismatch", label);

        ensure!(max_marginal_error < 1e-10, "{} marginal mismatch", label);

        ensure!(max_joint_error < 1e-10, "{} joint mismatch", label);

        for marginal in &tensor_result.marginals {
            ensure!(
                (marginal[0] + marginal[1] - 1.0).abs() < 1e-10,
                "Marginal normalization failed"
            );
        }

        let observation_names: BTreeMap<String, usize> = observations
            .iter()
            .map(|&(v, s)| (variable_names[v].to_string(), s))
            .collect();

        let record = RunRecord {
            label: label.to_string(),
            observations: observation_names,
            evidence_mass_enumeration: enum_result.evidence_mass,
            evidence_mass_contraction: tensor_result.evidence_mass,
            relative_evidence_error: relative_error,
            marginals_enumeration: enum_result.marginals,
            marginals_contraction: tensor_result.marginals,
            joint_tl_enumeration: enum_result.joint_tl,
            joint_tl_contraction: tensor_result.joint_tl,
            max_marginal_error,
            max_joint_error,
        };

        let out_dir = format!("artifacts/{}", label);

        fs::create_dir_all(&out_dir)?;

        fs::write(
            format!("{}/run.json", out_dir),
            serde_json::to_string_pretty(&record)?,
        )?;

        println!("Saved: {}/run.json", out_dir);
        println!("CASE {}: PASS", label);
    }

    println!("\n==================================");
    println!(" ALL EXACT-INFERENCE CHECKS PASSED");
    println!("==================================");

    Ok(())
}
