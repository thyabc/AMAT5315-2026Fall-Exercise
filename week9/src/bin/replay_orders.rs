use anyhow::{ensure, Context, Result};
use omeco::NestedEinsum;
use serde::Serialize;
use serde_json::Value;
use std::fs;

use tenferro_cpu::CpuBackend;
use tenferro_einsum::TensorEinsumExt;
use tenferro_tensor::{BackendSessionHost, Tensor};

struct Factor {
    scope: Vec<usize>,
    values: Vec<f64>,
}

#[derive(Serialize)]
struct ReplayRecord {
    method: String,
    seed: Option<u64>,
    replay_evidence_mass: f64,
    enumeration_evidence_mass: f64,
    relative_error: f64,
    passed: bool,
}

fn read_model() -> Result<(Vec<usize>, Vec<Factor>)> {
    let content = fs::read_to_string("asia.uai")?;
    let mut it = content.split_whitespace();

    ensure!(it.next() == Some("MARKOV"), "Expected MARKOV");

    let n: usize = it.next().context("Missing variable count")?.parse()?;

    let mut dims = Vec::new();
    for _ in 0..n {
        dims.push(it.next().context("Missing dimension")?.parse()?);
    }

    let nf: usize = it.next().context("Missing factor count")?.parse()?;

    let mut scopes = Vec::new();

    for _ in 0..nf {
        let rank: usize = it.next().context("Missing rank")?.parse()?;
        let mut scope = Vec::new();

        for _ in 0..rank {
            scope.push(it.next().context("Missing scope")?.parse::<usize>()?);
        }

        scopes.push(scope);
    }

    let mut factors = Vec::new();

    for scope in scopes {
        let count: usize = it.next().context("Missing table size")?.parse()?;

        let mut values = Vec::new();
        for _ in 0..count {
            values.push(it.next().context("Missing table value")?.parse::<f64>()?);
        }

        factors.push(Factor { scope, values });
    }

    Ok((dims, factors))
}

fn row_to_col(shape: &[usize], row: &[f64]) -> Vec<f64> {
    let mut col = vec![0.0; row.len()];

    for ci in 0..row.len() {
        let mut rem = ci;
        let mut coords = vec![0; shape.len()];

        for axis in 0..shape.len() {
            coords[axis] = rem % shape[axis];
            rem /= shape[axis];
        }

        let mut ri = 0;
        for axis in 0..shape.len() {
            ri = ri * shape[axis] + coords[axis];
        }

        col[ci] = row[ri];
    }

    col
}

fn build_inputs(
    dims: &[usize],
    factors: &[Factor],
) -> Result<(Vec<Option<Tensor>>, Vec<Vec<char>>)> {
    let mut tensors = Vec::new();
    let mut scopes = Vec::new();

    for factor in factors {
        let shape: Vec<usize> = factor.scope.iter().map(|&v| dims[v]).collect();

        let data = row_to_col(&shape, &factor.values);

        tensors.push(Some(Tensor::from_vec_col_major(shape, data)?));

        scopes.push(
            factor
                .scope
                .iter()
                .map(|&v| (b'a' + v as u8) as char)
                .collect(),
        );
    }

    // AD evidence: A=0 and D=0.
    for v in [0usize, 7usize] {
        let mut indicator = vec![0.0, 0.0];
        indicator[0] = 1.0;

        tensors.push(Some(Tensor::from_vec_col_major(vec![2], indicator)?));

        scopes.push(vec![(b'a' + v as u8) as char]);
    }

    Ok((tensors, scopes))
}

fn evaluate(
    tree: &NestedEinsum<char>,
    tensors: &mut [Option<Tensor>],
    scopes: &[Vec<char>],
    backend: &mut CpuBackend,
) -> Result<(Tensor, Vec<char>)> {
    match tree {
        NestedEinsum::Leaf { tensor_index } => {
            let tensor = tensors
                .get_mut(*tensor_index)
                .context("Invalid tensor index")?
                .take()
                .context("Tensor leaf used more than once")?;

            Ok((tensor, scopes[*tensor_index].clone()))
        }

        NestedEinsum::Node { args, eins } => {
            ensure!(args.len() == 2, "Expected binary tree node");

            let (left, left_labels) = evaluate(&args[0], tensors, scopes, backend)?;

            let (right, right_labels) = evaluate(&args[1], tensors, scopes, backend)?;

            ensure!(
                eins.ixs.len() == 2 && eins.ixs[0] == left_labels && eins.ixs[1] == right_labels,
                "Saved contraction tree does not match input labels"
            );

            let output_labels = eins.iy.clone();

            let left_s: String = left_labels.iter().collect();
            let right_s: String = right_labels.iter().collect();
            let output_s: String = output_labels.iter().collect();

            let equation = format!("{},{}->{}", left_s, right_s, output_s);

            let result = backend
                .with_backend_session(|session| [&left, &right].einsum(&equation, session))?;

            Ok((result, output_labels))
        }
    }
}

fn replay(method: &str) -> Result<()> {
    let plan_path = format!("runs/{}/order.json", method);
    let plan: Value = serde_json::from_str(&fs::read_to_string(&plan_path)?)?;

    let tree: NestedEinsum<char> = serde_json::from_value(plan["tree"].clone())?;

    let (dims, factors) = read_model()?;

    let (mut tensors, scopes) = build_inputs(&dims, &factors)?;

    let saved_scopes: Vec<String> = serde_json::from_value(plan["input_scopes"].clone())?;

    let actual_scopes: Vec<String> = scopes.iter().map(|s| s.iter().collect()).collect();

    ensure!(
        saved_scopes == actual_scopes,
        "Saved plan input scopes do not match model"
    );

    ensure!(tree.is_binary(), "Expected binary contraction tree");
    ensure!(tree.leaf_count() == tensors.len(), "Leaf count mismatch");

    let mut backend = CpuBackend::new();

    let (result, output_labels) = evaluate(&tree, &mut tensors, &scopes, &mut backend)?;

    ensure!(output_labels.is_empty(), "Expected scalar output");

    ensure!(
        tensors.iter().all(|x| x.is_none()),
        "Not all input tensors were consumed"
    );

    let replay_mass = result.as_slice::<f64>()?[0];

    let reference: Value = serde_json::from_str(&fs::read_to_string("artifacts/AD/run.json")?)?;

    let enum_mass = reference["evidence_mass_enumeration"]
        .as_f64()
        .context("Missing enumeration result")?;

    let relative_error = (replay_mass - enum_mass).abs() / enum_mass;

    let seed = plan["seed"].as_u64();

    println!("\n===== {} REPLAY =====", method.to_uppercase());
    println!("Replay evidence mass = {:.12}", replay_mass);
    println!("Enumeration mass     = {:.12}", enum_mass);
    println!("Relative error       = {:.3e}", relative_error);

    ensure!(
        relative_error < 1e-10,
        "{} replay does not match enumeration",
        method
    );

    let record = ReplayRecord {
        method: method.to_string(),
        seed,
        replay_evidence_mass: replay_mass,
        enumeration_evidence_mass: enum_mass,
        relative_error,
        passed: true,
    };

    fs::write(
        format!("runs/{}/replay.json", method),
        serde_json::to_string_pretty(&record)?,
    )?;

    println!("REPLAY {}: PASS", method.to_uppercase());

    Ok(())
}

fn main() -> Result<()> {
    replay("random")?;
    replay("greedy")?;

    println!("\nALL PLAN REPLAY CHECKS PASSED");

    Ok(())
}
