use anyhow::{bail, ensure, Context, Result};
use omeco::{optimize_code, EinCode, GreedyMethod, NestedEinsum, ScoreFunction, TreeSA};
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs;
use std::time::Instant;

use tenferro_cpu::CpuBackend;
use tenferro_einsum::{EinsumSubscripts, TensorEinsumExt};
use tenferro_tensor::{BackendSessionHost, Tensor};

const MODEL_PATH: &str = "promedus/Promedus_14.uai";
const EVIDENCE_PATH: &str = "promedus/Promedus_14.uai.evid";
const PR_PATH: &str = "promedus/Promedus_14.uai.PR";
const MAR_PATH: &str = "promedus/Promedus_14.uai.MAR";
const QUERY_VARS: [usize; 3] = [59, 102, 139];
const BP_DAMPING: f64 = 0.5;
const BP_TOLERANCE: f64 = 1e-10;
const BP_MAX_SWEEPS: usize = 1000;

#[derive(Clone)]
struct Factor {
    scope: Vec<usize>,
    // UAI stores factor tables in row-major order.
    values: Vec<f64>,
}

struct Model {
    dims: Vec<usize>,
    factors: Vec<Factor>,
}

struct PreparedModel {
    n_vars: usize,
    dims: Vec<usize>,
    factors: Vec<Factor>,
    evidence: BTreeMap<usize, usize>,
    scalar_factor_values: Vec<f64>,
    scalar_log_scale: f64,
}

fn next_usize<'a>(
    it: &mut std::str::SplitWhitespace<'a>,
    what: &str,
) -> Result<usize> {
    it.next()
        .with_context(|| format!("Missing {}", what))?
        .parse::<usize>()
        .with_context(|| format!("Invalid {}", what))
}

fn load_uai(path: &str) -> Result<Model> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Cannot read UAI model {}", path))?;
    let mut it = content.split_whitespace();

    ensure!(it.next() == Some("MARKOV"), "Expected MARKOV UAI model");

    let n_vars = next_usize(&mut it, "variable count")?;
    let mut dims = Vec::with_capacity(n_vars);

    for _ in 0..n_vars {
        dims.push(next_usize(&mut it, "variable dimension")?);
    }

    ensure!(
        dims.iter().all(|&d| d == 2),
        "This assignment expects binary Promedus variables"
    );

    let n_factors = next_usize(&mut it, "factor count")?;
    let mut scopes = Vec::with_capacity(n_factors);

    for _ in 0..n_factors {
        let rank = next_usize(&mut it, "factor rank")?;
        let mut scope = Vec::with_capacity(rank);

        for _ in 0..rank {
            let v = next_usize(&mut it, "scope variable ID")?;
            ensure!(v < n_vars, "Variable ID {} is out of range", v);
            scope.push(v);
        }

        scopes.push(scope);
    }

    let mut factors = Vec::with_capacity(n_factors);

    for scope in scopes {
        let count = next_usize(&mut it, "factor table size")?;
        let expected = scope.iter().map(|&v| dims[v]).product::<usize>();

        ensure!(
            count == expected,
            "Factor {:?}: expected {} values, found {}",
            scope,
            expected,
            count
        );

        let mut values = Vec::with_capacity(count);

        for _ in 0..count {
            let value = it
                .next()
                .context("Missing factor table value")?
                .parse::<f64>()?;
            ensure!(value.is_finite() && value >= 0.0, "Invalid factor value");
            values.push(value);
        }

        factors.push(Factor { scope, values });
    }

    ensure!(it.next().is_none(), "Unexpected trailing UAI data");

    Ok(Model { dims, factors })
}

fn load_evidence(path: &str, n_vars: usize) -> Result<BTreeMap<usize, usize>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Cannot read evidence {}", path))?;
    let mut it = content.split_whitespace();

    let n_obs = next_usize(&mut it, "evidence count")?;
    let mut evidence = BTreeMap::new();

    for _ in 0..n_obs {
        let var = next_usize(&mut it, "evidence variable ID")?;
        let state = next_usize(&mut it, "evidence state")?;

        ensure!(var < n_vars, "Evidence variable {} is out of range", var);
        ensure!(state < 2, "Evidence state must be 0 or 1");
        ensure!(
            evidence.insert(var, state).is_none(),
            "Duplicate evidence variable {}",
            var
        );
    }

    ensure!(it.next().is_none(), "Unexpected trailing evidence data");
    Ok(evidence)
}

// Apply evidence by fixing observed coordinates inside every factor.
// Factors that become scalars are retained in scalar_factor_values and their
// logarithms are added to scalar_log_scale; they are not silently discarded.
fn apply_evidence(model: &Model, evidence: &BTreeMap<usize, usize>) -> Result<PreparedModel> {
    let mut reduced_factors = Vec::new();
    let mut scalar_factor_values = Vec::new();
    let mut scalar_log_scale = 0.0_f64;

    for factor in &model.factors {
        let new_scope: Vec<usize> = factor
            .scope
            .iter()
            .copied()
            .filter(|v| !evidence.contains_key(v))
            .collect();

        let new_entries = 1usize
            .checked_shl(new_scope.len() as u32)
            .context("Reduced factor rank is too large")?;

        let mut reduced_values = Vec::with_capacity(new_entries);

        for new_linear in 0..new_entries {
            // Decode the reduced assignment; last variable changes fastest.
            let mut new_states = vec![0usize; new_scope.len()];
            let mut rem = new_linear;

            for p in (0..new_scope.len()).rev() {
                new_states[p] = rem % 2;
                rem /= 2;
            }

            // Reconstruct the original UAI row-major index.
            let mut old_linear = 0usize;
            let mut new_pos = 0usize;

            for &var in &factor.scope {
                let state = if let Some(&observed_state) = evidence.get(&var) {
                    observed_state
                } else {
                    let s = new_states[new_pos];
                    new_pos += 1;
                    s
                };

                old_linear = old_linear * 2 + state;
            }

            reduced_values.push(factor.values[old_linear]);
        }

        if new_scope.is_empty() {
            let scalar = reduced_values[0];
            ensure!(
                scalar > 0.0,
                "Evidence makes a scalar factor zero; evidence has zero mass"
            );
            scalar_factor_values.push(scalar);
            scalar_log_scale += scalar.ln();
        } else {
            reduced_factors.push(Factor {
                scope: new_scope,
                values: reduced_values,
            });
        }
    }

    ensure!(!reduced_factors.is_empty(), "No non-scalar factors remain");

    Ok(PreparedModel {
        n_vars: model.dims.len(),
        dims: model.dims.clone(),
        factors: reduced_factors,
        evidence: evidence.clone(),
        scalar_factor_values,
        scalar_log_scale,
    })
}

fn row_to_col(shape: &[usize], row: &[f64]) -> Vec<f64> {
    let mut col = vec![0.0; row.len()];

    for col_index in 0..row.len() {
        let mut remainder = col_index;
        let mut coordinates = vec![0usize; shape.len()];

        for axis in 0..shape.len() {
            coordinates[axis] = remainder % shape[axis];
            remainder /= shape[axis];
        }

        let mut row_index = 0usize;

        for axis in 0..shape.len() {
            row_index = row_index * shape[axis] + coordinates[axis];
        }

        col[col_index] = row[row_index];
    }

    col
}

fn build_size_map(scopes: &[Vec<usize>]) -> HashMap<usize, usize> {
    let mut sizes = HashMap::new();

    for scope in scopes {
        for &var in scope {
            sizes.insert(var, 2usize);
        }
    }

    sizes
}

fn tensor_size(labels: &[usize], sizes: &HashMap<usize, usize>) -> Result<usize> {
    labels.iter().try_fold(1usize, |acc, label| {
        acc.checked_mul(*sizes.get(label).context("Missing label dimension")?)
            .context("Tensor size overflow")
    })
}

#[derive(Serialize)]
struct StepRecord {
    step: usize,
    left_labels: Vec<usize>,
    right_labels: Vec<usize>,
    retained_labels: Vec<usize>,
    work: usize,
    output_elements: usize,
}

#[derive(Serialize)]
struct ComplexityRecord {
    method: String,
    work: usize,
    largest_tensor_elements: usize,
    steps: Vec<StepRecord>,
}

fn analyze_tree(
    tree: &NestedEinsum<usize>,
    input_scopes: &[Vec<usize>],
    sizes: &HashMap<usize, usize>,
    steps: &mut Vec<StepRecord>,
    largest: &mut usize,
) -> Result<Vec<usize>> {
    match tree {
        NestedEinsum::Leaf { tensor_index } => {
            let labels = input_scopes
                .get(*tensor_index)
                .context("Invalid tensor leaf index")?
                .clone();

            *largest = (*largest).max(tensor_size(&labels, sizes)?);
            Ok(labels)
        }

        NestedEinsum::Node { args, eins } => {
            ensure!(args.len() == 2, "Expected a binary contraction tree");

            let left = analyze_tree(&args[0], input_scopes, sizes, steps, largest)?;
            let right = analyze_tree(&args[1], input_scopes, sizes, steps, largest)?;

            ensure!(
                eins.ixs.len() == 2 && eins.ixs[0] == left && eins.ixs[1] == right,
                "Contraction tree input labels do not match its children"
            );

            let mut union = left.clone();

            for &label in &right {
                if !union.contains(&label) {
                    union.push(label);
                }
            }

            let work = tensor_size(&union, sizes)?;
            let output = eins.iy.clone();
            let output_elements = tensor_size(&output, sizes)?;

            *largest = (*largest).max(output_elements);

            steps.push(StepRecord {
                step: steps.len() + 1,
                left_labels: left,
                right_labels: right,
                retained_labels: output.clone(),
                work,
                output_elements,
            });

            Ok(output)
        }
    }
}

fn complexity_record(
    method: &str,
    tree: &NestedEinsum<usize>,
    scopes: &[Vec<usize>],
    sizes: &HashMap<usize, usize>,
    expected_output: &[usize],
) -> Result<ComplexityRecord> {
    let mut steps = Vec::new();
    let mut largest = 0usize;

    let output = analyze_tree(tree, scopes, sizes, &mut steps, &mut largest)?;

    ensure!(
        output == expected_output,
        "Tree root output {:?} does not match requested {:?}",
        output,
        expected_output
    );

    let work: usize = steps.iter().map(|s| s.work).sum();

    Ok(ComplexityRecord {
        method: method.to_string(),
        work,
        largest_tensor_elements: largest,
        steps,
    })
}

#[derive(Serialize)]
struct PlanRecord<'a> {
    version: u32,
    model: &'static str,
    method: String,
    output_variables: &'a [usize],
    seed: Option<u64>,
    seed_policy: &'static str,
    input_scopes: &'a [Vec<usize>],
    tree: &'a NestedEinsum<usize>,
}

fn save_plan(
    method: &str,
    tree: &NestedEinsum<usize>,
    scopes: &[Vec<usize>],
    output: &[usize],
    seed: Option<u64>,
    complexity: &ComplexityRecord,
) -> Result<()> {
    let dir = format!("runs/promedus/{}", method);
    fs::create_dir_all(&dir)?;

    let plan = PlanRecord {
        version: 1,
        model: "Promedus_14.uai",
        method: method.to_string(),
        output_variables: output,
        seed,
        seed_policy: if seed.is_some() {
            "omeco 0.2.6: SmallRng seed = trial_index + 42; ntrials=1"
        } else {
            "deterministic greedy; no random seed"
        },
        input_scopes: scopes,
        tree,
    };

    fs::write(
        format!("{}/order.json", dir),
        serde_json::to_string_pretty(&plan)?,
    )?;

    fs::write(
        format!("{}/complexity.json", dir),
        serde_json::to_string_pretty(complexity)?,
    )?;

    Ok(())
}

fn make_greedy_tree(
    scopes: &[Vec<usize>],
    output: &[usize],
) -> Result<NestedEinsum<usize>> {
    let code = EinCode::new(scopes.to_vec(), output.to_vec());
    let sizes = build_size_map(scopes);

    optimize_code(&code, &sizes, &GreedyMethod::default())
        .context("OMECO GreedyMethod did not return a contraction tree")
}

fn make_treesa_tree(
    scopes: &[Vec<usize>],
    output: &[usize],
) -> Result<NestedEinsum<usize>> {
    let code = EinCode::new(scopes.to_vec(), output.to_vec());
    let sizes = build_size_map(scopes);

    // Time-only objective. In omeco 0.2.6 each trial uses a SmallRng
    // seeded with trial_index + 42. With one trial this is reproducible.
    let mut config = TreeSA::default().with_ntrials(1).with_niters(50);
    config.score = ScoreFunction::time_optimized();

    optimize_code(&code, &sizes, &config)
        .context("OMECO TreeSA did not return a contraction tree")
}

struct ScaledTensor {
    tensor: Tensor,
    labels: Vec<usize>,
    log_scale: f64,
}

fn build_tensor_inputs(
    factors: &[Factor],
) -> Result<(Vec<Option<Tensor>>, Vec<Vec<usize>>, Vec<f64>)> {
    let mut tensors = Vec::with_capacity(factors.len());
    let mut scopes = Vec::with_capacity(factors.len());
    let mut log_scales = Vec::with_capacity(factors.len());

    for factor in factors {
        let max_value = factor.values.iter().copied().fold(0.0_f64, f64::max);

        ensure!(
            max_value > 0.0 && max_value.is_finite(),
            "A reduced factor is identically zero or invalid"
        );

        let normalized: Vec<f64> = factor.values.iter().map(|v| v / max_value).collect();
        let shape: Vec<usize> = factor.scope.iter().map(|_| 2usize).collect();
        let column_major = row_to_col(&shape, &normalized);

        tensors.push(Some(Tensor::from_vec_col_major(shape, column_major)?));
        scopes.push(factor.scope.clone());
        log_scales.push(max_value.ln());
    }

    Ok((tensors, scopes, log_scales))
}

fn evaluate_tree(
    tree: &NestedEinsum<usize>,
    tensors: &mut [Option<Tensor>],
    scopes: &[Vec<usize>],
    leaf_log_scales: &[f64],
    dims: &[usize],
    backend: &mut CpuBackend,
) -> Result<ScaledTensor> {
    match tree {
        NestedEinsum::Leaf { tensor_index } => {
            let tensor = tensors
                .get_mut(*tensor_index)
                .context("Invalid tensor leaf index")?
                .take()
                .context("A tensor leaf was used more than once")?;

            Ok(ScaledTensor {
                tensor,
                labels: scopes[*tensor_index].clone(),
                log_scale: leaf_log_scales[*tensor_index],
            })
        }

        NestedEinsum::Node { args, eins } => {
            ensure!(args.len() == 2, "Expected binary contraction node");

            let left = evaluate_tree(
                &args[0],
                tensors,
                scopes,
                leaf_log_scales,
                dims,
                backend,
            )?;

            let right = evaluate_tree(
                &args[1],
                tensors,
                scopes,
                leaf_log_scales,
                dims,
                backend,
            )?;

            ensure!(
                eins.ixs.len() == 2
                    && eins.ixs[0] == left.labels
                    && eins.ixs[1] == right.labels,
                "Saved tree does not match actual tensor labels"
            );

            let output_labels = eins.iy.clone();

            // Integer labels are essential here: Promedus has 414 variables,
            // so a single-character ASCII einsum equation is insufficient.
            let left_ids: Vec<u32> = left.labels.iter().map(|&v| v as u32).collect();
            let right_ids: Vec<u32> = right.labels.iter().map(|&v| v as u32).collect();
            let output_ids: Vec<u32> = output_labels.iter().map(|&v| v as u32).collect();

            let subscripts = EinsumSubscripts {
                inputs: vec![left_ids, right_ids],
                output: output_ids,
            };

            let result = backend.with_backend_session(|session| {
                [&left.tensor, &right.tensor].einsum_subscripts(&subscripts, session)
            })?;

            // Rescale every intermediate tensor. Its removed multiplier is
            // accumulated in log_scale, preserving the absolute partition sum.
            let mut values = result.as_slice::<f64>()?.to_vec();
            let max_value = values
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f64, f64::max);

            ensure!(
                max_value > 0.0 && max_value.is_finite(),
                "A contraction produced a zero or invalid intermediate"
            );

            for value in &mut values {
                *value /= max_value;
            }

            let shape: Vec<usize> = output_labels.iter().map(|&v| dims[v]).collect();
            let scaled_tensor = Tensor::from_vec_col_major(shape, values)?;

            Ok(ScaledTensor {
                tensor: scaled_tensor,
                labels: output_labels,
                log_scale: left.log_scale + right.log_scale + max_value.ln(),
            })
        }
    }
}

fn run_tree(
    tree: &NestedEinsum<usize>,
    factors: &[Factor],
    dims: &[usize],
    expected_output: &[usize],
) -> Result<(Vec<f64>, f64)> {
    let (mut tensors, scopes, leaf_log_scales) = build_tensor_inputs(factors)?;
    ensure!(
        tree.leaf_count() == tensors.len(),
        "Tree leaf count does not match tensor count"
    );

    let mut backend = CpuBackend::new();
    let result = evaluate_tree(
        tree,
        &mut tensors,
        &scopes,
        &leaf_log_scales,
        dims,
        &mut backend,
    )?;

    ensure!(
        result.labels == expected_output,
        "Contraction returned labels {:?}, expected {:?}",
        result.labels,
        expected_output
    );

    ensure!(
        tensors.iter().all(|tensor| tensor.is_none()),
        "Not all input tensors were consumed"
    );

    Ok((result.tensor.as_slice::<f64>()?.to_vec(), result.log_scale))
}

fn read_pr(path: &str) -> Result<f64> {
    let content = fs::read_to_string(path)?;
    let mut it = content.split_whitespace();

    ensure!(it.next() == Some("PR"), "Expected PR answer file");
    let value = it
        .next()
        .context("Missing PR value")?
        .parse::<f64>()?;

    ensure!(it.next().is_none(), "Unexpected trailing PR data");
    Ok(value)
}

fn read_mar(path: &str) -> Result<Vec<[f64; 2]>> {
    let content = fs::read_to_string(path)?;
    let mut it = content.split_whitespace();

    ensure!(it.next() == Some("MAR"), "Expected MAR answer file");
    let n = next_usize(&mut it, "MAR variable count")?;
    let mut marginals = Vec::with_capacity(n);

    for var in 0..n {
        let states = next_usize(&mut it, "MAR state count")?;
        ensure!(states == 2, "Variable {} is not binary in MAR", var);

        let p0 = it
            .next()
            .context("Missing MAR state-0 probability")?
            .parse::<f64>()?;
        let p1 = it
            .next()
            .context("Missing MAR state-1 probability")?
            .parse::<f64>()?;

        marginals.push([p0, p1]);
    }

    ensure!(it.next().is_none(), "Unexpected trailing MAR data");
    Ok(marginals)
}

#[derive(Serialize)]
struct MethodResult {
    method: String,
    work: usize,
    largest_tensor_elements: usize,
    planning_seconds: f64,
    contraction_seconds: f64,
    log10_partition_sum: f64,
}

#[derive(Serialize)]
struct MarginalResult {
    state0: f64,
    state1: f64,
    treesa_state0: f64,
    treesa_state1: f64,
    reference_state0: f64,
    reference_state1: f64,
    state0_absolute_error: f64,
    state1_absolute_error: f64,
    greedy_treesa_max_absolute_difference: f64,
}

#[derive(Serialize)]
struct ExactResult {
    model: String,
    variables: usize,
    original_factor_count: usize,
    reduced_non_scalar_factor_count: usize,
    scalar_factor_count: usize,
    scalar_factor_values: Vec<f64>,
    evidence: BTreeMap<usize, usize>,
    treesa_seed: u64,
    treesa_ntrials: usize,
    treesa_niters_per_beta: usize,
    treesa_beta_levels: usize,
    greedy: MethodResult,
    treesa: MethodResult,
    greedy_treesa_log10_absolute_difference: f64,
    published_pr_log10: f64,
    published_pr_absolute_error: f64,
    marginals: BTreeMap<String, MarginalResult>,
    maximum_greedy_treesa_marginal_difference: f64,
    maximum_mar_absolute_error: f64,
}

fn run_exact() -> Result<()> {
    println!("==============================================");
    println!(" PROMEDUS_14: GREEDY + TIME-ONLY TREESA EXACT");
    println!("==============================================");

    let model = load_uai(MODEL_PATH)?;
    ensure!(model.dims.len() == 414, "Expected 414 Promedus variables");
    ensure!(model.factors.len() == 414, "Expected 414 Promedus factors");

    let evidence = load_evidence(EVIDENCE_PATH, model.dims.len())?;
    ensure!(evidence.len() == 9, "Expected nine observed variables");

    let prepared = apply_evidence(&model, &evidence)?;
    let scopes: Vec<Vec<usize>> = prepared.factors.iter().map(|f| f.scope.clone()).collect();
    let sizes = build_size_map(&scopes);

    println!("Variables: {}", prepared.n_vars);
    println!("Original factors: {}", model.factors.len());
    println!("Observed variables: {}", prepared.evidence.len());
    println!("Non-scalar factors after evidence: {}", prepared.factors.len());
    println!("Scalar factors retained: {}", prepared.scalar_factor_values.len());
    println!("Active variable labels: {}", sizes.len());

    fs::create_dir_all("runs/promedus")?;
    fs::create_dir_all("artifacts/promedus")?;

    // Scalar evidence partition sum: Greedy plan.
    let start = Instant::now();
    let greedy_tree = make_greedy_tree(&scopes, &[])?;
    let greedy_planning_seconds = start.elapsed().as_secs_f64();

    let greedy_complexity =
        complexity_record("greedy", &greedy_tree, &scopes, &sizes, &[])?;
    save_plan(
        "greedy",
        &greedy_tree,
        &scopes,
        &[],
        None,
        &greedy_complexity,
    )?;

    let start = Instant::now();
    let (greedy_values, greedy_log_scale) =
        run_tree(&greedy_tree, &prepared.factors, &prepared.dims, &[])?;
    let greedy_contraction_seconds = start.elapsed().as_secs_f64();

    ensure!(greedy_values.len() == 1, "Scalar contraction should return one value");
    ensure!(greedy_values[0] > 0.0, "Greedy partition sum is zero");

    let greedy_log10 = (greedy_log_scale
        + prepared.scalar_log_scale
        + greedy_values[0].ln())
        / std::f64::consts::LN_10;

    // Scalar evidence partition sum: seeded, time-only TreeSA plan.
    let start = Instant::now();
    let treesa_tree = make_treesa_tree(&scopes, &[])?;
    let treesa_planning_seconds = start.elapsed().as_secs_f64();

    let treesa_complexity =
        complexity_record("treesa", &treesa_tree, &scopes, &sizes, &[])?;
    save_plan(
        "treesa",
        &treesa_tree,
        &scopes,
        &[],
        Some(42),
        &treesa_complexity,
    )?;

    let start = Instant::now();
    let (treesa_values, treesa_log_scale) =
        run_tree(&treesa_tree, &prepared.factors, &prepared.dims, &[])?;
    let treesa_contraction_seconds = start.elapsed().as_secs_f64();

    ensure!(treesa_values.len() == 1, "Scalar contraction should return one value");
    ensure!(treesa_values[0] > 0.0, "TreeSA partition sum is zero");

    let treesa_log10 = (treesa_log_scale
        + prepared.scalar_log_scale
        + treesa_values[0].ln())
        / std::f64::consts::LN_10;

    let exact_plan_difference = (greedy_log10 - treesa_log10).abs();

    println!("\n===== EXACT PARTITION SUM =====");
    println!("Greedy log10(Z_E) = {:.15}", greedy_log10);
    println!("TreeSA log10(Z_E) = {:.15}", treesa_log10);
    println!("Absolute difference = {:.3e}", exact_plan_difference);
    println!(
        "Greedy work={}, largest tensor={}",
        greedy_complexity.work, greedy_complexity.largest_tensor_elements
    );
    println!(
        "TreeSA work={}, largest tensor={}",
        treesa_complexity.work, treesa_complexity.largest_tensor_elements
    );

    // Compute three selected one-variable marginals using exact open-index
    // contractions. Both Greedy and TreeSA get a separate plan for each query.
    let published_mar = read_mar(MAR_PATH)?;
    ensure!(
        published_mar.len() == prepared.n_vars,
        "MAR variable count does not match UAI model"
    );

    let mut marginal_results = BTreeMap::new();
    let mut maximum_mar_error = 0.0_f64;
    let mut maximum_query_plan_difference = 0.0_f64;

    for query_var in QUERY_VARS {
        let output = vec![query_var];

        let greedy_method = format!("query-{}-greedy", query_var);
        let start = Instant::now();
        let query_greedy_tree = make_greedy_tree(&scopes, &output)?;
        let greedy_planning_seconds = start.elapsed().as_secs_f64();

        let query_greedy_complexity = complexity_record(
            &greedy_method,
            &query_greedy_tree,
            &scopes,
            &sizes,
            &output,
        )?;

        save_plan(
            &greedy_method,
            &query_greedy_tree,
            &scopes,
            &output,
            None,
            &query_greedy_complexity,
        )?;

        let start = Instant::now();
        let (greedy_raw, _greedy_query_log_scale) =
            run_tree(&query_greedy_tree, &prepared.factors, &prepared.dims, &output)?;
        let greedy_contraction_seconds = start.elapsed().as_secs_f64();

        let treesa_method = format!("query-{}-treesa", query_var);
        let start = Instant::now();
        let query_treesa_tree = make_treesa_tree(&scopes, &output)?;
        let treesa_planning_seconds = start.elapsed().as_secs_f64();

        let query_treesa_complexity = complexity_record(
            &treesa_method,
            &query_treesa_tree,
            &scopes,
            &sizes,
            &output,
        )?;

        save_plan(
            &treesa_method,
            &query_treesa_tree,
            &scopes,
            &output,
            Some(42),
            &query_treesa_complexity,
        )?;

        let start = Instant::now();
        let (treesa_raw, _treesa_query_log_scale) =
            run_tree(&query_treesa_tree, &prepared.factors, &prepared.dims, &output)?;
        let treesa_contraction_seconds = start.elapsed().as_secs_f64();

        ensure!(
            greedy_raw.len() == 2 && treesa_raw.len() == 2,
            "Binary query should return two states"
        );

        let greedy_normalizer = greedy_raw[0] + greedy_raw[1];
        let treesa_normalizer = treesa_raw[0] + treesa_raw[1];

        ensure!(
            greedy_normalizer > 0.0 && greedy_normalizer.is_finite(),
            "Invalid Greedy marginal normalizer for variable {}",
            query_var
        );
        ensure!(
            treesa_normalizer > 0.0 && treesa_normalizer.is_finite(),
            "Invalid TreeSA marginal normalizer for variable {}",
            query_var
        );

        let p0 = greedy_raw[0] / greedy_normalizer;
        let p1 = greedy_raw[1] / greedy_normalizer;
        let tp0 = treesa_raw[0] / treesa_normalizer;
        let tp1 = treesa_raw[1] / treesa_normalizer;

        let ref0 = published_mar[query_var][0];
        let ref1 = published_mar[query_var][1];

        let err0 = (p0 - ref0).abs();
        let err1 = (p1 - ref1).abs();
        let treesa_err0 = (tp0 - ref0).abs();
        let treesa_err1 = (tp1 - ref1).abs();

        let plan_difference = (p0 - tp0).abs().max((p1 - tp1).abs());
        maximum_query_plan_difference =
            maximum_query_plan_difference.max(plan_difference);
        maximum_mar_error = maximum_mar_error
            .max(err0)
            .max(err1)
            .max(treesa_err0)
            .max(treesa_err1);

        marginal_results.insert(
            query_var.to_string(),
            MarginalResult {
                state0: p0,
                state1: p1,
                treesa_state0: tp0,
                treesa_state1: tp1,
                reference_state0: ref0,
                reference_state1: ref1,
                state0_absolute_error: err0,
                state1_absolute_error: err1,
                greedy_treesa_max_absolute_difference: plan_difference,
            },
        );

        let marginal_file = format!("artifacts/promedus/marginal_{}.json", query_var);
        fs::write(
            marginal_file,
            serde_json::to_string_pretty(&serde_json::json!({
                "variable_id": query_var,
                "state_order": [0, 1],
                "greedy_probabilities": [p0, p1],
                "treesa_probabilities": [tp0, tp1],
                "reference_mar": [ref0, ref1],
                "greedy_absolute_errors": [err0, err1],
                "treesa_absolute_errors": [treesa_err0, treesa_err1],
                "greedy_treesa_max_absolute_difference": plan_difference,
                "greedy_planning_seconds": greedy_planning_seconds,
                "greedy_contraction_seconds": greedy_contraction_seconds,
                "treesa_planning_seconds": treesa_planning_seconds,
                "treesa_contraction_seconds": treesa_contraction_seconds,
                "greedy_work": query_greedy_complexity.work,
                "greedy_largest_tensor_elements": query_greedy_complexity.largest_tensor_elements,
                "treesa_work": query_treesa_complexity.work,
                "treesa_largest_tensor_elements": query_treesa_complexity.largest_tensor_elements
            }))?,
        )?;

        println!(
            "Variable {}: Greedy=({:.10},{:.10}), TreeSA=({:.10},{:.10}), plan diff={:.3e}, MAR max error={:.3e}",
            query_var, p0, p1, tp0, tp1, plan_difference, err0.max(err1).max(treesa_err0).max(treesa_err1)
        );
    }

    let published_pr = read_pr(PR_PATH)?;
    let pr_error = (greedy_log10 - published_pr).abs();

    let result = ExactResult {
        model: "Promedus_14.uai".to_string(),
        variables: prepared.n_vars,
        original_factor_count: model.factors.len(),
        reduced_non_scalar_factor_count: prepared.factors.len(),
        scalar_factor_count: prepared.scalar_factor_values.len(),
        scalar_factor_values: prepared.scalar_factor_values.clone(),
        evidence: prepared.evidence.clone(),
        treesa_seed: 42,
        treesa_ntrials: 1,
        treesa_niters_per_beta: 50,
        treesa_beta_levels: 300,
        greedy: MethodResult {
            method: "omeco GreedyMethod".to_string(),
            work: greedy_complexity.work,
            largest_tensor_elements: greedy_complexity.largest_tensor_elements,
            planning_seconds: greedy_planning_seconds,
            contraction_seconds: greedy_contraction_seconds,
            log10_partition_sum: greedy_log10,
        },
        treesa: MethodResult {
            method: "omeco TreeSA with ScoreFunction::time_optimized".to_string(),
            work: treesa_complexity.work,
            largest_tensor_elements: treesa_complexity.largest_tensor_elements,
            planning_seconds: treesa_planning_seconds,
            contraction_seconds: treesa_contraction_seconds,
            log10_partition_sum: treesa_log10,
        },
        greedy_treesa_log10_absolute_difference: exact_plan_difference,
        published_pr_log10: published_pr,
        published_pr_absolute_error: pr_error,
        marginals: marginal_results,
        maximum_greedy_treesa_marginal_difference: maximum_query_plan_difference,
        maximum_mar_absolute_error: maximum_mar_error,
    };

    fs::write(
        "artifacts/promedus/exact.json",
        serde_json::to_string_pretty(&result)?,
    )?;

    let settings = serde_json::json!({
        "model": MODEL_PATH,
        "evidence": EVIDENCE_PATH,
        "published_pr": PR_PATH,
        "published_mar": MAR_PATH,
        "treesa": {
            "crate": "omeco 0.2.6",
            "objective": "time-only",
            "score": "ScoreFunction::time_optimized()",
            "ntrials": 1,
            "niters_per_beta": 50,
            "beta_levels": 300,
            "seed": 42,
            "seed_policy": "SmallRng::seed_from_u64(trial_index + 42)"
        },
        "numeric_stability": "Every input factor and intermediate tensor is divided by its maximum absolute entry; log multipliers are accumulated and restored in log10(Z).",
        "tensor_labels": "u32 integer labels through tenferro_einsum::EinsumSubscripts"
    });

    fs::write(
        "runs/promedus/settings.json",
        serde_json::to_string_pretty(&settings)?,
    )?;

    println!("\n===== PUBLISHED ANSWER CHECK =====");
    println!("PR file log10(Z) = {:.10}", published_pr);
    println!("PR absolute error = {:.3e}", pr_error);
    println!("Maximum Greedy/TreeSA marginal difference = {:.3e}", maximum_query_plan_difference);
    println!("Maximum selected MAR error = {:.3e}", maximum_mar_error);

    ensure!(
        exact_plan_difference < 1e-10,
        "Greedy and TreeSA exact partition sums differ by more than 1e-10"
    );
    ensure!(
        maximum_query_plan_difference < 1e-10,
        "Greedy and TreeSA selected marginals differ by more than 1e-10"
    );
    ensure!(
        pr_error < 1e-4,
        "Exact log10 partition sum does not match published PR within 1e-4"
    );
    ensure!(
        maximum_mar_error < 1e-5,
        "Selected exact marginals do not match published MAR within 1e-5"
    );

    println!("\nPROMEDUS EXACT INFERENCE: PASS");
    println!("Saved artifacts/promedus/exact.json");
    Ok(())
}

// --------------------------- Loopy BP ---------------------------

type Messages = Vec<Vec<[f64; 2]>>;

#[derive(Serialize)]
struct BpSweepRecord {
    sweep: usize,
    undamped_residual: f64,
    cumulative_update_work: u64,
    max_selected_marginal_error: f64,
    selected_beliefs: BTreeMap<String, [f64; 2]>,
}

#[derive(Serialize)]
struct BpQueryResult {
    state0: f64,
    state1: f64,
    exact_state0: f64,
    exact_state1: f64,
    reference_mar_state1: f64,
    state0_absolute_error: f64,
    state1_absolute_error: f64,
}

#[derive(Serialize)]
struct BpResult {
    model: String,
    method: String,
    damping: f64,
    tolerance: f64,
    max_sweeps: usize,
    converged: bool,
    sweeps: usize,
    final_undamped_residual: f64,
    cumulative_update_work: u64,
    update_work_units_per_sweep: u64,
    runtime_seconds: f64,
    selected_queries: BTreeMap<String, BpQueryResult>,
    history: Vec<BpSweepRecord>,
}

fn normalize_message(x: [f64; 2]) -> Result<[f64; 2]> {
    let sum = x[0] + x[1];

    ensure!(
        sum.is_finite() && sum > 0.0,
        "Cannot normalize a zero or invalid BP message"
    );

    Ok([x[0] / sum, x[1] / sum])
}

fn build_adjacency(factors: &[Factor], n_vars: usize) -> Vec<Vec<(usize, usize)>> {
    let mut adjacency = vec![Vec::new(); n_vars];

    for (factor_index, factor) in factors.iter().enumerate() {
        for (position, &var) in factor.scope.iter().enumerate() {
            adjacency[var].push((factor_index, position));
        }
    }

    adjacency
}

fn uniform_messages(factors: &[Factor]) -> Messages {
    factors
        .iter()
        .map(|factor| vec![[0.5, 0.5]; factor.scope.len()])
        .collect()
}

fn proposals(
    factors: &[Factor],
    adjacency: &[Vec<(usize, usize)>],
    v_to_f: &Messages,
    f_to_v: &Messages,
) -> Result<(Messages, Messages, u64)> {
    let mut next_vf = uniform_messages(factors);
    let mut next_fv = uniform_messages(factors);
    let mut work = 0u64;

    // Variable -> factor messages.
    for (fi, factor) in factors.iter().enumerate() {
        for (position, &var) in factor.scope.iter().enumerate() {
            let mut raw = [1.0_f64, 1.0];

            for &(other_fi, other_pos) in &adjacency[var] {
                if other_fi == fi {
                    continue;
                }

                for state in 0..2 {
                    raw[state] *= f_to_v[other_fi][other_pos][state];
                    work += 1;
                }
            }

            next_vf[fi][position] = normalize_message(raw)?;
        }
    }

    // Factor -> variable messages. UAI factor tables are row-major.
    for (fi, factor) in factors.iter().enumerate() {
        let rank = factor.scope.len();

        for target in 0..rank {
            let mut sums = [0.0_f64, 0.0];
            let mut states = vec![0usize; rank];

            for linear in 0..factor.values.len() {
                let mut remainder = linear;

                for p in (0..rank).rev() {
                    states[p] = remainder % 2;
                    remainder /= 2;
                }

                let mut weight = factor.values[linear];

                for p in 0..rank {
                    if p != target {
                        weight *= v_to_f[fi][p][states[p]];
                        work += 1;
                    }
                }

                sums[states[target]] += weight;
                work += 1;
            }

            next_fv[fi][target] = normalize_message(sums)?;
        }
    }

    Ok((next_vf, next_fv, work))
}

fn message_residual(old: &Messages, proposed: &Messages) -> f64 {
    let mut residual = 0.0_f64;

    for fi in 0..old.len() {
        for position in 0..old[fi].len() {
            for state in 0..2 {
                residual = residual.max(
                    (old[fi][position][state] - proposed[fi][position][state]).abs(),
                );
            }
        }
    }

    residual
}

fn full_residual(
    old_vf: &Messages,
    old_fv: &Messages,
    new_vf: &Messages,
    new_fv: &Messages,
) -> f64 {
    message_residual(old_vf, new_vf).max(message_residual(old_fv, new_fv))
}

fn damp_messages(
    old: &Messages,
    proposed: &Messages,
    damping: f64,
) -> Result<Messages> {
    let mut result = old.clone();

    for fi in 0..old.len() {
        for position in 0..old[fi].len() {
            let raw = [
                (1.0 - damping) * old[fi][position][0]
                    + damping * proposed[fi][position][0],
                (1.0 - damping) * old[fi][position][1]
                    + damping * proposed[fi][position][1],
            ];

            result[fi][position] = normalize_message(raw)?;
        }
    }

    Ok(result)
}

fn compute_beliefs(
    adjacency: &[Vec<(usize, usize)>],
    f_to_v: &Messages,
) -> Result<Vec<[f64; 2]>> {
    let mut beliefs = Vec::with_capacity(adjacency.len());

    for edges in adjacency {
        let mut raw = [1.0_f64, 1.0];

        for &(fi, position) in edges {
            for state in 0..2 {
                raw[state] *= f_to_v[fi][position][state];
            }
        }

        beliefs.push(normalize_message(raw)?);
    }

    Ok(beliefs)
}

fn read_exact_selected() -> Result<BTreeMap<usize, [f64; 2]>> {
    let value: Value = serde_json::from_str(
        &fs::read_to_string("artifacts/promedus/exact.json")
            .context("Run Promedus exact inference before BP")?,
    )?;

    let mut exact = BTreeMap::new();

    for var in QUERY_VARS {
        let key = var.to_string();
        let state0 = value["marginals"][key.as_str()]["state0"]
            .as_f64()
            .with_context(|| format!("Missing exact state-0 marginal for {}", var))?;
        let state1 = value["marginals"][key.as_str()]["state1"]
            .as_f64()
            .with_context(|| format!("Missing exact state-1 marginal for {}", var))?;

        exact.insert(var, [state0, state1]);
    }

    Ok(exact)
}

fn run_bp() -> Result<()> {
    println!("\n==============================================");
    println!(" PROMEDUS_14: LOOPY BELIEF PROPAGATION");
    println!("==============================================");

    let model = load_uai(MODEL_PATH)?;
    let evidence = load_evidence(EVIDENCE_PATH, model.dims.len())?;
    let prepared = apply_evidence(&model, &evidence)?;

    // Scalar factors are constant multipliers and do not change normalized
    // BP messages. Non-scalar factor tables are individually rescaled.
    let mut factors = prepared.factors.clone();

    for factor in &mut factors {
        let max_value = factor.values.iter().copied().fold(0.0_f64, f64::max);
        ensure!(max_value > 0.0, "BP factor is identically zero");
        for value in &mut factor.values {
            *value /= max_value;
        }
    }

    let exact = read_exact_selected()?;
    let published_mar = read_mar(MAR_PATH)?;
    let adjacency = build_adjacency(&factors, prepared.n_vars);

    let mut v_to_f = uniform_messages(&factors);
    let mut f_to_v = uniform_messages(&factors);

    let mut history = Vec::with_capacity(BP_MAX_SWEEPS + 1);
    let mut cumulative_work = 0u64;
    let mut work_per_sweep = 0u64;
    let mut converged = false;

    let start = Instant::now();

    // Record the initial state (sweep 0), then up to 1000 completed updates.
    for sweep in 0..=BP_MAX_SWEEPS {
        let (proposal_vf, proposal_fv, work) =
            proposals(&factors, &adjacency, &v_to_f, &f_to_v)?;

        work_per_sweep = work;
        let residual = full_residual(&v_to_f, &f_to_v, &proposal_vf, &proposal_fv);
        let beliefs = compute_beliefs(&adjacency, &f_to_v)?;

        let mut selected_beliefs = BTreeMap::new();
        let mut max_error = 0.0_f64;

        for var in QUERY_VARS {
            let belief = beliefs[var];
            let reference = exact[&var];

            let error = (belief[0] - reference[0])
                .abs()
                .max((belief[1] - reference[1]).abs());

            max_error = max_error.max(error);
            selected_beliefs.insert(var.to_string(), belief);
        }

        history.push(BpSweepRecord {
            sweep,
            undamped_residual: residual,
            cumulative_update_work: cumulative_work,
            max_selected_marginal_error: max_error,
            selected_beliefs,
        });

        if sweep % 50 == 0 || residual < BP_TOLERANCE || sweep == BP_MAX_SWEEPS {
            println!(
                "BP sweep {:4}: residual={:.6e}, selected-query error={:.6e}, work={}",
                sweep, residual, max_error, cumulative_work
            );
        }

        if residual < BP_TOLERANCE {
            converged = true;
            break;
        }

        if sweep == BP_MAX_SWEEPS {
            break;
        }

        v_to_f = damp_messages(&v_to_f, &proposal_vf, BP_DAMPING)?;
        f_to_v = damp_messages(&f_to_v, &proposal_fv, BP_DAMPING)?;
        cumulative_work += work;
    }

    let runtime_seconds = start.elapsed().as_secs_f64();
    let final_record = history.last().context("BP history is empty")?;
    let mut selected_queries = BTreeMap::new();

    for var in QUERY_VARS {
        let key = var.to_string();
        let belief = final_record.selected_beliefs[&key];
        let exact_belief = exact[&var];

        selected_queries.insert(
            key,
            BpQueryResult {
                state0: belief[0],
                state1: belief[1],
                exact_state0: exact_belief[0],
                exact_state1: exact_belief[1],
                reference_mar_state1: published_mar[var][1],
                state0_absolute_error: (belief[0] - exact_belief[0]).abs(),
                state1_absolute_error: (belief[1] - exact_belief[1]).abs(),
            },
        );
    }

    let record = BpResult {
        model: "Promedus_14.uai".to_string(),
        method: "loopy_sum_product_bp".to_string(),
        damping: BP_DAMPING,
        tolerance: BP_TOLERANCE,
        max_sweeps: BP_MAX_SWEEPS,
        converged,
        sweeps: final_record.sweep,
        final_undamped_residual: final_record.undamped_residual,
        cumulative_update_work: final_record.cumulative_update_work,
        update_work_units_per_sweep: work_per_sweep,
        runtime_seconds,
        selected_queries,
        history,
    };

    fs::create_dir_all("artifacts/bp-promedus")?;
    fs::write(
        "artifacts/bp-promedus/run.json",
        serde_json::to_string_pretty(&record)?,
    )?;

    println!("\n===== PROMEDUS BP SUMMARY =====");
    println!("Converged: {}", record.converged);
    println!("Sweeps: {}", record.sweeps);
    println!(
        "Final undamped residual: {:.6e}",
        record.final_undamped_residual
    );
    println!(
        "Maximum selected-query error: {:.6e}",
        record
            .selected_queries
            .values()
            .map(|q| q.state0_absolute_error.max(q.state1_absolute_error))
            .fold(0.0_f64, f64::max)
    );
    println!("Cumulative update work: {}", record.cumulative_update_work);
    println!("Runtime: {:.3} s", record.runtime_seconds);

    for (var, q) in &record.selected_queries {
        println!(
            "Variable {}: BP state1={:.10}, exact state1={:.10}, error={:.6e}",
            var, q.state1, q.exact_state1, q.state1_absolute_error
        );
    }

    println!("Saved artifacts/bp-promedus/run.json");
    Ok(())
}

fn main() -> Result<()> {
    let mode = env::args().nth(1).unwrap_or_else(|| "all".to_string());

    match mode.as_str() {
        "exact" => run_exact(),
        "bp" => run_bp(),
        "all" => {
            run_exact()?;
            run_bp()?;
            println!("\nALL PROMEDUS TASKS FINISHED");
            Ok(())
        }
        other => {
            bail!(
                "Unknown mode '{}'. Use: all, exact, or bp",
                other
            );
        }
    }
}
