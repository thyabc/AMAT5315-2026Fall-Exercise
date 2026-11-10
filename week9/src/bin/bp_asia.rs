use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;

const VAR_NAMES: [&str; 8] = ["A", "T", "S", "L", "B", "E", "X", "D"];

#[derive(Clone)]
struct Factor {
    scope: Vec<usize>,
    values: Vec<f64>,
}

type Messages = Vec<Vec<[f64; 2]>>;

#[derive(Serialize)]
struct SweepRecord {
    sweep: usize,
    undamped_residual: f64,
    cumulative_update_work: u64,
    max_marginal_error: f64,
    beliefs: Vec<[f64; 2]>,
}

#[derive(Serialize)]
struct BpRecord {
    case: String,
    observations: BTreeMap<String, usize>,
    method: String,
    damping: f64,
    tolerance: f64,
    max_sweeps: usize,
    converged: bool,
    sweeps: usize,
    final_undamped_residual: f64,
    cumulative_update_work: u64,
    update_work_units_per_sweep: u64,
    update_work_definition: String,
    max_marginal_error: f64,
    final_beliefs: Vec<[f64; 2]>,
    disease_probabilities_state0: BTreeMap<String, f64>,
    disease_marginal_errors: BTreeMap<String, f64>,
    history: Vec<SweepRecord>,
}

fn load_model() -> Result<Vec<Factor>> {
    let content = fs::read_to_string("asia.uai")?;
    let mut it = content.split_whitespace();

    ensure!(it.next() == Some("MARKOV"), "Expected MARKOV");

    let n: usize = it.next().context("Missing variable count")?.parse()?;

    ensure!(n == 8, "Expected 8 ASIA variables");

    let mut dims = Vec::new();

    for _ in 0..n {
        let d: usize = it.next().context("Missing dimension")?.parse()?;
        ensure!(d == 2, "Expected binary variables");
        dims.push(d);
    }

    let nf: usize = it.next().context("Missing factor count")?.parse()?;

    let mut scopes = Vec::new();

    for _ in 0..nf {
        let rank: usize = it.next().context("Missing factor rank")?.parse()?;
        let mut scope = Vec::new();

        for _ in 0..rank {
            let v: usize = it.next().context("Missing variable ID")?.parse()?;
            ensure!(v < n, "Invalid variable ID");
            scope.push(v);
        }

        scopes.push(scope);
    }

    let mut factors = Vec::new();

    for scope in scopes {
        let count: usize = it.next().context("Missing table size")?.parse()?;

        let expected: usize = scope.iter().map(|_| 2).product();

        ensure!(count == expected, "Factor table size mismatch");

        let mut values = Vec::new();

        for _ in 0..count {
            values.push(it.next().context("Missing factor value")?.parse::<f64>()?);
        }

        factors.push(Factor { scope, values });
    }

    ensure!(it.next().is_none(), "Unexpected trailing UAI data");

    Ok(factors)
}

fn normalize(x: [f64; 2]) -> Result<[f64; 2]> {
    let sum = x[0] + x[1];

    ensure!(
        sum.is_finite() && sum > 0.0,
        "Cannot normalize a zero or invalid message"
    );

    Ok([x[0] / sum, x[1] / sum])
}

fn build_adjacency(factors: &[Factor], n_vars: usize) -> Vec<Vec<(usize, usize)>> {
    let mut adjacency = vec![Vec::new(); n_vars];

    for (fi, factor) in factors.iter().enumerate() {
        for (pos, &v) in factor.scope.iter().enumerate() {
            adjacency[v].push((fi, pos));
        }
    }

    adjacency
}

fn uniform_messages(factors: &[Factor]) -> Messages {
    factors
        .iter()
        .map(|f| vec![[0.5, 0.5]; f.scope.len()])
        .collect()
}

// Compute one synchronous, undamped BP update F(m).
//
// v -> f: product of all incoming factor messages except f.
// f -> v: sum over all other variables in the factor.
//
// Work units:
// - one multiplication per incoming message state,
// - one factor-table assignment contribution and its message products.
fn proposals(
    factors: &[Factor],
    adjacency: &[Vec<(usize, usize)>],
    v_to_f: &Messages,
    f_to_v: &Messages,
) -> Result<(Messages, Messages, u64)> {
    let mut next_vf = uniform_messages(factors);
    let mut next_fv = uniform_messages(factors);

    let mut work = 0_u64;

    // Variable-to-factor messages.
    for (fi, factor) in factors.iter().enumerate() {
        for (pos, &v) in factor.scope.iter().enumerate() {
            let mut raw = [1.0_f64, 1.0];

            for &(other_fi, other_pos) in &adjacency[v] {
                if other_fi == fi {
                    continue;
                }

                for s in 0..2 {
                    raw[s] *= f_to_v[other_fi][other_pos][s];
                    work += 1;
                }
            }

            next_vf[fi][pos] = normalize(raw)?;
        }
    }

    // Factor-to-variable messages.
    for (fi, factor) in factors.iter().enumerate() {
        let rank = factor.scope.len();

        for target in 0..rank {
            let mut sums = [0.0_f64, 0.0];
            let mut states = vec![0_usize; rank];

            for idx in 0..factor.values.len() {
                // Decode row-major binary assignment.
                let mut rem = idx;

                for p in (0..rank).rev() {
                    states[p] = rem % 2;
                    rem /= 2;
                }

                let mut weight = factor.values[idx];

                for p in 0..rank {
                    if p != target {
                        weight *= v_to_f[fi][p][states[p]];
                        work += 1;
                    }
                }

                sums[states[target]] += weight;
                work += 1;
            }

            next_fv[fi][target] = normalize(sums)?;
        }
    }

    Ok((next_vf, next_fv, work))
}

fn message_residual(old: &Messages, proposed: &Messages) -> f64 {
    let mut result = 0.0_f64;

    for fi in 0..old.len() {
        for pos in 0..old[fi].len() {
            for s in 0..2 {
                result = result.max((old[fi][pos][s] - proposed[fi][pos][s]).abs());
            }
        }
    }

    result.max(message_residual_single(old, proposed))
}

fn message_residual_single(old: &Messages, proposed: &Messages) -> f64 {
    let mut result = 0.0_f64;

    for fi in 0..old.len() {
        for pos in 0..old[fi].len() {
            for s in 0..2 {
                result = result.max((old[fi][pos][s] - proposed[fi][pos][s]).abs());
            }
        }
    }

    result
}

fn full_residual(
    old_vf: &Messages,
    old_fv: &Messages,
    new_vf: &Messages,
    new_fv: &Messages,
) -> f64 {
    message_residual(old_vf, new_vf).max(message_residual(old_fv, new_fv))
}

fn damp(old: &Messages, proposed: &Messages, eta: f64) -> Result<Messages> {
    let mut result = uniform_messages(
        &old.iter()
            .map(|m| Factor {
                scope: vec![0; m.len()],
                values: vec![1.0; 2_usize.pow(m.len() as u32)],
            })
            .collect::<Vec<_>>(),
    );

    for fi in 0..old.len() {
        for pos in 0..old[fi].len() {
            let raw = [
                (1.0 - eta) * old[fi][pos][0] + eta * proposed[fi][pos][0],
                (1.0 - eta) * old[fi][pos][1] + eta * proposed[fi][pos][1],
            ];

            result[fi][pos] = normalize(raw)?;
        }
    }

    Ok(result)
}

fn compute_beliefs(adjacency: &[Vec<(usize, usize)>], f_to_v: &Messages) -> Result<Vec<[f64; 2]>> {
    let mut beliefs = Vec::new();

    for edges in adjacency {
        let mut raw = [1.0_f64, 1.0];

        for &(fi, pos) in edges {
            for s in 0..2 {
                raw[s] *= f_to_v[fi][pos][s];
            }
        }

        beliefs.push(normalize(raw)?);
    }

    Ok(beliefs)
}

fn max_marginal_error(beliefs: &[[f64; 2]], exact: &[[f64; 2]]) -> f64 {
    beliefs
        .iter()
        .zip(exact.iter())
        .flat_map(|(a, b)| (0..2).map(move |s| (a[s] - b[s]).abs()))
        .fold(0.0_f64, f64::max)
}

fn run_case(
    label: &str,
    observations: Vec<(usize, usize)>,
    damping: f64,
    tolerance: f64,
    max_sweeps: usize,
) -> Result<()> {
    println!("\n====================================");
    println!("BP CASE: {}", label);
    println!("====================================");

    let mut factors = load_model()?;

    let mut observation_names = BTreeMap::new();

    // Apply evidence as unary indicator factors.
    // This is equivalent to multiplying the original factor
    // network by the corresponding observation indicators.
    for &(v, state) in &observations {
        ensure!(state < 2, "Invalid evidence state");

        let mut indicator = vec![0.0, 0.0];
        indicator[state] = 1.0;

        factors.push(Factor {
            scope: vec![v],
            values: indicator,
        });

        observation_names.insert(VAR_NAMES[v].to_string(), state);
    }

    let adjacency = build_adjacency(&factors, 8);

    let exact_path = format!("artifacts/{}/run.json", label);

    let exact_json: Value = serde_json::from_str(
        &fs::read_to_string(&exact_path)
            .with_context(|| format!("Missing exact result: {}", exact_path))?,
    )?;

    let exact: Vec<[f64; 2]> = serde_json::from_value(exact_json["marginals_contraction"].clone())?;

    ensure!(exact.len() == 8, "Expected 8 exact marginals");

    // All directed messages start uniformly.
    let mut v_to_f = uniform_messages(&factors);
    let mut f_to_v = uniform_messages(&factors);

    let mut history = Vec::new();
    let mut cumulative_work = 0_u64;
    let mut work_per_sweep = 0_u64;
    let mut converged = false;

    // sweep=0 is the initial uniform-message state.
    // sweep=1..max_sweeps are completed damped updates.
    // Residual and beliefs are always evaluated at the same state.
    for completed_sweeps in 0..=max_sweeps {
        let (proposal_vf, proposal_fv, work) = proposals(&factors, &adjacency, &v_to_f, &f_to_v)?;

        work_per_sweep = work;

        let residual = full_residual(&v_to_f, &f_to_v, &proposal_vf, &proposal_fv);

        let beliefs = compute_beliefs(&adjacency, &f_to_v)?;

        let error = max_marginal_error(&beliefs, &exact);

        history.push(SweepRecord {
            sweep: completed_sweeps,
            undamped_residual: residual,
            cumulative_update_work: cumulative_work,
            max_marginal_error: error,
            beliefs: beliefs.clone(),
        });

        if completed_sweeps % 10 == 0 || residual < tolerance || completed_sweeps == max_sweeps {
            println!(
                "{} sweep {:4}: residual={:.4e}, marginal_error={:.6e}, update_work={}",
                label, completed_sweeps, residual, error, cumulative_work
            );
        }

        if residual < tolerance {
            converged = true;
            break;
        }

        if completed_sweeps == max_sweeps {
            break;
        }

        // Synchronous damped update:
        // both directions use the same previous message state.
        v_to_f = damp(&v_to_f, &proposal_vf, damping)?;
        f_to_v = damp(&f_to_v, &proposal_fv, damping)?;

        // Count work for the completed message update.
        cumulative_work += work;
    }

    let final_state = history.last().context("Missing BP history")?;

    let mut disease_probabilities = BTreeMap::new();
    let mut disease_errors = BTreeMap::new();

    for v in [1_usize, 3, 4] {
        disease_probabilities.insert(VAR_NAMES[v].to_string(), final_state.beliefs[v][0]);

        disease_errors.insert(
            VAR_NAMES[v].to_string(),
            (final_state.beliefs[v][0] - exact[v][0]).abs(),
        );
    }

    let record = BpRecord {
        case: label.to_string(),
        observations: observation_names,
        method: "loopy_sum_product_bp".to_string(),
        damping,
        tolerance,
        max_sweeps,
        converged,
        sweeps: final_state.sweep,
        final_undamped_residual: final_state.undamped_residual,
        cumulative_update_work: final_state.cumulative_update_work,
        update_work_units_per_sweep: work_per_sweep,
        update_work_definition:
            "Incoming-message multiplications plus factor-table assignment products and accumulations; normalization overhead excluded."
                .to_string(),
        max_marginal_error: final_state.max_marginal_error,
        final_beliefs: final_state.beliefs.clone(),
        disease_probabilities_state0: disease_probabilities,
        disease_marginal_errors: disease_errors,
        history,
    };

    let out_dir = format!("artifacts/bp-{}", label);
    fs::create_dir_all(&out_dir)?;

    fs::write(
        format!("{}/run.json", out_dir),
        serde_json::to_string_pretty(&record)?,
    )?;

    println!("\n===== BP SUMMARY: {} =====", label);
    println!("Converged: {}", converged);
    println!("Sweeps: {}", record.sweeps);
    println!(
        "Final undamped residual: {:.6e}",
        record.final_undamped_residual
    );
    println!("Maximum marginal error: {:.6e}", record.max_marginal_error);
    println!("Cumulative update work: {}", record.cumulative_update_work);

    for (name, probability) in &record.disease_probabilities_state0 {
        println!("P({}=0) = {:.10}", name, probability);
    }

    println!("Saved {}/run.json", out_dir);

    Ok(())
}

fn main() -> Result<()> {
    let damping = 0.5;
    let tolerance = 1e-10;
    let max_sweeps = 1000;

    println!("====================================");
    println!(" WEEK 9: ASIA LOOPY BELIEF PROPAGATION");
    println!("====================================");

    println!("Damping: {}", damping);
    println!("Tolerance: {:.1e}", tolerance);
    println!("Maximum sweeps: {}", max_sweeps);

    run_case("prior", vec![], damping, tolerance, max_sweeps)?;

    run_case("AD", vec![(0, 0), (7, 0)], damping, tolerance, max_sweeps)?;

    run_case(
        "ADX",
        vec![(0, 0), (7, 0), (6, 0)],
        damping,
        tolerance,
        max_sweeps,
    )?;

    println!("\n====================================");
    println!(" ALL BP CASES FINISHED");
    println!("====================================");

    Ok(())
}
