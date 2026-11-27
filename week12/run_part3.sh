#!/usr/bin/env bash
# Run inside the existing course repository's week12/ after Part 2.
set -Eeuo pipefail
trap 'printf "\nStopped at line %s. Send the last terminal output to ChatGPT.\n" "$LINENO" >&2' ERR
if [[ "$(basename "$PWD")" != "week12" ]]; then
    printf 'Please cd into your existing week12 folder first.\n' >&2
    exit 1
fi
for tool in cargo rustc cmake gcc g++ git python3 uv; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'Missing tool: %s\n' "$tool" >&2
        exit 1
    fi
done
for file in Cargo.toml Cargo.lock src/main.rs src/bnb.rs branchopt.design.toml README.md; do
    if [[ ! -f "$file" ]]; then
        printf 'Missing Part 2 file: %s\n' "$file" >&2
        exit 1
    fi
done
python3 - <<'PYFIRST'
from pathlib import Path
import json
for policy,folder in [('most-frac','bnb_most'),('strong','bnb_strong')]:
    p=Path('artifacts')/folder/'run.json'
    if not p.exists():
        raise SystemExit(f'Complete Part 2 first: {p} is missing.')
    r=json.loads(p.read_text())
    if r.get('validation_status')!='passed' or not r['summary'].get('lp_accounting_passed'):
        raise SystemExit(f'Part 2 validation must pass: {p}')
    a=r['parameters']
    if (a['spins'],a['edges'],a['count'],a['seed'],a['policy'])!=(22,52,48,2026,policy):
        raise SystemExit(f'Unexpected Part 2 settings: {p}')
    print(f"Existing {policy}: PASSED; mean nodes={r['summary']['mean_nodes']}")
PYFIRST
git rev-parse --show-toplevel >/dev/null
mkdir -p src artifacts/bnb_labels artifacts/eval_test
backup="artifacts/setup_backups/part3-$(date +%Y%m%dT%H%M%S)-$$"
mkdir -p "$backup"
for file in src/main.rs src/bnb.rs src/nn.rs train_branch.py train_branch.design.toml plot_neural_bnb.py branchopt.design.toml README.md DEEPSEEK_REVIEW.md run_part3.sh artifacts/model.json artifacts/bnb_labels/run.json artifacts/bnb_labels/dataset.json artifacts/bnb_strong/dataset.json artifacts/eval_test/run.json artifacts/eval_test/parity.json artifacts/eval_test/part3_checks.json artifacts/neural_bnb.png; do
    if [[ -f "$file" ]]; then
        mkdir -p "$backup/$(dirname "$file")"
        cp -- "$file" "$backup/$file"
    fi
done
if [[ "$(realpath "$0")" != "$(realpath -m run_part3.sh)" ]]; then
    cp -- "$0" run_part3.sh
fi
printf '\n[1/6] Adding label export, JAX training and Rust inference; existing files are backed up...\n'
cat > 'src/main.rs' <<'WEEK12_PART3_FILE'
mod bnb;
mod nn;
use highs::{HighsModelStatus, Model, RowProblem, Sense};
use serde::Serialize;
use serde_json::{json, Value};
use std::{collections::BTreeSet, env, error::Error, fs, io, path::PathBuf, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn fail(message: impl Into<String>) -> Box<dyn Error> {
    io::Error::new(io::ErrorKind::InvalidInput, message.into()).into()
}
fn check(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(fail(message)) }
}

#[derive(Clone, Debug, Serialize)]
struct Edge { u: usize, v: usize, j: f64 }
#[derive(Clone, Debug, Serialize)]
struct Graph { n: usize, edges: Vec<Edge>, h: Vec<f64> }
#[derive(Clone, Debug, Serialize)]
struct Solve { energy: f64, x: Vec<f64>, z: Vec<f64>, runtime_ms: f64 }

// All arithmetic in the generator wraps modulo 2^64.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn index(&mut self, k: usize) -> usize { (self.next() % k as u64) as usize }
}

fn generate(n: usize, m: usize, seed: u64, id: usize) -> Result<(u64, Graph)> {
    let mut seeder = SplitMix64(seed ^ (id as u64).wrapping_mul(0xD1B54A32D192ED03));
    let instance_seed = seeder.next();
    let mut rng = SplitMix64(instance_seed);
    let mut pairs = BTreeSet::new();
    let mut degree = vec![0usize; n];
    for i in 0..n {
        while degree[i] < 2.min(n - 1) {
            let j = rng.index(n);
            if i != j && pairs.insert((i.min(j), i.max(j))) {
                degree[i] += 1;
                degree[j] += 1;
            }
        }
    }
    check(pairs.len() <= m, "minimum-degree construction exceeded requested edge count")?;
    while pairs.len() < m {
        let i = rng.index(n);
        let j = rng.index(n);
        if i != j { pairs.insert((i.min(j), i.max(j))); }
    }
    // BTreeSet visits edges in the required lexicographic order.
    let edges = pairs.into_iter().map(|(u, v)| Edge {
        u, v, j: if rng.next() & 1 == 0 { 1.0 } else { -1.0 },
    }).collect();
    let h = (0..n).map(|_| ((rng.next() % 3) as i64 - 1) as f64).collect();
    Ok((instance_seed, Graph { n, edges, h }))
}

fn configure(model: &mut Model) {
    model.make_quiet();
    model.set_option("presolve", "off");
    model.set_option("threads", 1);
    model.set_option("mip_rel_gap", 0.0);
    model.set_option("mip_abs_gap", 0.0);
}

fn rounded(x: &[f64]) -> Vec<u8> {
    x.iter().map(|&v| if v >= 0.5 { 1 } else { 0 }).collect()
}
fn physical_energy(g: &Graph, bits: &[u8]) -> f64 {
    let s: Vec<f64> = bits.iter().map(|&x| 1.0 - 2.0 * x as f64).collect();
    g.edges.iter().map(|e| e.j * s[e.u] * s[e.v]).sum::<f64>()
        + g.h.iter().zip(s.iter()).map(|(h, s)| h * s).sum::<f64>()
}

fn solve_spin(g: &Graph, integer: bool) -> Result<Solve> {
    let start = Instant::now();
    let mut pb = RowProblem::new();
    let x: Vec<_> = g.h.iter().map(|&h| {
        if integer { pb.add_integer_column(-2.0 * h, 0.0..=1.0) }
        else { pb.add_column(-2.0 * h, 0.0..=1.0) }
    }).collect();
    for e in &g.edges {
        // z is XOR (bond disagreement), NOT the product x_u*x_v.
        let z = pb.add_column(-2.0 * e.j, 0.0..=1.0);
        pb.add_row(..=0.0, &[(z, 1.0), (x[e.u], -1.0), (x[e.v], -1.0)]);
        pb.add_row(..=2.0, &[(z, 1.0), (x[e.u], 1.0), (x[e.v], 1.0)]);
        pb.add_row(..=0.0, &[(x[e.u], 1.0), (x[e.v], -1.0), (z, -1.0)]);
        pb.add_row(..=0.0, &[(x[e.u], -1.0), (x[e.v], 1.0), (z, -1.0)]);
    }
    let mut model = pb.optimise(Sense::Minimise);
    configure(&mut model);
    let solved = model.solve();
    check(solved.status() == HighsModelStatus::Optimal,
          &format!("HiGHS did not prove optimality: {:?}", solved.status()))?;
    let values = solved.get_solution().columns().to_vec();
    let xs = values[..g.n].to_vec();
    let zs = values[g.n..].to_vec();
    let constant = g.h.iter().sum::<f64>() + g.edges.iter().map(|e| e.j).sum::<f64>();
    let energy = constant
        - 2.0 * g.h.iter().zip(&xs).map(|(h, x)| h * x).sum::<f64>()
        - 2.0 * g.edges.iter().zip(&zs).map(|(e, z)| e.j * z).sum::<f64>();
    check(values.iter().all(|v| v.is_finite() && *v >= -1e-6 && *v <= 1.0 + 1e-6),
          "invalid primal variable bounds")?;
    if integer {
        check(xs.iter().all(|v| (v - v.round()).abs() <= 1e-6), "fractional MILP spin")?;
        let bits = rounded(&xs);
        check((physical_energy(g, &bits) - energy).abs() <= 1e-6,
              "MILP objective disagrees with the original spin Hamiltonian")?;
        for (e, z) in g.edges.iter().zip(&zs) {
            check((*z - (bits[e.u] ^ bits[e.v]) as f64).abs() <= 1e-6,
                  "integer spins did not force continuous z to XOR")?;
        }
    }
    Ok(Solve { energy, x: xs, z: zs, runtime_ms: start.elapsed().as_secs_f64() * 1e3 })
}

fn solve_workshop(integer: bool) -> Result<Value> {
    let mut pb = RowProblem::new();
    let x = if integer { pb.add_integer_column(5.0, 0.0..) }
            else { pb.add_column(5.0, 0.0..) };
    let y = if integer { pb.add_integer_column(6.0, 0.0..) }
            else { pb.add_column(6.0, 0.0..) };
    pb.add_row(..=5.0, &[(x, 1.0), (y, 1.0)]);
    pb.add_row(..=28.0, &[(x, 4.0), (y, 7.0)]);
    let mut model = pb.optimise(Sense::Maximise);
    configure(&mut model);
    let solved = model.solve();
    check(solved.status() == HighsModelStatus::Optimal, "workshop solve was not optimal")?;
    let a = solved.get_solution().columns().to_vec();
    let r = [a[0].round(), a[1].round()];
    Ok(json!({"x": a, "objective": 5.0*a[0] + 6.0*a[1],
              "nearest_rounding": r,
              "nearest_rounding_feasible": r[0]+r[1] <= 5.0+1e-6 && 4.0*r[0]+7.0*r[1] <= 28.0+1e-6}))
}

#[derive(Serialize)]
struct Record {
    id: usize, instance_seed: u64, graph: Graph, lp: Solve, milp: Solve,
    gap: f64, relative_gap: Option<f64>, fractional_spins: usize,
    rounded_bits: Vec<u8>, rounded_energy: f64, rounding_excess: f64,
}
fn mean(v: impl Iterator<Item = f64>, n: usize) -> f64 { v.sum::<f64>() / n as f64 }
fn metric(name: &str, value: impl std::fmt::Display, ms: f64) {
    println!("gap\t{name}\t{value}\t{ms:.6}");
}

#[derive(Serialize)]
struct Args { mode: String, policy: String, spins: usize, edges: usize, count: usize,
    seed: u64, out: PathBuf, export_labels: bool, model: Option<PathBuf> }
fn arguments() -> Result<Args> {
    let mut a = Args { mode: "gap".into(), policy: "most-frac".into(), spins: 22, edges: 52, count: 16,
                       seed: 2026, out: "artifacts/gap".into(), export_labels: false, model: None };
    let mut it = env::args().skip(1);
    while let Some(k) = it.next() {
        if k == "--help" {
            println!("branchopt --mode gap|bnb|eval --policy most-frac|strong|neural --export-labels --model artifacts/model.json --spins 22 --edges 52 --count 48 --seed 2026 --out artifacts/bnb_strong");
            std::process::exit(0);
        }
        if k == "--export-labels" { a.export_labels = true; continue; }
        let v = it.next().ok_or_else(|| fail(format!("missing value for {k}")))?;
        match k.as_str() {
            "--mode" => a.mode = v, "--policy" => a.policy = v, "--spins" => a.spins = v.parse()?,
            "--edges" => a.edges = v.parse()?, "--count" => a.count = v.parse()?,
            "--seed" => a.seed = v.parse()?, "--out" => a.out = v.into(),
            "--model" => a.model = Some(v.into()),
            _ => return Err(fail(format!("unknown option {k}"))),
        }
    }
    check(["gap","bnb","eval"].contains(&a.mode.as_str()), "supported modes: gap | bnb | eval")?;
    check(["most-frac","strong","neural"].contains(&a.policy.as_str()), "unsupported branching policy")?;
    check(!a.export_labels || (a.mode=="bnb" && a.policy=="strong" && a.count<=48),
          "--export-labels requires bnb mode, strong policy and count<=48")?;
    check((a.mode!="eval" && a.policy!="neural") || a.model.is_some(),
          "neural policy and eval mode require --model")?;
    check(a.spins >= 3 && a.spins <= 10000 && a.count > 0, "invalid spins or count")?;
    check(a.edges >= a.spins && a.edges <= a.spins * (a.spins - 1) / 2,
          "edge count must be between n and n*(n-1)/2")?;
    Ok(a)
}

fn main() -> Result<()> {
    let a = arguments()?;
    if a.mode == "bnb" { return bnb::run(&a); }
    if a.mode == "eval" { return bnb::evaluate(&a); }
    let started = Instant::now();
    fs::create_dir_all(&a.out)?;
    let square = Graph { n: 4, h: vec![0.0; 4], edges: vec![
        Edge { u: 0, v: 1, j: 1.0 }, Edge { u: 1, v: 2, j: 1.0 },
        Edge { u: 2, v: 3, j: 1.0 }, Edge { u: 0, v: 3, j: -1.0 }] };
    let slp = solve_spin(&square, false)?;
    let sip = solve_spin(&square, true)?;
    let sq_round = physical_energy(&square, &rounded(&slp.x));
    let exhaustive = (0..16).map(|mask| {
        let bits: Vec<u8> = (0..4).map(|i| ((mask >> i) & 1) as u8).collect();
        physical_energy(&square, &bits)
    }).fold(f64::INFINITY, f64::min);
    check((slp.energy + 4.0).abs() < 1e-6 && (sip.energy + 2.0).abs() < 1e-6
          && (sq_round - 2.0).abs() < 1e-6 && (exhaustive - sip.energy).abs() < 1e-6,
          "opening square checks failed")?;
    let wlp = solve_workshop(false)?;
    let wip = solve_workshop(true)?;
    check((wlp["objective"].as_f64().unwrap() - 83.0/3.0).abs() < 1e-6
          && (wip["objective"].as_f64().unwrap() - 27.0).abs() < 1e-6
          && wlp["nearest_rounding_feasible"] == json!(false), "workshop checks failed")?;
    println!("mode\tmetric\tvalue\truntime_ms");
    metric("square_lp", slp.energy, slp.runtime_ms);
    metric("square_integer", sip.energy, sip.runtime_ms);
    metric("square_rounded", sq_round, 0.0);
    metric("square_lp_x", serde_json::to_string(&slp.x)?, 0.0);
    metric("square_integer_x", serde_json::to_string(&sip.x)?, 0.0);
    metric("workshop_lp", &wlp, 0.0);
    metric("workshop_integer", &wip, 0.0);
    let mut records = Vec::new();
    for id in 0..a.count {
        let (instance_seed, graph) = generate(a.spins, a.edges, a.seed, id)?;
        let lp = solve_spin(&graph, false)?;
        let milp = solve_spin(&graph, true)?;
        let rounded_bits = rounded(&lp.x);
        let rounded_energy = physical_energy(&graph, &rounded_bits);
        let gap = milp.energy - lp.energy;
        let excess = rounded_energy - milp.energy;
        check(gap >= -1e-6 && excess >= -1e-6, "invalid lower or upper energy bound")?;
        let fractional_spins = lp.x.iter().filter(|&&v| v > 1e-6 && v < 1.0-1e-6).count();
        metric(&format!("instance_{id:02}_lp"), lp.energy, lp.runtime_ms);
        metric(&format!("instance_{id:02}_integer"), milp.energy, milp.runtime_ms);
        records.push(Record { id, instance_seed, graph, gap,
            relative_gap: if milp.energy.abs() > 1e-12 { Some(gap/milp.energy.abs()) } else { None },
            fractional_spins, rounded_bits, rounded_energy, rounding_excess: excess, lp, milp });
    }
    let lp_mean = mean(records.iter().map(|r| r.lp.energy), a.count);
    let exact_mean = mean(records.iter().map(|r| r.milp.energy), a.count);
    let gap_mean = mean(records.iter().map(|r| r.gap), a.count);
    let frac_mean = mean(records.iter().map(|r| r.fractional_spins as f64), a.count);
    let excess_mean = mean(records.iter().map(|r| r.rounding_excess), a.count);
    let suboptimal = records.iter().filter(|r| r.rounding_excess > 1e-6).count();
    let official = a.spins == 22 && a.edges == 52 && a.count == 16 && a.seed == 2026;
    let reference_ok = (lp_mean + 52.0).abs() < 1e-3 && (exact_mean + 38.25).abs() < 1e-3
        && (gap_mean - 13.75).abs() < 1e-3 && (excess_mean - 34.25).abs() < 1e-3
        && (frac_mean - 22.0).abs() < 1e-6 && suboptimal == 16;
    let summary = json!({"mean_lp_bound": lp_mean, "mean_exact_energy": exact_mean,
        "mean_integrality_gap": gap_mean, "mean_fractional_spins": frac_mean,
        "suboptimal_roundings": suboptimal, "mean_rounding_excess": excess_mean,
        "official_reference_applicable": official,
        "official_reference_passed": if official { Some(reference_ok) } else { None }});
    let elapsed = started.elapsed().as_secs_f64() * 1e3;
    let run = json!({"schema": "branchopt.gap.v1", "mode": "gap", "parameters": a,
        "validation_status": if official && !reference_ok { "failed_reference" } else { "passed" },
        "solver": {"crate": "highs=2.4.0", "presolve": false, "threads": 1,
                   "mip_rel_gap": 0.0, "mip_abs_gap": 0.0},
        "square": {"graph": square, "lp": slp, "milp": sip,
                    "rounded_energy": sq_round, "bruteforce_energy": exhaustive},
        "workshop": {"lp": wlp, "milp": wip}, "summary": summary,
        "instances": records, "runtime_ms": elapsed});
    fs::write(a.out.join("run.json"), serde_json::to_vec_pretty(&run)?)?;
    for (name, value) in summary.as_object().unwrap() { metric(name, value, elapsed); }
    // A reference mismatch is reported AFTER preserving the actual raw evidence.
    check(!official || reference_ok, "reference mismatch; inspect the saved run.json before proceeding")?;
    eprintln!("Saved {}", a.out.join("run.json").display());
    Ok(())
}
WEEK12_PART3_FILE
cat > 'src/bnb.rs' <<'WEEK12_PART3_FILE'
//! Exact depth-first Branch-and-Bound. Trial LPs never mutate search state.
use super::*;

const EPS: f64 = 1e-6;
const TIE: f64 = 1e-12;

struct Lp { energy: f64, x: Vec<f64>, z: Vec<f64>, duals: Vec<f64> }

fn solve_lp(g: &Graph, fixed: &[i8]) -> Result<Option<Lp>> {
    let mut pb = RowProblem::new();
    let x: Vec<_> = g.h.iter().enumerate().map(|(i, &h)| {
        let (lo, hi) = if fixed[i] < 0 { (0.0, 1.0) }
                       else { (fixed[i] as f64, fixed[i] as f64) };
        // Node relaxations contain ONLY continuous variables.
        pb.add_column(-2.0 * h, lo..=hi)
    }).collect();
    for e in &g.edges {
        let z = pb.add_column(-2.0 * e.j, 0.0..=1.0);
        pb.add_row(..=0.0, &[(z,1.0),(x[e.u],-1.0),(x[e.v],-1.0)]);
        pb.add_row(..=2.0, &[(z,1.0),(x[e.u],1.0),(x[e.v],1.0)]);
        pb.add_row(..=0.0, &[(x[e.u],1.0),(x[e.v],-1.0),(z,-1.0)]);
        pb.add_row(..=0.0, &[(x[e.u],-1.0),(x[e.v],1.0),(z,-1.0)]);
    }
    let mut model = pb.optimise(Sense::Minimise);
    configure(&mut model);
    let solved = model.solve();
    if solved.status() == HighsModelStatus::Infeasible { return Ok(None); }
    check(solved.status() == HighsModelStatus::Optimal,
          &format!("node LP has unexpected status {:?}", solved.status()))?;
    let solution=solved.get_solution();
    let v = solution.columns().to_vec();
    let duals = solution.dual_rows().to_vec();
    let energy = g.h.iter().sum::<f64>() + g.edges.iter().map(|e|e.j).sum::<f64>()
        - 2.0*g.h.iter().zip(&v[..g.n]).map(|(h,x)|h*x).sum::<f64>()
        - 2.0*g.edges.iter().zip(&v[g.n..]).map(|(e,z)|e.j*z).sum::<f64>();
    check(energy.is_finite(), "node LP returned a non-finite energy")?;
    Ok(Some(Lp { energy, x: v[..g.n].to_vec(), z: v[g.n..].to_vec(), duals }))
}

#[derive(Serialize)]
struct LabelNode {
    node:usize, candidates:Vec<usize>, features:Vec<Vec<f64>>,
    gains:Vec<[f64;2]>, scores:Vec<f64>, pruned_children:Vec<u8>,
    fixed:Vec<i8>, lp_x:Vec<f64>, lp_z:Vec<f64>, row_duals:Vec<f64>,
    lp_bound:f64, incumbent:Option<f64>, chosen_spin:usize,
}

fn feature_rows(g:&Graph,lp:&Lp,fixed:&[i8],incumbent:f64,candidates:&[usize]) -> Vec<Vec<f64>> {
    let x:Vec<f64>=lp.x.iter().map(|v|v.clamp(0.0,1.0)).collect();
    let spins:Vec<f64>=x.iter().map(|v|1.0-2.0*v).collect();
    let mut incident=vec![Vec::new();g.n];
    for (k,e) in g.edges.iter().enumerate() {
        incident[e.u].push((k,e.v)); incident[e.v].push((k,e.u));
    }
    let dmax=incident.iter().map(Vec::len).max().unwrap_or(0).max(1) as f64;
    let dual_scale=lp.duals.iter().map(|v|v.abs()).fold(1e-9_f64,f64::max);
    let gap_ratio=if incumbent.is_finite() {
        ((incumbent-lp.energy)/incumbent.abs().max(1.0)).clamp(0.0,2.0)
    } else {1.0};
    candidates.iter().map(|&i| {
        let degree=incident[i].len() as f64;
        let denom=degree.max(1.0);
        let (mut bias,mut effective,mut dual,mut gap,mut neighbor_frac)=(0.0, g.h[i], 0.0, 0.0, 0.0);
        for &(k,j) in &incident[i] {
            let e=&g.edges[k];
            bias+=if e.j>0.0 {1.0} else if e.j<0.0 {-1.0} else {0.0};
            effective+=e.j*spins[j];
            dual+=lp.duals[4*k..4*k+4].iter().map(|v|v.abs()).sum::<f64>()/dual_scale;
            gap+=(lp.z[k]-0.5*(1.0-spins[i]*spins[j])).abs();
            if fixed[j]<0 {neighbor_frac+=2.0*x[j].min(1.0-x[j]);}
        }
        effective/=degree+1.0;
        vec![x[i],2.0*x[i].min(1.0-x[i]),g.h[i],g.h[i].abs(),degree/dmax,bias/denom,
             effective,effective.abs(),dual/denom,gap/denom,neighbor_frac/denom,gap_ratio]
    }).collect()
}

#[derive(Clone)]
struct Node { fixed: Vec<i8>, parent: Option<usize>, depth: usize, branch: Option<(usize,u8)> }

#[derive(Serialize)]
struct SearchResult {
    energy: f64, bits: Vec<u8>, nodes: usize, lp_solves: usize,
    node_lp_solves: usize, trial_lp_solves: usize,
    total_ms: f64, lp_ms: f64, inference_ms: f64, first_splits: Vec<usize>, trace: Vec<Value>,
    #[serde(skip_serializing)] labels: Vec<LabelNode>,
}

fn search(g: &Graph, policy: &str, model: Option<&nn::Network>, export_labels: bool) -> Result<SearchResult> {
    check(["most-frac","strong","neural"].contains(&policy), "unsupported B&B policy")?;
    check(policy!="neural" || model.is_some(), "neural policy requires a model")?;
    let start = Instant::now();
    let mut stack = vec![Node { fixed: vec![-1;g.n], parent: None, depth: 0, branch: None }];
    let mut incumbent = f64::INFINITY;
    let mut best_bits = Vec::new();
    let (mut nodes, mut trials) = (0usize,0usize);
    let mut lp_ms = 0.0;
    let mut trace = Vec::new();
    let mut first_splits = Vec::new();
    let mut labels=Vec::new();
    let mut inference_ms=0.0;
    while let Some(node) = stack.pop() {
        nodes += 1;
        let lp_start = Instant::now();
        let result = solve_lp(g,&node.fixed)?;
        lp_ms += lp_start.elapsed().as_secs_f64()*1e3;
        let mut audit = json!({"node":nodes,"parent":node.parent,"depth":node.depth,
            "branch":node.branch,"fixed":node.fixed,
            "incumbent_before": if incumbent.is_finite(){Some(incumbent)}else{None}});
        let Some(lp) = result else {
            audit["status"] = json!("infeasible");
            trace.push(audit);
            continue;
        };
        audit["lp_bound"] = json!(lp.energy);
        audit["x"] = json!(lp.x);
        // For minimization the LP is a LOWER bound, incumbent an UPPER bound.
        if lp.energy >= incumbent-EPS {
            audit["status"] = json!("pruned_bound");
            trace.push(audit);
            continue;
        }
        let candidates: Vec<usize> = (0..g.n).filter(|&i|
            node.fixed[i]<0 && lp.x[i]>EPS && lp.x[i]<1.0-EPS).collect();
        if candidates.is_empty() {
            let bits = rounded(&lp.x);
            check(lp.x.iter().all(|v|(v-v.round()).abs()<=EPS), "integral node has fractional spins")?;
            let energy = physical_energy(g,&bits);
            check((energy-lp.energy).abs()<=EPS, "integral-node energy disagrees with Hamiltonian")?;
            incumbent = energy;
            best_bits = bits;
            audit["status"] = json!("integer");
            audit["incumbent_after"] = json!(incumbent);
            trace.push(audit);
            continue;
        }
        let record_label=export_labels && policy=="strong" && candidates.len()>=2 && labels.len()<80;
        let mut features=Vec::new();
        if record_label || policy=="neural" {
            let t=Instant::now();
            features=feature_rows(g,&lp,&node.fixed,incumbent,&candidates);
            inference_ms+=t.elapsed().as_secs_f64()*1e3;
        }
        let mut chosen = candidates[0];
        let mut strong_audit = Vec::new();
        if policy == "most-frac" {
            let mut best_distance = (lp.x[chosen]-0.5).abs();
            for &i in candidates.iter().skip(1) {
                let distance = (lp.x[i]-0.5).abs();
                if distance < best_distance-TIE { chosen=i; best_distance=distance; }
            }
        } else if policy=="strong" {
            let mut best_score: f64 = -1.0;
            // This immutable snapshot is used for EVERY trial at this node.
            let incumbent_at_node = incumbent;
            for &i in &candidates {
                let mut gains = [0.0_f64;2];
                let mut child_bounds = [None;2];
                let mut eliminated = [false;2];
                for value in 0..2 {
                    let mut fixed = node.fixed.clone();
                    fixed[i]=value as i8;
                    trials+=1;
                    let trial_start=Instant::now();
                    let child=solve_lp(g,&fixed)?;
                    lp_ms+=trial_start.elapsed().as_secs_f64()*1e3;
                    if let Some(child)=child {
                        child_bounds[value]=Some(child.energy);
                        eliminated[value]=child.energy>=incumbent_at_node-EPS;
                        gains[value]=if eliminated[value] {1e6} else {child.energy-lp.energy};
                    } else { eliminated[value]=true; gains[value]=1e6; }
                    // No incumbent update, node push, cache reuse or bound mutation here.
                }
                let score=gains[0].max(EPS)*gains[1].max(EPS);
                strong_audit.push(json!({"spin":i,"gains":gains,"score":score,
                    "child_lp_bounds":child_bounds,"eliminated":eliminated}));
                if score > best_score+TIE*best_score.abs().max(1.0) {
                    best_score=score; chosen=i;
                }
            }
        }
        if policy=="neural" {
            let t=Instant::now();
            let scores=model.unwrap().scores(&features)?;
            let mut best=f64::NEG_INFINITY;
            for (k,&score) in scores.iter().enumerate() {
                if score>best+TIE {best=score;chosen=candidates[k];}
            }
            inference_ms+=t.elapsed().as_secs_f64()*1e3;
        }
        if record_label {
            let gains:Vec<[f64;2]>=strong_audit.iter().map(|v|[
                v["gains"][0].as_f64().unwrap(),v["gains"][1].as_f64().unwrap()]).collect();
            let scores=strong_audit.iter().map(|v|v["score"].as_f64().unwrap()).collect();
            let pruned_children=strong_audit.iter().map(|v|v["eliminated"].as_array().unwrap()
                .iter().filter(|b|b.as_bool()==Some(true)).count() as u8).collect();
            labels.push(LabelNode{node:nodes,candidates:candidates.clone(),features,gains,scores,
                pruned_children,fixed:node.fixed.clone(),lp_x:lp.x.clone(),lp_z:lp.z.clone(),
                row_duals:lp.duals.clone(),lp_bound:lp.energy,
                incumbent:if incumbent.is_finite(){Some(incumbent)}else{None},chosen_spin:chosen});
        }
        if first_splits.len()<2 { first_splits.push(chosen); }
        audit["status"]=json!("branched");
        audit["chosen_spin"]=json!(chosen);
        audit["candidates"]=json!(candidates);
        if policy=="strong" { audit["strong_trials"]=json!(strong_audit); }
        // LIFO: push the farther child first, then the nearer child.
        // In particular an exact 0.5 visits x_i=1 first.
        let near=if lp.x[chosen]>=0.5 {1u8}else{0u8};
        for value in [1-near,near] {
            let mut fixed=node.fixed.clone();
            fixed[chosen]=value as i8;
            stack.push(Node{fixed,parent:Some(nodes),depth:node.depth+1,branch:Some((chosen,value))});
        }
        trace.push(audit);
    }
    let total_ms=start.elapsed().as_secs_f64()*1e3;
    check(incumbent.is_finite() && best_bits.len()==g.n,"search finished without a feasible incumbent")?;
    Ok(SearchResult { energy:incumbent,bits:best_bits,nodes,lp_solves:nodes+trials,
        node_lp_solves:nodes,trial_lp_solves:trials,total_ms,lp_ms,inference_ms,first_splits,trace,labels })
}

fn workshop_trace() -> Result<Value> {
    // Generic LP construction at each visited box, with actual pruning.
    let mut stack=vec![([0.0,0.0],[f64::INFINITY,f64::INFINITY],"root".to_string())];
    let mut best=f64::NEG_INFINITY;
    let mut rows=Vec::new();
    while let Some((lo,hi,branch))=stack.pop() {
        let mut pb=RowProblem::new();
        let x=pb.add_column(5.0,lo[0]..=hi[0]);
        let y=pb.add_column(6.0,lo[1]..=hi[1]);
        pb.add_row(..=5.0,&[(x,1.0),(y,1.0)]);
        pb.add_row(..=28.0,&[(x,4.0),(y,7.0)]);
        let mut model=pb.optimise(Sense::Maximise);
        configure(&mut model);
        let solved=model.solve();
        if solved.status()==HighsModelStatus::Infeasible {
            rows.push(json!({"branch":branch,"status":"infeasible"}));
            continue;
        }
        check(solved.status()==HighsModelStatus::Optimal,"workshop node not optimal")?;
        let values=solved.get_solution().columns().to_vec();
        let objective=5.0*values[0]+6.0*values[1];
        let status;
        if objective<=best+EPS { status="pruned_bound"; }
        else if values.iter().all(|v|(v-v.round()).abs()<=EPS) {
            best=5.0*values[0].round()+6.0*values[1].round();
            status="integer";
        } else {
            status="branched";
            let i=if values[1].fract().abs()>EPS {1}else{0};
            let mut upper_lo=lo; upper_lo[i]=values[i].ceil();
            let mut lower_hi=hi; lower_hi[i]=values[i].floor();
            stack.push((upper_lo,hi,format!("x{} >= {}",i+1,upper_lo[i])));
            stack.push((lo,lower_hi,format!("x{} <= {}",i+1,lower_hi[i])));
        }
        rows.push(json!({"branch":branch,"status":status,"lp_value":objective,"x":values}));
    }
    check(rows.len()==3 && (best-27.0).abs()<EPS
          && rows[0]["status"]=="branched" && rows[1]["status"]=="integer"
          && rows[2]["status"]=="pruned_bound", "workshop must terminate after three nodes")?;
    Ok(json!({"nodes":rows,"node_count":rows.len(),"integer_optimum":best}))
}

#[derive(Serialize)]
struct BnbRecord { id:usize,instance_seed:u64,graph:Graph,oracle:Solve,search:SearchResult,error:f64 }

pub(super) fn run(a: &Args) -> Result<()> {
    fs::create_dir_all(&a.out)?;
    let start=Instant::now();
    let model=a.model.as_ref().map(nn::Network::load).transpose()?;
    let workshop=workshop_trace()?;
    println!("mode\tmetric\tvalue\truntime_ms");
    for (i,row) in workshop["nodes"].as_array().unwrap().iter().enumerate() {
        println!("bnb\tworkshop_node_{}\t{}\t0",i+1,row);
    }
    let mut records=Vec::new();
    for id in 0..a.count {
        let (instance_seed,graph)=generate(a.spins,a.edges,a.seed,id)?;
        // The oracle is outside the timed search and is NEVER supplied to it.
        let oracle=solve_spin(&graph,true)?;
        let result=search(&graph,&a.policy,model.as_ref(),a.export_labels)?;
        let error=(result.energy-oracle.energy).abs();
        println!("bnb\tinstance_{id:02}_nodes\t{}\t{:.6}",result.nodes,result.total_ms);
        println!("bnb\tinstance_{id:02}_lp_solves\t{}\t{:.6}",result.lp_solves,result.total_ms);
        records.push(BnbRecord{id,instance_seed,graph,oracle,search:result,error});
    }
    let max_error=records.iter().map(|r|r.error).fold(0.0,f64::max);
    let accounting_ok=records.iter().all(|r|r.search.node_lp_solves==r.search.nodes
        && r.search.lp_solves==r.search.nodes+r.search.trial_lp_solves
        && (a.policy=="strong" || r.search.trial_lp_solves==0));
    let valid=max_error<=EPS && accounting_ok;
    let summary=json!({"policy":a.policy,"instances":a.count,"max_energy_error":max_error,
        "mean_nodes":mean(records.iter().map(|r|r.search.nodes as f64),a.count),
        "mean_lp_solves":mean(records.iter().map(|r|r.search.lp_solves as f64),a.count),
        "mean_trial_lp_solves":mean(records.iter().map(|r|r.search.trial_lp_solves as f64),a.count),
        "mean_total_ms":mean(records.iter().map(|r|r.search.total_ms),a.count),
        "mean_lp_ms":mean(records.iter().map(|r|r.search.lp_ms),a.count),
        "lp_accounting_passed":accounting_ok,"workshop_nodes":3,
        "lp_solves_equal_nodes":records.iter().all(|r|r.search.lp_solves==r.search.nodes)});
    let elapsed=start.elapsed().as_secs_f64()*1e3;
    let run=json!({"schema":"branchopt.bnb.v1","mode":"bnb","parameters":a,
        "validation_status":if valid{"passed"}else{"failed_validation"},
        "settings":{"presolve":false,"threads":1,"integrality_tolerance":EPS,
            "bound_tolerance":EPS,"tie_tolerance":TIE,"initial_incumbent":"none",
            "trial_cache_reuse":false,"oracle_in_search_time":false,
            "lp_time_includes":"model setup, solve and extraction for visited and trial LPs"},
        "workshop":workshop,"summary":summary,"instances":records,"runtime_ms":elapsed});
    fs::write(a.out.join("run.json"),serde_json::to_vec_pretty(&run)?)?;
    for (name,value) in summary.as_object().unwrap() {
        println!("bnb\t{name}\t{value}\t{elapsed:.6}");
    }
    if a.export_labels {
        let mut dataset_nodes=Vec::new();
        for record in &records {
            for label in &record.search.labels {
                let mut v=serde_json::to_value(label)?;
                v["instance_id"]=json!(record.id);
                v["split"]=json!(if record.id<36 {"train"} else {"validation"});
                dataset_nodes.push(v);
            }
        }
        let graphs:Vec<Value>=records.iter().map(|r|json!({"id":r.id,"instance_seed":r.instance_seed,"graph":r.graph})).collect();
        let dataset=json!({"schema":"branchopt.labels.v1","parameters":a,
            "feature_names":nn::FEATURE_NAMES,"instances":graphs,"nodes":dataset_nodes,
            "max_nodes_per_instance":80,"minimum_candidates":2,
            "source_validation_status":if valid{"passed"}else{"failed"}});
        fs::write(a.out.join("dataset.json"),serde_json::to_vec_pretty(&dataset)?)?;
        eprintln!("Saved {}",a.out.join("dataset.json").display());
    }
    check(valid,"B&B validation failed; inspect the saved raw run.json")?;
    eprintln!("Saved {}",a.out.join("run.json").display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_search_agrees_with_exhaustive_enumeration() -> Result<()> {
        for (n,m) in [(3,3),(4,6),(6,12),(8,16)] {
            for id in 0..2 {
                let (_,g)=generate(n,m,8179,id)?;
                let expected=(0..(1usize<<n)).map(|mask| {
                    let bits:Vec<u8>=(0..n).map(|i|((mask>>i)&1)as u8).collect();
                    physical_energy(&g,&bits)
                }).fold(f64::INFINITY,f64::min);
                for policy in ["most-frac","strong"] {
                    let result=search(&g,policy,None,false)?;
                    assert!((result.energy-expected).abs()<1e-9);
                    assert_eq!(result.lp_solves,result.nodes+result.trial_lp_solves);
                    assert_eq!(result.node_lp_solves,result.nodes);
                    if policy=="most-frac" {assert_eq!(result.lp_solves,result.nodes);}
                }
            }
        }
        Ok(())
    }
    #[test]
    fn workshop_prunes_after_three_actual_lp_solves() -> Result<()> {
        let trace=workshop_trace()?;
        assert_eq!(trace["node_count"],json!(3));
        assert!((trace["nodes"][2]["lp_value"].as_f64().unwrap()-26.75).abs()<1e-9);
        Ok(())
    }
}

pub(super) fn evaluate(a:&Args) -> Result<()> {
    fs::create_dir_all(&a.out)?;
    let network=nn::Network::load(a.model.as_ref().ok_or_else(||fail("eval requires a model"))?)?;
    let parity=network.verify_probes()?;
    fs::write(a.out.join("parity.json"),serde_json::to_vec_pretty(&parity)?)?;
    check(parity.max_probability_error<1e-4,"JAX/Rust probability parity failed")?;
    let policies=["most-frac","strong","neural"];
    let start=Instant::now();
    let mut rows=Vec::new();
    let mut counts=vec![0.0;3]; let mut times=vec![0.0;3]; let mut solves=vec![0.0;3];
    let mut lp_times=vec![0.0;3]; let mut inference_times=vec![0.0;3];
    let mut max_error=0.0_f64; let mut accounting=true;
    for id in 0..a.count {
        let (instance_seed,graph)=generate(a.spins,a.edges,a.seed,id)?;
        let oracle=solve_spin(&graph,true)?;
        let mut searches=serde_json::Map::new(); let mut errors=serde_json::Map::new();
        for (p,&policy) in policies.iter().enumerate() {
            let result=search(&graph,policy,Some(&network),false)?;
            let error=(result.energy-oracle.energy).abs();
            max_error=max_error.max(error);
            accounting &= result.lp_solves==result.nodes+result.trial_lp_solves
                && result.node_lp_solves==result.nodes
                && (policy=="strong" || (result.trial_lp_solves==0 && result.lp_solves==result.nodes));
            counts[p]+=result.nodes as f64; times[p]+=result.total_ms; solves[p]+=result.lp_solves as f64;
            lp_times[p]+=result.lp_ms; inference_times[p]+=result.inference_ms;
            println!("eval\tinstance_{id:02}_{policy}_nodes\t{}\t{:.6}",result.nodes,result.total_ms);
            errors.insert(policy.into(),json!(error));
            searches.insert(policy.into(),serde_json::to_value(&result)?);
        }
        rows.push(json!({"id":id,"instance_seed":instance_seed,"graph":graph,"oracle":oracle,
            "searches":searches,"energy_errors":errors}));
    }
    let mut policy_summary=serde_json::Map::new();
    for (p,&policy) in policies.iter().enumerate() {
        policy_summary.insert(policy.into(),json!({"mean_nodes":counts[p]/a.count as f64,
            "mean_lp_solves":solves[p]/a.count as f64,"mean_total_ms":times[p]/a.count as f64,
            "mean_lp_ms":lp_times[p]/a.count as f64,"mean_inference_ms":inference_times[p]/a.count as f64,
            "lp_solves_equal_nodes":rows.iter().all(|r|r["searches"][policy]["lp_solves"]==r["searches"][policy]["nodes"])}));
    }
    let largest=rows.iter().max_by_key(|r|
        r["searches"]["most-frac"]["nodes"].as_i64().unwrap()
        - r["searches"]["neural"]["nodes"].as_i64().unwrap()).unwrap();
    let savings=json!({"instance_id":largest["id"],"ground_energy":largest["oracle"]["energy"],
        "most-frac":largest["searches"]["most-frac"],"neural":largest["searches"]["neural"]});
    let valid=max_error<=EPS && accounting;
    let summary=json!({"policies":policy_summary,"max_energy_error":max_error,
        "lp_accounting_passed":accounting,"neural_saves_nodes":counts[2]<counts[0],
        "probability_parity":parity,"largest_saving":savings});
    let run=json!({"schema":"branchopt.eval.v1","parameters":a,"mode":"eval",
        "validation_status":if valid{"passed"}else{"failed_exactness_or_accounting"},
        "summary":summary,"instances":rows,"runtime_ms":start.elapsed().as_secs_f64()*1e3});
    fs::write(a.out.join("run.json"),serde_json::to_vec_pretty(&run)?)?;
    println!("eval\tmax_energy_error\t{max_error}\t0");
    println!("eval\tlp_accounting_passed\t{accounting}\t0");
    println!("eval\tmax_probe_probability_error\t{}\t0",summary["probability_parity"]["max_probability_error"]);
    eprintln!("Saved {}",a.out.join("run.json").display());
    check(valid,"evaluation failed; inspect saved raw evidence")
}

#[cfg(test)]
mod learned_tests {
    use super::*;
    #[test]
    fn candidate_features_follow_the_twelve_definitions() {
        let g=Graph{n:3,h:vec![0.0;3],edges:vec![Edge{u:0,v:1,j:1.0},Edge{u:0,v:2,j:1.0},Edge{u:1,v:2,j:1.0}]};
        let lp=Lp{energy:-3.0,x:vec![0.5;3],z:vec![1.0;3],duals:vec![1.0;12]};
        let rows=feature_rows(&g,&lp,&[-1,-1,-1],f64::INFINITY,&[0]);
        let expected=[0.5,1.0,0.0,0.0,1.0,1.0,0.0,0.0,4.0,0.5,1.0,1.0];
        for (a,b) in rows[0].iter().zip(expected){assert!((a-b).abs()<1e-12);}
        let row=feature_rows(&g,&lp,&[-1,0,-1],-2.0,&[0]);
        assert!((row[0][10]-0.5).abs()<1e-12);
        assert!((row[0][11]-0.5).abs()<1e-12);
    }
    #[test]
    fn adversarial_network_changes_order_without_changing_optimum() -> Result<()> {
        let network=nn::adversarial_fixture();
        for id in 0..4 {
            let (_,g)=generate(6,12,17041,id)?;
            let expected=(0..64).map(|mask| {
                let bits:Vec<u8>=(0..6).map(|i|((mask>>i)&1)as u8).collect();
                physical_energy(&g,&bits)
            }).fold(f64::INFINITY,f64::min);
            let result=search(&g,"neural",Some(&network),false)?;
            assert!((result.energy-expected).abs()<1e-9);
            assert_eq!(result.lp_solves,result.nodes);
            assert_eq!(result.trial_lp_solves,0);
        }
        Ok(())
    }
}
WEEK12_PART3_FILE
cat > 'src/nn.rs' <<'WEEK12_PART3_FILE'
//! Pure f64 inference. This module has no solver dependency or LP calls.
use super::*;
use serde::Deserialize;

pub(super) const FEATURE_NAMES: [&str;12] = ["x_val","frac_dist","field_h","abs_field_h",
    "deg_norm","bond_bias","eff_field","abs_eff_field","mean_bond_dual","mean_bond_gap",
    "mean_nbr_frac","gap_ratio"];

#[derive(Clone,Deserialize)]
struct Stats { mean:Vec<f64>, std:Vec<f64> }
#[derive(Clone,Deserialize)]
struct Weights { w1:Vec<Vec<f64>>, b1:Vec<f64>, w2:Vec<f64>, b2:f64 }
#[derive(Clone,Deserialize)]
struct Probe { features:Vec<Vec<f64>>, logits:Vec<f64>, probabilities:Vec<f64> }
#[derive(Clone,Deserialize)]
pub(super) struct Network {
    schema:String, feature_names:Vec<String>, stats:Stats, weights:Weights, probes:Vec<Probe>,
}
#[derive(Serialize)]
pub(super) struct Parity {
    pub probe_count:usize, pub max_probability_error:f64, pub max_logit_error:f64,
}

impl Network {
    pub fn load(path:&PathBuf) -> Result<Self> {
        let model:Self=serde_json::from_slice(&fs::read(path)?)?;
        model.validate()?;
        Ok(model)
    }
    fn validate(&self) -> Result<()> {
        check(self.schema=="branchopt.mlp.v1","unsupported model schema")?;
        check(self.feature_names.iter().map(String::as_str).eq(FEATURE_NAMES.iter().copied()),
              "model feature order does not match Rust")?;
        check(self.stats.mean.len()==12 && self.stats.std.len()==12,"invalid feature statistics")?;
        let w=&self.weights;
        check(w.w1.len()==12 && w.w1.iter().all(|row|row.len()==48)
              && w.b1.len()==48 && w.w2.len()==48,"network must have architecture 12-48-1")?;
        check(self.stats.mean.iter().all(|v|v.is_finite())
              && self.stats.std.iter().all(|v|v.is_finite() && *v>=0.0)
              && w.w1.iter().flatten().chain(w.b1.iter()).chain(w.w2.iter()).all(|v|v.is_finite())
              && w.b2.is_finite(),"non-finite model coefficient")
    }
    pub fn scores(&self,features:&[Vec<f64>]) -> Result<Vec<f64>> {
        let mut result=Vec::with_capacity(features.len());
        for row in features {
            check(row.len()==12 && row.iter().all(|v|v.is_finite()),"invalid neural feature row")?;
            let mut input=[0.0;12];
            for f in 0..12 {input[f]=(row[f]-self.stats.mean[f])/self.stats.std[f].max(1e-6);}
            let mut score=self.weights.b2;
            for h in 0..48 {
                let mut value=self.weights.b1[h];
                for f in 0..12 {value+=input[f]*self.weights.w1[f][h];}
                score+=self.weights.w2[h]*value.tanh();
            }
            check(score.is_finite(),"non-finite neural score")?;
            result.push(score);
        }
        Ok(result)
    }
    pub fn probabilities(&self,features:&[Vec<f64>]) -> Result<Vec<f64>> {
        let scores=self.scores(features)?;
        check(!scores.is_empty(),"empty candidate softmax")?;
        let top=scores.iter().copied().fold(f64::NEG_INFINITY,f64::max);
        let mut p:Vec<f64>=scores.iter().map(|s|(s-top).exp()).collect();
        let sum=p.iter().sum::<f64>();
        for v in &mut p {*v/=sum;}
        Ok(p)
    }
    pub fn verify_probes(&self) -> Result<Parity> {
        check(self.probes.len()==6,"model must contain exactly six validation probes")?;
        let (mut prob_error,mut logit_error)=(0.0_f64,0.0_f64);
        for probe in &self.probes {
            let logits=self.scores(&probe.features)?;
            let probs=self.probabilities(&probe.features)?;
            check(probe.features.len()>=2 && probe.logits.len()==logits.len()
                  && probe.probabilities.len()==probs.len(),"invalid probe dimensions")?;
            check(probe.logits.iter().chain(probe.probabilities.iter()).all(|v|v.is_finite()),
                  "non-finite JAX probe")?;
            for (a,b) in logits.iter().zip(&probe.logits) {logit_error=logit_error.max((a-b).abs());}
            for (a,b) in probs.iter().zip(&probe.probabilities) {prob_error=prob_error.max((a-b).abs());}
        }
        Ok(Parity{probe_count:self.probes.len(),max_probability_error:prob_error,max_logit_error:logit_error})
    }
}

#[cfg(test)]
pub(super) fn adversarial_fixture() -> Network {
    let mut w1=vec![vec![0.0;48];12];
    w1[2][0]=100.0; w1[4][1]=-50.0;
    let mut w2=vec![0.0;48]; w2[0]=1.0; w2[1]=1.0;
    Network{schema:"branchopt.mlp.v1".into(),feature_names:FEATURE_NAMES.iter().map(|s|s.to_string()).collect(),
        stats:Stats{mean:vec![0.0;12],std:vec![1.0;12]},
        weights:Weights{w1,b1:vec![0.0;48],w2,b2:0.0},probes:Vec::new()}
}
WEEK12_PART3_FILE
cat > 'train_branch.py' <<'WEEK12_PART3_FILE'
#!/usr/bin/env python3
"""JAX x64 imitation learning; the validation/test graphs never enter training."""
import os
os.environ["JAX_PLATFORMS"]="cpu"
import argparse
from collections import Counter
import json
import hashlib
import math
from pathlib import Path
import platform
import time
import jax
jax.config.update("jax_enable_x64",True)
import jax.numpy as jnp
import numpy as np

FEATURES=["x_val","frac_dist","field_h","abs_field_h","deg_norm","bond_bias",
          "eff_field","abs_eff_field","mean_bond_dual","mean_bond_gap","mean_nbr_frac","gap_ratio"]

def independent_features(graph,node):
    """Independent Python recomputation from the recorded parent LP state."""
    x=np.clip(np.asarray(node["lp_x"],dtype=np.float64),0,1)
    s=1-2*x
    z=np.asarray(node["lp_z"],dtype=np.float64)
    dual=np.asarray(node["row_duals"],dtype=np.float64)
    fixed=node["fixed"]
    edges=graph["edges"]
    assert len(dual)==4*len(edges)
    incident=[[] for _ in range(graph["n"])]
    for k,e in enumerate(edges):
        incident[e["u"]].append((k,e["v"]))
        incident[e["v"]].append((k,e["u"]))
    dmax=max(1,max(map(len,incident)))
    ddual=max(1e-9,float(np.max(np.abs(dual),initial=0)))
    best=node["incumbent"]
    ratio=1.0 if best is None else min(2,max(0,(best-node["lp_bound"])/max(abs(best),1)))
    result=[]
    for i in node["candidates"]:
        degree=len(incident[i]); denominator=max(1,degree)
        bias=0.0; effective=float(graph["h"][i]); dual_sum=0.0; gap=0.0; frac=0.0
        for k,j in incident[i]:
            coupling=edges[k]["j"]
            bias+=float(coupling>0)-float(coupling<0)
            effective+=coupling*s[j]
            dual_sum+=sum(abs(float(v)) for v in dual[4*k:4*k+4])/ddual
            gap+=abs(z[k]-0.5*(1-s[i]*s[j]))
            if fixed[j]<0:
                frac+=2*min(x[j],1-x[j])
        effective/=degree+1
        result.append([x[i],2*min(x[i],1-x[i]),graph["h"][i],abs(graph["h"][i]),
            degree/dmax,bias/denominator,effective,abs(effective),dual_sum/denominator,
            gap/denominator,frac/denominator,ratio])
    return np.asarray(result,dtype=np.float64)

def checked_dataset(path):
    data=json.loads(Path(path).read_text())
    assert data["schema"]=="branchopt.labels.v1"
    assert data["feature_names"]==FEATURES
    assert data["source_validation_status"]=="passed"
    graphs={r["id"]:r["graph"] for r in data["instances"]}
    assert len(graphs)==len(data["instances"])
    counts=Counter(); seen=set(); maximum=0.0
    train=[]; validation=[]
    for node in data["nodes"]:
        instance=node["instance_id"]
        assert 0<=instance<48 and instance in graphs
        assert node["split"]==("train" if instance<36 else "validation")
        key=(instance,node["node"])
        assert key not in seen
        seen.add(key); counts[instance]+=1
        assert 2<=len(node["candidates"])<=graphs[instance]["n"]
        assert len(set(node["candidates"]))==len(node["candidates"])
        assert node["chosen_spin"] in node["candidates"]
        actual=np.asarray(node["features"],dtype=np.float64)
        expected=independent_features(graphs[instance],node)
        assert actual.shape==(len(node["candidates"]),12) and np.isfinite(actual).all()
        np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-10)
        maximum=max(maximum,float(np.max(np.abs(actual-expected))))
        gains=np.asarray(node["gains"],dtype=np.float64)
        scores=np.asarray(node["scores"],dtype=np.float64)
        assert gains.shape==(len(actual),2) and scores.shape==(len(actual),)
        np.testing.assert_allclose(scores,np.maximum(gains[:,0],1e-6)*np.maximum(gains[:,1],1e-6),rtol=1e-12,atol=1e-12)
        (train if instance<36 else validation).append(node)
    assert train and len(validation)>=6,"Need training nodes and at least six validation nodes"
    assert max(counts.values())<=80
    train_ids=sorted({n["instance_id"] for n in train})
    val_ids=sorted({n["instance_id"] for n in validation})
    assert not set(train_ids)&set(val_ids)
    provenance={"parameters":data["parameters"],
        "graph_hashes":[hashlib.sha256(json.dumps(g,sort_keys=True,separators=(",",":")).encode()).hexdigest() for g in graphs.values()]}
    return train,validation,train_ids,val_ids,maximum,provenance

def padded_nodes(nodes,mean,std,width):
    x=np.zeros((len(nodes),width,12),dtype=np.float64)
    mask=np.zeros((len(nodes),width),dtype=bool)
    target=np.zeros((len(nodes),width),dtype=np.float64)
    for i,node in enumerate(nodes):
        raw=np.asarray(node["features"],dtype=np.float64)
        k=len(raw)
        x[i,:k]=(raw-mean)/np.maximum(std,1e-6)
        mask[i,:k]=True
        score=np.asarray(node["scores"],dtype=np.float64)
        best=float(score.max())
        tied=np.abs(score-best)<=1e-9*max(1,abs(best))
        target[i,:k]=tied/tied.sum()
    return jnp.asarray(x),jnp.asarray(mask),jnp.asarray(target)

def logits(params,x):
    return jnp.tanh(x@params["w1"]+params["b1"])@params["w2"]+params["b2"]

def cross_entropy(params,x,mask,target):
    # A finite mask value avoids 0*(-infinity) in padded targets and gradients.
    masked=jnp.where(mask,logits(params,x),-1e30)
    log_probability=jax.nn.log_softmax(masked,axis=-1)
    return -jnp.mean(jnp.sum(target*log_probability,axis=-1))

def objective(params,x,mask,target):
    return cross_entropy(params,x,mask,target)+2e-4*(jnp.sum(params["w1"]**2)+jnp.sum(params["w2"]**2))

@jax.jit
def train_step(params,m,v,count,x,mask,target):
    value,grad=jax.value_and_grad(objective)(params,x,mask,target)
    m=jax.tree_util.tree_map(lambda a,g:0.9*a+0.1*g,m,grad)
    v=jax.tree_util.tree_map(lambda a,g:0.999*a+0.001*g*g,v,grad)
    params=jax.tree_util.tree_map(lambda p,a,b:p-4e-3*(a/(1-0.9**count))/(jnp.sqrt(b/(1-0.999**count))+1e-8),params,m,v)
    return params,m,v,value

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",default="artifacts/bnb_strong/dataset.json")
    parser.add_argument("--out",default="artifacts/model.json")
    parser.add_argument("--epochs",type=int,default=120)
    args=parser.parse_args()
    assert args.epochs>0
    train,val,train_ids,val_ids,feature_error,provenance=checked_dataset(args.dataset)
    raw_train=np.concatenate([np.asarray(n["features"],dtype=np.float64) for n in train])
    mean=raw_train.mean(axis=0); std=raw_train.std(axis=0)
    width=max(len(n["candidates"]) for n in train+val)
    train_batch=padded_nodes(train,mean,std,width)
    val_batch=padded_nodes(val,mean,std,width)
    key1,key2=jax.random.split(jax.random.PRNGKey(2026))
    params={"w1":jax.random.normal(key1,(12,48),dtype=jnp.float64)*math.sqrt(2/60),
        "b1":jnp.zeros(48,dtype=jnp.float64),
        "w2":jax.random.normal(key2,(48,),dtype=jnp.float64)*math.sqrt(2/49),
        "b2":jnp.asarray(0.0,dtype=jnp.float64)}
    assert all(a.dtype==jnp.float64 for a in jax.tree_util.tree_leaves(params))
    assert jax.default_backend()=="cpu" and jax.config.x64_enabled
    m=jax.tree_util.tree_map(jnp.zeros_like,params)
    v=jax.tree_util.tree_map(jnp.zeros_like,params)
    ce=jax.jit(cross_entropy)
    rng=np.random.default_rng(2026)
    history=[]; best_params=None; best_ce=math.inf; best_epoch=None; step=0
    started=time.perf_counter()
    print(f"JAX {jax.__version__}; CPU float64; train nodes={len(train)}, val nodes={len(val)}",flush=True)
    print(f"Train instances={len(train_ids)}, val instances={len(val_ids)}, overlap=0; feature error={feature_error:.3e}",flush=True)
    for epoch in range(args.epochs+1):
        if epoch:
            permutation=rng.permutation(len(train))
            for start in range(0,len(train),64):
                indices=permutation[start:start+64]
                batch=tuple(a[indices] for a in train_batch)
                step+=1
                params,m,v,_=train_step(params,m,v,jnp.asarray(step,dtype=jnp.int32),*batch)
        train_ce=float(ce(params,*train_batch)); val_ce=float(ce(params,*val_batch))
        assert math.isfinite(train_ce) and math.isfinite(val_ce),"Non-finite training loss"
        history.append({"epoch":epoch,"train_ce":train_ce,"validation_ce":val_ce})
        if epoch==1 or (epoch>0 and epoch%5==0):
            if val_ce<best_ce:
                best_ce=val_ce; best_epoch=epoch
                best_params=jax.tree_util.tree_map(lambda a:np.asarray(a).copy(),params)
        if epoch in [0,1] or epoch%10==0:
            print(f"epoch {epoch:3d}: train CE={train_ce:.6f}; validation CE={val_ce:.6f}",flush=True)
    assert best_params is not None
    selected={k:jnp.asarray(v,dtype=jnp.float64) for k,v in best_params.items()}
    probes=[]
    for index in np.linspace(0,len(val)-1,6,dtype=int):
        node=val[int(index)]
        raw=np.asarray(node["features"],dtype=np.float64)
        scores=logits(selected,jnp.asarray((raw-mean)/np.maximum(std,1e-6)))
        probabilities=jax.nn.softmax(scores)
        probes.append({"instance_id":node["instance_id"],"node":node["node"],
            "candidates":node["candidates"],"features":raw.tolist(),
            "logits":np.asarray(scores).tolist(),"probabilities":np.asarray(probabilities).tolist()})
    masked=jnp.where(train_batch[1],logits(selected,train_batch[0]),-1e30)
    p=np.asarray(jax.nn.softmax(masked,axis=-1))
    pad_probability=float(np.max(p[~np.asarray(train_batch[1])],initial=0))
    assert pad_probability==0.0
    summary={"train_nodes":len(train),"validation_nodes":len(val),
        "train_instances":len(train_ids),"validation_instances":len(val_ids),
        "train_instance_ids":train_ids,"validation_instance_ids":val_ids,"instance_overlap":[],
        "best_epoch":best_epoch,"best_validation_ce":best_ce,"epochs":args.epochs,
        "batch_size":64,"learning_rate":4e-3,"weight_l2":2e-4,"biases_regularized":False,
        "seed":2026,"feature_check_max_abs_error":feature_error,
        "maximum_padded_probability":pad_probability,"training_wall_seconds":time.perf_counter()-started,
        "initialization":"Xavier-normal weights; zero biases; JAX PRNGKey(2026)",
        "normalization":"population mean/std over real candidate rows from training instances only",
        "checkpoint_rule":"lowest validation CE among epoch 1 and every fifth epoch"}
    output={"schema":"branchopt.mlp.v1","feature_names":FEATURES,
        "stats":{"mean":mean.tolist(),"std":std.tolist()},
        "weights":{k:np.asarray(v).tolist() for k,v in best_params.items()},
        "history":history,"training_summary":summary,"probes":probes,"data_provenance":provenance,
        "environment":{"jax":jax.__version__,"numpy":np.__version__,"platform":platform.platform(),
            "backend":jax.default_backend(),"x64":jax.config.x64_enabled}}
    path=Path(args.out); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,indent=2,allow_nan=False))
    print("Saved",path.resolve(),flush=True)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    main()
WEEK12_PART3_FILE
cat > 'train_branch.design.toml' <<'WEEK12_PART3_FILE'
usage = "uv run --no-project --python 3.12 --with jax==0.12.1 python train_branch.py --dataset artifacts/bnb_strong/dataset.json --out artifacts/model.json"
description = "Imitate Strong Branching with a shared candidate-scoring MLP"

[args]
dataset = "JSON exported by branchopt --mode bnb --policy strong --export-labels"
out = "JSON model, train-only statistics, complete CE history, and six validation probes"
epochs = "120 for the course experiment; smaller values are for development only"

[runtime]
python = "3.12, provisioned with uv"
jax = "0.12.1"
platform = "CPU"
precision = "float64, enabled before constructing JAX arrays"

[data]
training_instance_ids = [0, 35]
validation_instance_ids = [36, 47]
split_ranges_are_inclusive = true
minimum_candidates = 2
maximum_nodes_per_instance = 80
normalization = "Population mean/std over real candidate rows from training nodes only; denominator max(std, 1e-6). Padding is zero in standardized coordinates."
labels = "Uniform probability on scores within 1e-9 * max(1, abs(best_score)) of the best Strong Branching score."
feature_audit = "Independently recompute all twelve features in Python from the recorded parent LP state before training."

[network]
dimensions = [12, 48, 1]
activation = "tanh"
aggregation = "Masked softmax over the real candidates at each node"
initialization = "Xavier-normal with JAX PRNGKey(2026); zero biases"
w1_layout = "12 input rows by 48 hidden columns; Rust uses w1[input][hidden]"

[optimization]
algorithm = "Adam"
learning_rate = 0.004
batch_size = 64
epochs = 120
seed = 2026
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
weight_l2 = 0.0002
regularize_biases = false
objective = "Mean node cross-entropy + 0.0002 * (sum(w1^2) + sum(w2^2))"
shuffle = "numpy default_rng(2026), a new permutation each epoch"
checkpoint = "Lowest validation cross-entropy among epoch 1 and every fifth epoch; no test-driven selection"
history = "Training and validation cross-entropy without L2 after every epoch, including an initial epoch 0"

[validation]
probe_count = 6
maximum_probability_error = 0.0001
search_energy_tolerance = 0.000001
test = "16 unseen graphs with n=22, m=52, seed=3026"
policies = ["most-frac", "strong", "neural"]
require_neural_nodes_below_most_frac = true
neural_lp_solves = "Equal explored nodes; no Strong trial solves"

[writes]
"artifacts/model.json" = "Best model, feature names, train statistics, history, probes, provenance and software environment"
"artifacts/eval_test/run.json" = "Actual exact-search evidence on unseen graphs"
"artifacts/eval_test/part3_checks.json" = "Explicit pass/fail criteria, measured summaries and largest node saving"
"artifacts/neural_bnb.png" = "Cross-entropy history and independently sorted explored-node curves"
WEEK12_PART3_FILE
cat > 'plot_neural_bnb.py' <<'WEEK12_PART3_FILE'
#!/usr/bin/env python3
"""Plot measured training/evaluation evidence and report all Part 3 criteria."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument("--model",default="artifacts/model.json")
parser.add_argument("--run",default="artifacts/eval_test/run.json")
parser.add_argument("--out",default="artifacts/neural_bnb.png")
parser.add_argument("--require-course-settings",action="store_true")
args=parser.parse_args()
model=json.loads(Path(args.model).read_text())
run=json.loads(Path(args.run).read_text())
training=model["training_summary"]; summary=run["summary"]
history=model["history"]; records=run["instances"]
train_ids=set(training["train_instance_ids"]); val_ids=set(training["validation_instance_ids"])
source_hashes=set(model["data_provenance"]["graph_hashes"])
test_hashes={hashlib.sha256(json.dumps(r["graph"],sort_keys=True,separators=(",",":")).encode()).hexdigest() for r in records}
allowed=[h for h in history if h["epoch"]==1 or (h["epoch"]>0 and h["epoch"]%5==0)]
checks={
    "exactness_and_lp_accounting":run["validation_status"]=="passed" and summary["max_energy_error"]<=1e-6 and summary["lp_accounting_passed"],
    "train_validation_instance_overlap_zero":not bool(train_ids&val_ids) and not training["instance_overlap"],
    "instance_split_correct":all(0<=i<36 for i in train_ids) and all(36<=i<48 for i in val_ids),
    "unseen_test_graphs":not bool(source_hashes&test_hashes),
    "cpu_float64":model["environment"]["backend"]=="cpu" and model["environment"]["x64"] is True,
    "six_probability_probes":summary["probability_parity"]["probe_count"]==6 and summary["probability_parity"]["max_probability_error"]<1e-4,
    "independent_feature_recomputation":training["feature_check_max_abs_error"]<1e-10,
    "padded_candidates_have_zero_probability":training["maximum_padded_probability"]==0,
    "checkpoint_selected_by_validation":abs(training["best_validation_ce"]-min(h["validation_ce"] for h in allowed))<1e-10,
    "training_ce_reduced":history[-1]["train_ce"]<history[0]["train_ce"],
    "neural_uses_no_trial_lps":all(r["searches"]["neural"]["trial_lp_solves"]==0 and r["searches"]["neural"]["lp_solves"]==r["searches"]["neural"]["nodes"] for r in records),
    "most_frac_lp_solves_equal_nodes":all(r["searches"]["most-frac"]["lp_solves"]==r["searches"]["most-frac"]["nodes"] for r in records),
    "neural_saves_nodes":summary["neural_saves_nodes"],
}
if args.require_course_settings:
    source=model["data_provenance"]["parameters"]; evaluation=run["parameters"]
    checks["course_settings"]=(source["spins"],source["edges"],source["count"],source["seed"])==(22,52,48,2026) and \
        (evaluation["spins"],evaluation["edges"],evaluation["count"],evaluation["seed"])==(22,52,16,3026) and \
        training["epochs"]==120 and training["batch_size"]==64 and training["seed"]==2026 and \
        training["learning_rate"]==4e-3 and training["weight_l2"]==2e-4 and \
        not training["biases_regularized"] and model["environment"]["jax"]=="0.12.1"

plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout="constrained")
axes[0].plot([h["epoch"] for h in history],[h["train_ce"] for h in history],label="Training",color="#3765ad")
axes[0].plot([h["epoch"] for h in history],[h["validation_ce"] for h in history],label="Validation",color="#d0762b")
axes[0].scatter([training["best_epoch"]],[training["best_validation_ce"]],marker="*",s=120,color="#923b23",zorder=4,label="Saved checkpoint")
axes[0].set(xlabel="Epoch",ylabel="Cross-entropy (without L2)",title="Imitating Strong Branching")
for policy,color,label in [("most-frac","#3765ad","Most fractional"),("neural","#329568","Neural"),("strong","#d0762b","Strong branching")]:
    axes[1].plot(range(1,len(records)+1),sorted(r["searches"][policy]["nodes"] for r in records),label=label,color=color,lw=2)
axes[1].set(xlabel="Independent sorted rank",ylabel="Explored nodes",title="Exact search on unseen graphs")
for ax in axes:
    ax.grid(axis="y",alpha=0.2); ax.legend(frameon=False)
p=run["parameters"]
fig.suptitle(f"Learned branching: train/validation split by instance; test n={p['spins']}, m={p['edges']}, seed={p['seed']}")
out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out,dpi=180)

largest=summary["largest_saving"]
largest_brief={"instance_id":largest["instance_id"],"ground_energy":largest["ground_energy"]}
for policy in ["most-frac","neural"]:
    largest_brief[policy]={k:largest[policy][k] for k in ["nodes","total_ms","first_splits"]}
report={"checks":checks,"all_passed":all(checks.values()),"training_summary":training,
    "probability_parity":summary["probability_parity"],"max_energy_error":summary["max_energy_error"],
    "policies":summary["policies"],"largest_saving":largest_brief}
Path(args.run).parent.joinpath("part3_checks.json").write_text(json.dumps(report,indent=2))
print("Saved",out.resolve())
print(f"Train: {training['train_nodes']} nodes / {len(train_ids)} instances; validation: {training['validation_nodes']} nodes / {len(val_ids)} instances; overlap: {len(train_ids&val_ids)}")
print(f"Best checkpoint: epoch {training['best_epoch']}, validation CE={training['best_validation_ce']:.6f}")
print("Max feature error:",training["feature_check_max_abs_error"])
print("Max JAX/Rust probability error:",summary["probability_parity"]["max_probability_error"])
print("Max energy error vs HiGHS:",summary["max_energy_error"])
print("\nPolicy       Mean nodes   Mean LP solves   Mean time (ms)")
for policy in ["most-frac","neural","strong"]:
    s=summary["policies"][policy]
    print(f"{policy:<12} {s['mean_nodes']:10.3f} {s['mean_lp_solves']:16.3f} {s['mean_total_ms']:16.3f}")
print("\nLargest node saving:",json.dumps(largest_brief))
print("\nChecks:",json.dumps(checks,indent=2))
if not all(checks.values()):
    raise SystemExit("Part 3 criteria not met: "+", ".join(k for k,v in checks.items() if not v))
print("\nPART 3 VALIDATION PASSED")
WEEK12_PART3_FILE
python3 - <<'PYDOCS'
from pathlib import Path
import re
p=Path('branchopt.design.toml')
t=p.read_text() if p.exists() else ''
t=re.sub(re.escape('# WEEK12_PART3_BEGIN')+r'.*?'+re.escape('# WEEK12_PART3_END')+r'\n?', '', t, flags=re.S)
p.write_text(t.rstrip()+'\n\n'+'# WEEK12_PART3_BEGIN\n[args.labels]\nexport_labels = "Optional --export-labels flag; default false; only permitted for bnb + strong. Record at least two candidates and at most 80 branch nodes per instance."\n\n[args.eval]\nmodel = "--model path/to/model.json; required for neural branching and eval"\n\n[benchmark.bnb.features]\nsource = "Parent graph, parent LP x/z/row-duals, box bounds, current feasible incumbent. Extract before Strong trial solves."\nconventions = "x is clamped to [0,1]; s=1-2x; dmax=max(1,max degree); ddual=max(1e-9,max absolute row dual); an empty incident-edge mean is zero."\nordered = [\n  "x_val = x_i",\n  "frac_dist = 2*min(x_i,1-x_i)",\n  "field_h = h_i",\n  "abs_field_h = abs(h_i)",\n  "deg_norm = degree(i)/dmax",\n  "bond_bias = sum_incident sign(J_e)/max(1,degree(i))",\n  "eff_field = (h_i + sum_incident J_ij*s_j)/(degree(i)+1)",\n  "abs_eff_field = abs(eff_field)",\n  "mean_bond_dual = mean_incident(sum_four_rows abs(dual_e))/ddual",\n  "mean_bond_gap = mean_incident abs(z_e - 0.5*(1-s_i*s_j))",\n  "mean_nbr_frac = mean_neighbors(0 if fixed else 2*min(x_j,1-x_j))",\n  "gap_ratio = clamp((incumbent-parent_bound)/max(abs(incumbent),1),0,2), or 1 without incumbent",\n]\nneural = "Normalize with exported train-only mean/std, apply shared 12-48(tanh)-1 f64 MLP to candidates, choose largest score and map back to the spin index. No trial LPs; all exact search bounds and pruning are unchanged."\n\n[benchmark.eval]\ntest = "16 unseen graphs: n=22, m=52, seed=3026"\npolicies = ["most-frac", "strong", "neural"]\nevidence = "Compare every exact energy with a separate HiGHS MILP oracle. Record actual nodes, LP solves, trial solves, total/LP/feature-plus-inference time, full traces and first two split indices. Oracle and model probe checks are outside search timers."\nchecks = "Energy error <=1e-6; most-frac/neural LP solves equal explored nodes; six JAX/Rust probability errors <1e-4; neural uses fewer mean nodes than most-frac. Save actual evidence even if the learned policy fails to reduce nodes."\n\n[writes.labels]\n"<out>/dataset.json" = "Graph IDs and seeds; candidates, features, gains, scores, pruned-child counts and original parent LP state for every exported node"\n\n[writes.eval]\n"<out>/run.json" = "Measured three-policy results on identical unseen graphs and all validation summaries"\n"<out>/parity.json" = "Six-probe maximum probability and logit differences between JAX and Rust"\n# WEEK12_PART3_END\n')
p=Path('README.md')
t=p.read_text() if p.exists() else ''
t=re.sub(re.escape('<!-- WEEK12_PART3_BEGIN -->')+r'.*?'+re.escape('<!-- WEEK12_PART3_END -->')+r'\n?', '', t, flags=re.S)
p.write_text(t.rstrip()+'\n\n'+"<!-- WEEK12_PART3_BEGIN -->\n## Part 3: Imitation learning for branching\n\nRun `bash ./week12_part3.sh` from the existing `week12/` directory after Part 2.\nThe saved `run_part3.sh` is the same self-contained reproduction script.\nThe script runs the four small Rust checks, installs the updated executable,\nexports Strong labels, trains JAX on the CPU, runs the unseen evaluation, plots\nthe measured evidence, and makes one local commit after all criteria pass.\n\nStrong labels use 48 graphs with n=22, m=52, seed=2026. The optional\n`--export-labels` flag defaults to false. Only branch nodes with at least two\ncandidates are recorded, with a cap of 80 records per graph; the search still\nfinishes normally. Instance IDs 0..35 are training graphs and 36..47 are\nvalidation graphs. The labels contain the original parent x/z, four row-duals\nper bond, fixed-spin box, incumbent, candidate indices, features, gains and scores.\nFeatures are extracted before any Strong child trials and never use the oracle.\nPython independently recomputes the twelve features before training.\nThe complete feature order and formulas are frozen in branchopt.design.toml.\n\nLabel collection writes to `artifacts/bnb_labels/`; its dataset is also copied\nto the course's canonical `artifacts/bnb_strong/dataset.json`. This keeps the\nexisting Part 2 run.json, timings and bnb.png available for comparison. The\nscript verifies that the exported graph IDs, graph data, node counts and LP\nsolve counts agree with the earlier Strong run before starting training.\n\nThe shared 12->48(tanh)->1 MLP uses JAX 0.12.1, CPU float64. Population mean and\nstd are fitted to real candidate rows from training graphs only; the std\ndenominator is floored at 1e-6. Padded rows are finite zeros in standardized\ncoordinates and have zero target and masked probability. Near-best Strong\nscores share a uniform target using tolerance 1e-9*max(1,abs(best)). Xavier-normal\nweights and zero biases are initialized with JAX PRNGKey(2026). Adam uses\nlearning rate 0.004, batch size 64, 120 epochs, and 0.0002 times the sum of\nsquared weights, excluding biases. A seeded NumPy generator shuffles training\nnodes each epoch. CE histories exclude regularization. The saved checkpoint\nhas the lowest validation CE among epoch 1 and every fifth epoch. Test graphs\nnever select a checkpoint or influence the training statistics.\n\n`artifacts/model.json` contains the best weights, feature statistics, complete\ntraining/validation CE history, six validation probes, graph provenance and\nsoftware environment. Rust uses the same w1[input][hidden] matrix layout in\nf64, verifies every probe probability to <1e-4, and also records logit error.\nInvalid or non-finite features/model outputs stop the run with an error.\n\nNeural branching scores only the current candidate spins. It changes which\nspin is split; it does not supply bounds, prune nodes, seed an incumbent, or\nsolve trial LPs. Splits still partition the box into x_i=0 and x_i=1. The\nexisting LP lower-bound pruning and feasible integer incumbents preserve the\nexact search, independently of model quality. Every test energy is checked\nagainst a separate HiGHS MILP optimum. Most-frac and neural must each perform\nexactly one LP solve per explored node.\n\nTraining and validation states come from the Strong policy. Neural search\ncan encounter different states and incumbents, so validation CE alone does not\nguarantee improved search. A separate test on 16 unseen graphs (n=22, m=52,\nseed=3026) compares most-frac, strong and neural. This test measures model\nperformance without changing the selected checkpoint. Actual values are saved\neven when a performance criterion fails; no reference values replace results.\n\nReproduce individual steps from `week12/`:\n\n```bash\nbranchopt --mode bnb --spins 22 --edges 52 --count 48 --seed 2026 --policy strong --export-labels --out artifacts/bnb_labels\ncp artifacts/bnb_labels/dataset.json artifacts/bnb_strong/dataset.json\nuv run --no-project --python 3.12 --with jax==0.12.1 python train_branch.py --dataset artifacts/bnb_strong/dataset.json --out artifacts/model.json\nbranchopt --mode eval --spins 22 --edges 52 --count 16 --seed 3026 --model artifacts/model.json --out artifacts/eval_test\nuv run --no-project --python 3.12 --with matplotlib==3.10.7 python plot_neural_bnb.py --require-course-settings\n```\n\nThe figure `artifacts/neural_bnb.png` shows train/validation CE and independently\nsorted node-count curves. The raw test run stores each graph, oracle, exact\nsearch trace, node/LP counts, timings and first splits. The printed summary and\n`artifacts/eval_test/part3_checks.json` report actual dataset sizes, zero overlap,\nbest validation CE, probe error, energy error, policy means and the instance\nwith the largest node saving. Total search time includes trace overhead; LP\ntime includes model construction, solve and extraction; feature-plus-inference\ntime measures feature construction and the neural forward pass. MILP oracles,\nmodel loading, probe checks and JSON file writing are outside search timers.\n\nPre-delivery checks used small development graphs: eight mixed-field graphs\nchecked against exhaustive enumeration, the workshop's three actual nodes,\na hand-computed feature example, and adversarial-network searches checked\nagainst exhaustive enumeration. An additional n=6 development pipeline checked\nlabel export, all 120 training epochs, JSON model export, six-probe Rust parity,\nexact unseen searches and plotting. These are development checks; the course's\nn=22 experiments are executed on the user's machine by the reproduction script.\nThe independent static review and its dispositions are in DEEPSEEK_REVIEW.md.\n<!-- WEEK12_PART3_END -->\n")
p=Path('DEEPSEEK_REVIEW.md')
t=p.read_text() if p.exists() else ''
t=re.sub(re.escape('<!-- WEEK12_PART3_BEGIN -->')+r'.*?'+re.escape('<!-- WEEK12_PART3_END -->')+r'\n?', '', t, flags=re.S)
p.write_text(t.rstrip()+'\n\n'+"<!-- WEEK12_PART3_BEGIN -->\n## Part 3: DeepSeek review of learning and inference excerpts\n\nOn 2026-11-27 ChatGPT sent the displayed feature formulas, label-export contract,\nJAX network/loss/Adam code, checkpoint rule, model/probe schema and Rust inference\ncontract to DeepSeek in the same review conversation:\nhttps://chat.deepseek.com/a/chat/s/fc47b89b-c851-4ed2-a26b-c7409e906862\n\nScope: independent static review of the supplied excerpts and interface\ncontracts. DeepSeek did not run the programs or review the entire source tree.\nIts response reported no definite error in those excerpts and identified five\nrisks. The following are paraphrases and checked dispositions, not test claims\nmade by DeepSeek:\n\n1. **Bond-gap meaning.** The expression involving 0.5*(1-s_i*s_j) is not an LP\n   constraint-violation measure at fractional x. We retain it because it is the\n   feature explicitly defined by the learning sheet; the formula is documented\n   exactly and independently recomputed in Python.\n2. **Dual scale and bond bias.** The excerpt omitted the dual-scale assignment.\n   Full Rust code uses max(1e-9,max|row_dual|), and bond bias counts the signs of\n   incident J, as specified. The feature check and hand-computed unit example\n   cover these definitions.\n3. **Policy-state distribution shift.** Training/validation labels come from\n   Strong states, whereas deployment states and incumbents come from neural\n   search. This is a limitation of imitation learning, not train/test leakage.\n   The script preserves the required instance split and train-only statistics,\n   then evaluates the fixed checkpoint on separately seeded unseen graphs.\n4. **Masked padding and Adam.** Padding is finite zero after standardization,\n   masks are boolean, each recorded node has at least two real candidates, and\n   padded targets are zero. A finite -1e30 masked logit avoids zero times\n   negative infinity. Adam moments start at zero; its bias-correction step\n   increments once per batch from one. Training stops on non-finite CE and\n   checks zero padded probability.\n5. **Matrix layout and exactness.** Rust computes w1[input][hidden] and the same\n   normalization, biases and tanh. Six probes compare actual probabilities and\n   record logits. Neural only chooses an eligible spin and makes no trial LPs;\n   the existing exact partition, incumbent and pruning logic are retained.\n   Non-finite model coefficients/features/scores produce an error instead of\n   allowing an unchecked argmax. A fallback is not needed to report failure\n   safely. Both branch policies remain checked against actual exact energies.\n\nChatGPT's pre-delivery development checks passed four Rust tests and a separate\nsmall n=6 label/train/evaluate/plot pipeline. That pipeline ran 120 JAX epochs\nand checked six Rust/JAX probes, exact energies, no graph overlap, LP accounting\nand the figure from raw JSON. It is not a substitute for the formal n=22 local\nexperiments. The bootstrap runs the formal dataset and unseen evaluation on the\nuser's computer and saves the actual results, including any failed criterion.\n<!-- WEEK12_PART3_END -->\n")
PYDOCS

export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
printf '\n[2/6] Checking feature formulas, exactness, LP accounting and workshop pruning...\n'
cargo test --release --locked -- --test-threads=1 2>&1 | tee artifacts/bnb_labels/tests.log
cargo install --path . --force --locked 2>&1 | tee artifacts/bnb_labels/build.log
branchopt_bin="${CARGO_HOME:-$HOME/.cargo}/bin/branchopt"
printf '\n[3/6] Exporting Strong labels from 48 graphs (36 training / 12 validation instances)...\n'
"$branchopt_bin" --mode bnb --spins 22 --edges 52 --count 48 --policy strong \
    --export-labels --seed 2026 --out artifacts/bnb_labels \
    2>&1 | tee artifacts/bnb_labels/stdout.txt
python3 - <<'PYLABELS'
from pathlib import Path
import json
old=json.loads(Path('artifacts/bnb_strong/run.json').read_text())
new=json.loads(Path('artifacts/bnb_labels/run.json').read_text())
assert new['validation_status']=='passed'
assert len(old['instances'])==len(new['instances'])==48
for a,b in zip(old['instances'],new['instances']):
    assert a['id']==b['id'] and a['instance_seed']==b['instance_seed'] and a['graph']==b['graph'], 'Label run graph differs from Part 2'
    assert a['search']['nodes']==b['search']['nodes'], 'Export changed Strong explored nodes'
    assert a['search']['lp_solves']==b['search']['lp_solves'], 'Export changed Strong LP accounting'
print('Label export agrees with all 48 original Part 2 graphs and search counts.')
PYLABELS
cp -- artifacts/bnb_labels/dataset.json artifacts/bnb_strong/dataset.json
printf '\n[4/6] Training a 12-48-1 MLP for 120 epochs using JAX CPU float64...\n'
uv run --no-project --python 3.12 --with jax==0.12.1 \
    python train_branch.py --dataset artifacts/bnb_strong/dataset.json --out artifacts/model.json \
    2>&1 | tee artifacts/bnb_labels/training.log
printf '\n[5/6] Verifying six JAX/Rust probes and comparing three policies on 16 unseen graphs...\n'
"$branchopt_bin" --mode eval --spins 22 --edges 52 --count 16 --seed 3026 \
    --model artifacts/model.json --out artifacts/eval_test \
    2>&1 | tee artifacts/eval_test/stdout.txt
uv run --no-project --python 3.12 --with matplotlib==3.10.7 \
    python plot_neural_bnb.py --model artifacts/model.json --run artifacts/eval_test/run.json \
    --out artifacts/neural_bnb.png --require-course-settings \
    2>&1 | tee artifacts/eval_test/comparison.txt
{
    date -Iseconds
    uname -srmo
    rustc --version
    cargo --version
    cmake --version
    uv --version
} > artifacts/eval_test/environment.txt
printf '\n[6/6] Committing the Part 3 source files locally...\n'
files=(src/main.rs src/bnb.rs src/nn.rs train_branch.py train_branch.design.toml plot_neural_bnb.py branchopt.design.toml README.md DEEPSEEK_REVIEW.md run_part3.sh)
git add -- "${files[@]}"
if git diff --cached --quiet -- "${files[@]}"; then
    printf 'Part 3 source is already committed.\n'
else
    git commit --only -m "week12: learn branching policy with JAX and verify Rust inference" -- "${files[@]}"
fi
printf '\nPART 3 COMPLETE\nFigure: %s/artifacts/neural_bnb.png\n' "$PWD"
printf 'Model: %s/artifacts/model.json\n' "$PWD"
printf 'Checks: artifacts/eval_test/part3_checks.json\n'
printf 'Raw runs: artifacts/bnb_labels/run.json and artifacts/eval_test/run.json\n'
printf 'Source was committed locally; no git push was run.\n'
