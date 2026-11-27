#!/usr/bin/env bash
# Run inside the existing course repository's week12/ after Part 3.
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
for file in Cargo.toml Cargo.lock src/main.rs src/bnb.rs src/nn.rs train_branch.py train_branch.design.toml branchopt.design.toml README.md artifacts/model.json artifacts/eval_test/part3_checks.json; do
    if [[ ! -f "$file" ]]; then
        printf 'Missing Part 3 file: %s\n' "$file" >&2
        exit 1
    fi
done
python3 - <<'PYFIRST'
from pathlib import Path
import json
report=json.loads(Path('artifacts/eval_test/part3_checks.json').read_text())
if report.get('all_passed') is not True:
    raise SystemExit('Part 3 checks must all pass before Part 4.')
model=json.loads(Path('artifacts/model.json').read_text())
a=model['data_provenance']['parameters']
if (a['spins'],a['edges'],a['count'],a['seed'])!=(22,52,48,2026):
    raise SystemExit('The trained model must use the course n=22,m=52,count=48,seed=2026 dataset.')
for figure in ['gap.png','bnb.png','neural_bnb.png']:
    if not Path('artifacts',figure).exists():
        raise SystemExit('Missing earlier figure: '+figure)
print('Part 3 checks: PASSED. Reusing the existing model without retraining.')
print('Saved validation CE:',model['training_summary']['best_validation_ce'])
PYFIRST
git rev-parse --show-toplevel >/dev/null
if [[ -n "$(git ls-files -- artifacts)" ]]; then
    printf 'Some artifacts are tracked in Git. Send git status output before publishing.\n' >&2
    exit 1
fi
if ! git check-ignore -q artifacts/model.json; then
    printf 'The existing artifacts/ ignore rule is missing. Send git status output.\n' >&2
    exit 1
fi
mkdir -p src artifacts/eval_test artifacts/eval_transfer
backup="artifacts/setup_backups/part4-$(date +%Y%m%dT%H%M%S)-$$"
mkdir -p "$backup"
for file in src/main.rs src/bnb.rs plot_eval.py summarize_results.py branchopt.design.toml README.md DEEPSEEK_REVIEW.md RESULTS.md run_part4.sh artifacts/eval.png artifacts/part4_checks.json artifacts/part4_model.json artifacts/part4_environment.json; do
    if [[ -f "$file" ]]; then
        mkdir -p "$backup/$(dirname "$file")"
        cp -- "$file" "$backup/$file"
    fi
done
cp -a -- artifacts/eval_test "$backup/eval_test"
cp -a -- artifacts/eval_transfer "$backup/eval_transfer"
if [[ ! -d artifacts/part3_eval_snapshot ]]; then
    cp -a -- artifacts/eval_test artifacts/part3_eval_snapshot
fi
if [[ "$(realpath "$0")" != "$(realpath -m run_part4.sh)" ]]; then
    cp -- "$0" run_part4.sh
fi
python3 - <<'PYMODELBEFORE'
from pathlib import Path
import hashlib,json
p=Path('artifacts/model.json')
Path('artifacts/part4_model.json').write_text(json.dumps({'model_path':str(p),'before_sha256':hashlib.sha256(p.read_bytes()).hexdigest()},indent=2))
PYMODELBEFORE
printf '\n[1/6] Adding root-cached branching, timing audit and result summary...\n'
cat > 'src/main.rs' <<'WEEK12_PART4_FILE'
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
            println!("branchopt --mode gap|bnb|eval --policy most-frac|strong|neural|root-neural --export-labels --model artifacts/model.json --spins 22 --edges 52 --count 48 --seed 2026 --out artifacts/bnb_strong");
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
    check(["most-frac","strong","neural","root-neural"].contains(&a.policy.as_str()), "unsupported branching policy")?;
    check(!a.export_labels || (a.mode=="bnb" && a.policy=="strong" && a.count<=48),
          "--export-labels requires bnb mode, strong policy and count<=48")?;
    check((a.mode!="eval" && !["neural","root-neural"].contains(&a.policy.as_str())) || a.model.is_some(),
          "neural/root-neural policies and eval mode require --model")?;
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
WEEK12_PART4_FILE
cat > 'src/bnb.rs' <<'WEEK12_PART4_FILE'
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
    feature_calls: usize, inference_calls: usize,
    root_raw_scores: Option<Vec<f64>>, root_scores: Option<Vec<f64>>,
    #[serde(skip_serializing)] labels: Vec<LabelNode>,
}

fn standardized_root_scores(raw: &[f64]) -> Result<Vec<f64>> {
    check(!raw.is_empty() && raw.iter().all(|v|v.is_finite()), "invalid root logits")?;
    let average=raw.iter().sum::<f64>()/raw.len() as f64;
    let std=(raw.iter().map(|v|(v-average).powi(2)).sum::<f64>()/raw.len() as f64).sqrt();
    let scaled:Vec<f64>=raw.iter().map(|v|(v-average)/std.max(1e-6)).collect();
    check(scaled.iter().all(|v|v.is_finite()), "invalid standardized root scores")?;
    Ok(scaled)
}

fn search(g: &Graph, policy: &str, model: Option<&nn::Network>, export_labels: bool) -> Result<SearchResult> {
    check(["most-frac","strong","neural","root-neural"].contains(&policy), "unsupported B&B policy")?;
    check(!["neural","root-neural"].contains(&policy) || model.is_some(), "learned policy requires a model")?;
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
    let (mut feature_calls,mut inference_calls)=(0usize,0usize);
    let mut root_raw_scores=None;
    let mut root_scores:Option<Vec<f64>>=None;
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
        if policy=="root-neural" && node.depth==0 {
            let t=Instant::now();
            let all_spins:Vec<usize>=(0..g.n).collect();
            let root_features=feature_rows(g,&lp,&node.fixed,incumbent,&all_spins);
            feature_calls+=1;
            let raw=model.unwrap().scores(&root_features)?;
            inference_calls+=1;
            root_scores=Some(standardized_root_scores(&raw)?);
            root_raw_scores=Some(raw);
            inference_ms+=t.elapsed().as_secs_f64()*1e3;
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
            feature_calls+=1;
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
            inference_calls+=1;
            let mut best=f64::NEG_INFINITY;
            for (k,&score) in scores.iter().enumerate() {
                if score>best+TIE {best=score;chosen=candidates[k];}
            }
            inference_ms+=t.elapsed().as_secs_f64()*1e3;
        }
        if policy=="root-neural" {
            let cached=root_scores.as_ref().ok_or_else(||fail("missing root score cache"))?;
            let mut best=f64::NEG_INFINITY;
            for &i in &candidates {
                let score=cached[i]+3.0*2.0*lp.x[i].min(1.0-lp.x[i]);
                if score>best+TIE {best=score;chosen=i;}
            }
            audit["cached_root_score"]=json!(cached[chosen]);
            audit["branch_score"]=json!(best);
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
        node_lp_solves:nodes,trial_lp_solves:trials,total_ms,lp_ms,inference_ms,first_splits,trace,
        feature_calls,inference_calls,root_raw_scores,root_scores,labels })
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
    let policies=["most-frac","strong","neural","root-neural"];
    let start=Instant::now();
    let mut rows=Vec::new();
    let mut counts=vec![0.0;4]; let mut times=vec![0.0;4]; let mut solves=vec![0.0;4];
    let mut lp_times=vec![0.0;4]; let mut inference_times=vec![0.0;4];
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
                && (policy=="strong" || (result.trial_lp_solves==0 && result.lp_solves==result.nodes))
                && (policy!="root-neural" || (result.inference_calls==1 && result.feature_calls==1));
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
        "root_neural_saves_nodes":counts[3]<counts[0],
        "probability_parity":parity,"largest_saving":savings});
    let run=json!({"schema":"branchopt.eval.v1","parameters":a,"mode":"eval",
        "validation_status":if valid{"passed"}else{"failed_exactness_or_accounting"},
        "settings":{"presolve":false,"threads":1,"oracle_in_search_time":false,
            "initial_incumbent":"none","root_score_standard_deviation":"population; floor 1e-6",
            "root_fractionality_weight":3.0,"inference_ms_includes":"feature extraction and network; root normalization included",
            "total_ms_includes":"LP setup/solve/extraction, branch management, feature/network cost, trace recording"},
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
            for policy in ["neural","root-neural"] {
                let result=search(&g,policy,Some(&network),false)?;
                assert!((result.energy-expected).abs()<1e-9);
                assert_eq!(result.lp_solves,result.nodes);
                assert_eq!(result.trial_lp_solves,0);
                if policy=="root-neural" {
                    assert_eq!(result.feature_calls,1);
                    assert_eq!(result.inference_calls,1);
                    let raw=result.root_raw_scores.as_ref().unwrap();
                    let cached=result.root_scores.as_ref().unwrap();
                    assert_eq!(raw.len(),g.n);
                    assert_eq!(cached.len(),g.n);
                    for row in &result.trace {
                        if row["status"]!="branched" {continue;}
                        let mut chosen=usize::MAX; let mut best=f64::NEG_INFINITY;
                        for index in row["candidates"].as_array().unwrap() {
                            let i=index.as_u64().unwrap() as usize;
                            let x=row["x"][i].as_f64().unwrap();
                            let score=cached[i]+3.0*2.0*x.min(1.0-x);
                            if score>best+TIE {best=score;chosen=i;}
                        }
                        assert_eq!(row["chosen_spin"],json!(chosen));
                    }
                }
            }
        }
        Ok(())
    }
    #[test]
    fn root_standardization_uses_population_std_and_floor() -> Result<()> {
        assert_eq!(standardized_root_scores(&[1.0,3.0])?,vec![-1.0,1.0]);
        assert_eq!(standardized_root_scores(&[7.0,7.0,7.0])?,vec![0.0;3]);
        let tiny=standardized_root_scores(&[0.0,1e-8])?;
        assert!((tiny[0]+0.005).abs()<1e-12 && (tiny[1]-0.005).abs()<1e-12);
        Ok(())
    }
}
WEEK12_PART4_FILE
cat > 'plot_eval.py' <<'WEEK12_PART4_FILE'
#!/usr/bin/env python3
"""Audit actual four-policy runs, plot size transfer, and preserve all criteria."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

POLICIES=['most-frac','strong','neural','root-neural']
LEARNED=['neural','root-neural']

def graph_hash(g):
    return hashlib.sha256(json.dumps(g,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def root_cache_ok(search,n):
    raw=search['root_raw_scores']; cached=search['root_scores']
    if search['feature_calls']!=1 or search['inference_calls']!=1 or len(raw)!=n or len(cached)!=n:
        return False
    average=sum(raw)/n
    std=math.sqrt(sum((v-average)**2 for v in raw)/n)
    if max(abs(b-(a-average)/max(std,1e-6)) for a,b in zip(raw,cached))>1e-8:
        return False
    for node in search['trace']:
        if node['status']!='branched':
            continue
        best=-math.inf; chosen=None
        for i in node['candidates']:
            x=node['x'][i]
            score=cached[i]+3*2*min(x,1-x)
            if score>best+1e-12:
                best=score; chosen=i
        if chosen!=node['chosen_spin'] or abs(best-node['branch_score'])>1e-8:
            return False
    return True

def audit(run):
    rows=run['instances']; checks={}
    checks['rust_validation_passed']=run['validation_status']=='passed'
    checks['four_policies_present']=all(set(r['searches'])==set(POLICIES) for r in rows)
    maximum=0.0
    for row in rows:
        g=row['graph']
        for policy in POLICIES:
            s=row['searches'][policy]; bits=s['bits']
            assert len(bits)==g['n'] and set(bits)<={0,1}
            spins=[1-2*b for b in bits]
            physical=sum(e['j']*spins[e['u']]*spins[e['v']] for e in g['edges'])+sum(h*v for h,v in zip(g['h'],spins))
            maximum=max(maximum,abs(s['energy']-row['oracle']['energy']),abs(s['energy']-physical))
    checks['all_energies_exact']=maximum<=1e-6
    checks['lp_accounting']=all(s['node_lp_solves']==s['nodes'] and s['lp_solves']==s['nodes']+s['trial_lp_solves'] for r in rows for s in r['searches'].values())
    checks['non_strong_lp_solves_equal_nodes']=all(r['searches'][p]['lp_solves']==r['searches'][p]['nodes'] and r['searches'][p]['trial_lp_solves']==0 for r in rows for p in ['most-frac']+LEARNED)
    checks['root_scored_once_and_cache_rule_correct']=all(root_cache_ok(r['searches']['root-neural'],r['graph']['n']) for r in rows)
    checks['neural_scored_once_per_branch']=all(r['searches']['neural']['feature_calls']==r['searches']['neural']['inference_calls']==sum(n['status']=='branched' for n in r['searches']['neural']['trace']) for r in rows)
    checks['timers_consistent']=all(all(math.isfinite(s[k]) and s[k]>=0 for k in ['total_ms','lp_ms','inference_ms']) and s['total_ms']+1e-6>=s['lp_ms']+s['inference_ms'] for r in rows for s in r['searches'].values())
    parity=run['summary']['probability_parity']
    checks['six_jax_rust_probes']=parity['probe_count']==6 and parity['max_probability_error']<1e-4
    summaries={}
    for p in POLICIES:
        values=[r['searches'][p] for r in rows]
        summaries[p]={f'mean_{key}':sum(s[key] for s in values)/len(values) for key in ['nodes','lp_solves','total_ms','lp_ms','inference_ms']}
        summaries[p]['lp_solves_equal_nodes']=all(s['lp_solves']==s['nodes'] for s in values)
    performance={}
    for p in LEARNED:
        summaries[p]['strong_speedup']=summaries['strong']['mean_total_ms']/max(summaries[p]['mean_total_ms'],1e-12)
        summaries[p]['most_frac_speedup']=summaries['most-frac']['mean_total_ms']/max(summaries[p]['mean_total_ms'],1e-12)
        performance[p+'_fewer_nodes']=summaries[p]['mean_nodes']<summaries['most-frac']['mean_nodes']
        # A conservative, explicit operational check for "many times faster".
        performance[p+'_at_least_2x_faster_than_strong']=summaries[p]['strong_speedup']>=2
    return {'checks':checks,'performance_checks':performance,'max_energy_error':maximum,'policies':summaries,'parameters':run['parameters'],'probability_parity':parity}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--test',default='artifacts/eval_test/run.json')
    parser.add_argument('--transfer',default='artifacts/eval_transfer/run.json')
    parser.add_argument('--model',default='artifacts/model.json')
    parser.add_argument('--model-provenance')
    parser.add_argument('--out',default='artifacts/eval.png')
    parser.add_argument('--require-course-settings',action='store_true')
    args=parser.parse_args()
    runs=[json.loads(Path(p).read_text()) for p in [args.test,args.transfer]]
    model_bytes=Path(args.model).read_bytes(); model=json.loads(model_bytes)
    reports=[audit(r) for r in runs]
    source_hashes=set(model['data_provenance']['graph_hashes'])
    eval_hashes=[{graph_hash(row['graph']) for row in run['instances']} for run in runs]
    shared={'unseen_graphs':all(not source_hashes&h for h in eval_hashes),
        'evaluation_sets_disjoint':not bool(eval_hashes[0]&eval_hashes[1])}
    if args.model_provenance:
        identity=json.loads(Path(args.model_provenance).read_text())
        current=hashlib.sha256(model_bytes).hexdigest()
        shared['model_file_unchanged']=identity['before_sha256']==identity['after_sha256']==current
    if args.require_course_settings:
        source=model['data_provenance']['parameters']; t=model['training_summary']
        shared['course_settings']=(source['spins'],source['edges'],source['count'],source['seed'])==(22,52,48,2026) and \
            all((r['parameters']['spins'],r['parameters']['edges'],r['parameters']['count'],r['parameters']['seed'])==expected for r,expected in zip(runs,[(22,52,16,3026),(30,72,8,4026)])) and \
            (t['epochs'],t['batch_size'],t['seed'])==(120,64,2026)
        shared['model_identity_recorded']='model_file_unchanged' in shared and shared['model_file_unchanged']

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.1),layout='constrained')
    labels=['Most fractional','Strong','Neural','Root-neural']
    colors=['#527ab3','#d5923b','#399875','#8c6ab0']
    positions=np.arange(2)
    for k,(p,color,label) in enumerate(zip(POLICIES,colors,labels)):
        axes[0].bar(positions+(k-1.5)*0.19,[r['policies'][p]['mean_nodes'] for r in reports],width=0.18,label=label,color=color)
    axes[0].set_xticks(positions,[f"{r['parameters']['spins']} spins / {r['parameters']['edges']} bonds" for r in runs])
    axes[0].set(ylabel='Mean explored nodes',title='Same model on two graph sizes')
    axes[0].legend(frameon=False)
    big=reports[1]['policies']; pos=np.arange(4)
    lp=np.array([big[p]['mean_lp_ms'] for p in POLICIES])
    inference=np.array([big[p]['mean_inference_ms'] for p in POLICIES])
    total=np.array([big[p]['mean_total_ms'] for p in POLICIES])
    other=np.maximum(0,total-lp-inference)
    axes[1].bar(pos,lp,label='LP setup, solve, extraction',color='#a6b3c7')
    axes[1].bar(pos,inference,bottom=lp,label='Features and inference',color='#399875')
    axes[1].bar(pos,other,bottom=lp+inference,label='Tree management and recording',color='#e0aa68')
    for i in range(4):
        axes[1].text(i,total[i]+max(total)*0.025,f'{total[i]:.2f} ms\ninference {inference[i]:.3f}',ha='center',va='bottom',fontsize=8)
    axes[1].set_xticks(pos,['Most-frac','Strong','Neural','Root-neural'])
    axes[1].set(ylabel='Mean search wall time (ms)',title=f"Time breakdown: {runs[1]['parameters']['spins']} spins",ylim=(0,max(total)*1.27))
    axes[1].legend(frameon=False,fontsize=8,loc='upper right')
    for ax in axes: ax.grid(axis='y',alpha=0.16); ax.set_axisbelow(True)
    fig.suptitle('Size transfer and the cost of learned branching')
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=180)

    all_passed=all(shared.values()) and all(all(r['checks'].values()) and all(r['performance_checks'].values()) for r in reports)
    result={'all_passed':all_passed,'shared_checks':shared,'model_sha256':hashlib.sha256(model_bytes).hexdigest(),
        'performance_threshold_note':'At least 2x faster than Strong is an explicit conservative script criterion; all actual speedups are reported.',
        'evaluations':reports}
    report_path=out.parent/'part4_checks.json'
    report_path.write_text(json.dumps(result,indent=2,allow_nan=False))
    print('Saved',out.resolve()); print('Saved',report_path.resolve())
    for report in reports:
        a=report['parameters']
        print(f"\n{a['count']} graphs: n={a['spins']}, m={a['edges']}, seed={a['seed']}")
        print('Policy          Nodes   LP solves   Total ms      LP ms   Inference ms')
        for p in POLICIES:
            s=report['policies'][p]
            print(f"{p:<13} {s['mean_nodes']:7.3f} {s['mean_lp_solves']:11.3f} {s['mean_total_ms']:10.3f} {s['mean_lp_ms']:10.3f} {s['mean_inference_ms']:14.4f}")
        print('Largest energy error:',report['max_energy_error'])
        print('LP solves = nodes:',{p:report['policies'][p]['lp_solves_equal_nodes'] for p in ['most-frac']+LEARNED})
        print('Speedups over Strong:',{p:round(report['policies'][p]['strong_speedup'],3) for p in LEARNED})
        print('Correctness:',json.dumps(report['checks']))
        print('Performance:',json.dumps(report['performance_checks']))
    print('Shared checks:',json.dumps(shared))
    if not all_passed:
        raise SystemExit('Part 4 criteria not met. Actual runs, figure and checks have been preserved.')
    print('\nPART 4 VALIDATION PASSED')

if __name__=='__main__': main()
WEEK12_PART4_FILE
cat > 'summarize_results.py' <<'WEEK12_PART4_FILE'
#!/usr/bin/env python3
"""Generate the course evidence summary from the user's actual local run files."""
import json
from pathlib import Path

def read(path): return json.loads(Path(path).read_text())

def main():
    gap=read('artifacts/gap/run.json')
    most=read('artifacts/bnb_most/run.json')
    strong=read('artifacts/bnb_strong/run.json')
    part3=read('artifacts/eval_test/part3_checks.json')
    part4=read('artifacts/part4_checks.json')
    model=read('artifacts/model.json')
    environment=read('artifacts/part4_environment.json')
    assert gap['validation_status']==most['validation_status']==strong['validation_status']=='passed'
    assert part3['all_passed'] and part4['all_passed']
    for name in ['gap.png','bnb.png','neural_bnb.png','eval.png']:
        assert Path('artifacts',name).is_file(),f'Missing figure: {name}'
    g=gap['summary']; t=model['training_summary']
    lines=['# Week 12: Measured results and presentation notes','',
        'All numbers below come from the saved local JSON evidence. Timings are single-run wall-clock measurements on this computer and include search bookkeeping and recording. They are not a statistical hardware benchmark.','',
        '## 1. MILP, relaxation and rounding','',
        'Let x_i=(1-s_i)/2 and introduce continuous bond variables z_ij in [0,1]. Minimize C-2 sum_i h_i*x_i-2 sum_(i,j) J_ij*z_ij, where C=sum_i h_i+sum_(i,j) J_ij. Each bond obeys z<=x_i+x_j, z<=2-x_i-x_j, z>=x_i-x_j and z>=x_j-x_i. Binary spins force z to their XOR.','',
        'Relaxing the spins to continuous [0,1] permits x_i=1/2, so each bond can independently take its favorable z value. This hides inconsistent spin preferences around frustrated cycles. Rounding the spins does not preserve the favorable LP bond values.','',
        f"On {gap['parameters']['count']} graphs (n={gap['parameters']['spins']}, m={gap['parameters']['edges']}, seed={gap['parameters']['seed']}):",'',
        '| Metric | Actual mean |','| --- | ---: |',
        f"| LP lower bound | {g['mean_lp_bound']:.6f} |",
        f"| Exact ground energy | {g['mean_exact_energy']:.6f} |",
        f"| Integrality gap | {g['mean_integrality_gap']:.6f} |",
        f"| Rounding excess above the optimum | {g['mean_rounding_excess']:.6f} |",'',
        'The four-spin frustrated square has LP -4, exact energy -2 and nearest-rounded energy +2; the local reference check verifies these values. In the opening workshop maximum problem, the actual search visits the root LP (83/3), the integer x2<=2 child (27), and prunes the x2>=3 child with LP upper bound 26.75.','',
        'Figure: [gap.png](artifacts/gap.png).','',
        '## 2. The price of Strong Branching','',
        'Most-frac chooses the spin closest to 1/2. Strong Branching tests both children of every candidate, using the product of child bound gains. Trials never update the incumbent or seed a cache. A smaller tree can therefore cost many more LP solves.','',
        '| Policy | Mean nodes | Mean LP solves | Mean total ms |',
        '| --- | ---: | ---: | ---: |']
    for p,r in [('most-frac',most),('strong',strong)]:
        s=r['summary']
        lines.append(f"| {p} | {s['mean_nodes']:.3f} | {s['mean_lp_solves']:.3f} | {s['mean_total_ms']:.3f} |")
    lines+=['','These values use the original 48-instance Part 2 run files, whose timing evidence was preserved. Figure: [bnb.png](artifacts/bnb.png).','',
        '## 3. Imitating Strong without its trial LPs','',
        f"Training: {t['train_nodes']} nodes from {t['train_instances']} graphs. Validation: {t['validation_nodes']} nodes from {t['validation_instances']} disjoint graphs. Instance overlap: {len(t['instance_overlap'])}.",
        f"Best saved checkpoint: epoch {t['best_epoch']}; validation CE without regularization: {t['best_validation_ce']:.6f}.",
        f"Largest six-probe JAX/Rust probability difference: {part3['probability_parity']['max_probability_error']:.3e}. Largest test energy error: {part3['max_energy_error']:.3e}.",'',
        'The shared 12-48(tanh)-1 candidate scorer uses local, size-independent graph and LP features. Training-only mean/std and masked candidate softmax prevent validation statistics and padded candidates from influencing training. Labels are split by graph, not by node. The model changes the branching spin; LP bounds, complete child partitions and feasible incumbents retain responsibility for exactness.','',
        '| Policy | Original Part 3 mean nodes | Original mean total ms |',
        '| --- | ---: | ---: |']
    for p in ['most-frac','neural','strong']:
        s=part3['policies'][p]
        lines.append(f"| {p} | {s['mean_nodes']:.3f} | {s['mean_total_ms']:.3f} |")
    largest=part3['largest_saving']
    lines+=['',f"Largest Part 3 saving: instance {largest['instance_id']}, exact energy {largest['ground_energy']:.6f}. Most-frac: {largest['most-frac']['nodes']} nodes, first splits {largest['most-frac']['first_splits']}; neural: {largest['neural']['nodes']} nodes, first splits {largest['neural']['first_splits']}.",'',
        'Figure: [neural_bnb.png](artifacts/neural_bnb.png).','',
        '## 4. Root caching, time breakdown and size transfer','',
        'Root-neural scores all spins once at the root, centers its logits and divides by max(population std,1e-6). It then chooses the candidate maximizing cached_root_score + 3*2*min(x_i,1-x_i). It computes no deeper-node features or network outputs. Neural recomputes features and scores at each branch node. The 22-spin model is reused unchanged on the 30-spin graphs; there is no transfer retraining.','',
        '| Graph size / count / seed | Policy | Nodes | LP solves | Total ms | LP ms | Features + inference ms | Speedup vs Strong |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for evaluation in part4['evaluations']:
        a=evaluation['parameters']
        for p in ['most-frac','strong','neural','root-neural']:
            s=evaluation['policies'][p]
            speed=f"{s['strong_speedup']:.2f}x" if 'strong_speedup' in s else '—'
            lines.append(f"| {a['spins']} spins / {a['count']} / {a['seed']} | {p} | {s['mean_nodes']:.3f} | {s['mean_lp_solves']:.3f} | {s['mean_total_ms']:.3f} | {s['mean_lp_ms']:.3f} | {s['mean_inference_ms']:.4f} | {speed} |")
    maximum=max(r['max_energy_error'] for r in part4['evaluations'])
    lines+=['',f"Largest energy error across both sizes and all four policies: {maximum:.3e}. Every non-Strong rule has LP solves equal to explored nodes. Root scoring is checked to run exactly once per graph, and its candidate choice is independently recomputed from the saved cache at every branch.",'',
        'The time comparison includes feature and network cost. Model loading, the six probes, separate HiGHS MILP oracles and final JSON writing are outside each search timer. LP time includes constructing, solving and extracting both visited LPs and Strong trial LPs. Root cached-score arithmetic is counted as tree management. The plotted remainder also includes trace recording.','',
        'Both learned policies must reduce mean nodes on both sizes. The script also checks at least 2x speedup over Strong, a conservative numerical interpretation of the learning sheet\'s qualitative timing requirement. Actual ratios are shown above; no reference values are substituted.','',
        'Figure: [eval.png](artifacts/eval.png).','',
        '## Reproduction, environment and evidence','',
        'Run the saved stages in order from week12/:','',
        '```bash','bash run_part1.sh','bash run_part2.sh','bash run_part3.sh','bash run_part4.sh','```','',
        f"Model SHA-256 before and after both evaluations: `{part4['model_sha256']}`.",'',
        '```json',json.dumps(environment,indent=2,ensure_ascii=False),'```','',
        'Keep the four figures, raw run folders, pred files and model.json under artifacts/ on this computer. Source, both design files, reproduction/plot scripts, README.md, this results sheet and the independent review log belong in Git. The artifact directory is ignored.','',
        'All formal statistics in this document are produced by the local reproduction runs. DEEPSEEK_REVIEW.md records the actual static-review scope and the response to each material comment.']
    Path('RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('Saved',Path('RESULTS.md').resolve())

if __name__=='__main__': main()
WEEK12_PART4_FILE
python3 - <<'PYDOCS'
from pathlib import Path
import re
p=Path('branchopt.design.toml')
t=p.read_text() if p.exists() else ''
t=re.sub(re.escape('# WEEK12_PART4_BEGIN')+r'.*?'+re.escape('# WEEK12_PART4_END')+r'\n?', '', t, flags=re.S)
p.write_text(t.rstrip()+'\n\n'+'# WEEK12_PART4_BEGIN\n[benchmark.eval.root_neural]\npolicy = """\nScore all n spins at the root. Subtract their mean and divide by max(std,1e-6),\nusing the population standard deviation. At each branch maximize this cached\nroot score + 3 * 2 min(x_i,1-x_i) over eligible candidate spins.\n"""\ntiming = "Record total search wall time, LP time and feature-plus-inference time. Include the single root cache construction in total and inference time; subsequent cached-score arithmetic belongs to tree management."\naudit = "Store feature/inference call counts, raw root logits, standardized cache and branch trace. Require exactly one root feature/network call per instance and no non-Strong trial LPs."\ntransfer = "Reuse the unchanged model on 16 graphs n=22,m=52,seed=3026 and 8 graphs n=30,m=72,seed=4026, without retraining. Compare most-frac, strong, neural and root-neural on each graph."\nevidence = "artifacts/eval.png and artifacts/part4_checks.json. Show nodes for both sizes and total/LP/inference wall time on the larger graphs. Keep actual timing and performance outcomes."\n# WEEK12_PART4_END\n')
p=Path('README.md')
t=p.read_text() if p.exists() else ''
t=re.sub(re.escape('<!-- WEEK12_PART4_BEGIN -->')+r'.*?'+re.escape('<!-- WEEK12_PART4_END -->')+r'\n?', '', t, flags=re.S)
p.write_text(t.rstrip()+'\n\n'+'<!-- WEEK12_PART4_BEGIN -->\n## Part 4: Inference overhead and larger graphs\n\nAfter Part 3, run `bash ./week12_part4.sh` from the same `week12/` directory.\nThe saved `run_part4.sh` is a self-contained reproduction of this stage.\nThe script reuses `artifacts/model.json` without retraining, fingerprints its\nbytes before and after both evaluations, checks the exact-search implementation,\nruns all four policies at both graph sizes, plots actual evidence and creates\n`RESULTS.md` from local JSON files. Original Part 3 evidence is copied to\n`artifacts/part3_eval_snapshot/` before the 22-spin evaluation is refreshed.\nExisting stage files and overwritten evidence are backed up under artifacts/.\n\nRoot-neural extracts all n spin features once from the root LP, scores them\nwith the same trained MLP, and standardizes the root logits with their population\nmean/std (std floor 1e-6). At each later branch it chooses the candidate with\nthe largest cached root score + 3*2*min(x_i,1-x_i). The cache is local to one\nsearch invocation and never crosses graphs or policies. No deeper-node features\nor forward passes occur. Root scores are computed even for an integral root.\nNeither learned policy changes any bounds, incumbent rules or child partitions.\nAll non-Strong policies have zero trial solves and LP solves equal visited nodes.\n\nTiming is collected inside each search. Total time includes all LP construction,\nsolve/extraction, graph-feature and network work, branch management and trace\nrecording. LP time includes Strong trial solves. Feature-plus-inference time\nincludes all per-node neural work, or the one root feature/forward/normalization\nstep. Later cached-score arithmetic belongs to tree management. Oracle MILPs,\nmodel loading, six-probe verification and final file writing are outside search\ntimers. These are actual single-run wall-clock measurements; no timing guarantee\nor statistical claim is inferred from one laptop run.\n\nReproduce the evaluation and chart:\n\n```bash\ncargo test --release --locked -- --test-threads=1\ncargo install --path . --force --locked\nbranchopt --mode eval --spins 22 --edges 52 --count 16 --seed 3026 --model artifacts/model.json --out artifacts/eval_test\nbranchopt --mode eval --spins 30 --edges 72 --count 8 --seed 4026 --model artifacts/model.json --out artifacts/eval_transfer\nuv run --no-project --python 3.12 --with matplotlib==3.10.7 python plot_eval.py\npython3 summarize_results.py\n```\n\nFor the fingerprint, complete environment capture and strict course-setting\nchecks, use `bash run_part4.sh`. The script saves actual runs and the figure\nbefore deciding whether performance criteria pass. It validates every energy\nagainst the separate HiGHS optimum, LP accounting, six probability probes,\none root scoring call, every cached-score branch choice, disjoint unseen test\ngraphs, unchanged model bytes, and consistent timer components. Both neural\nrules must use fewer mean nodes than most-frac at both sizes. To give the\nlearning sheet\'s qualitative "many times faster than Strong" requirement a\ntestable threshold, this script requires at least 2x while printing the full\nmeasured ratios. A failed performance criterion keeps all actual evidence.\n\nThe five small Rust tests cover enumeration-based exactness, feature formulas,\nthe three-node workshop, population root normalization (including constant and\ntiny-variance logits), and root-cache use through actual search traces. A small\ndevelopment-only two-size integration checks JSON export, plotting and the\nindependent cache audit; formal 22/30-spin experiments run on the user\'s PC.\n\nDeliverable figures are `artifacts/gap.png`, `artifacts/bnb.png`,\n`artifacts/neural_bnb.png` and `artifacts/eval.png`. Keep them with all raw runs,\npred files and model.json locally. `RESULTS.md` contains measured tables,\nenvironment details and notes for explaining the four figures. After the final\nlocal commit, publish the source commits with `git push` from the course repo.\n<!-- WEEK12_PART4_END -->\n')
p=Path('DEEPSEEK_REVIEW.md')
t=p.read_text() if p.exists() else ''
t=re.sub(re.escape('<!-- WEEK12_PART4_BEGIN -->')+r'.*?'+re.escape('<!-- WEEK12_PART4_END -->')+r'\n?', '', t, flags=re.S)
p.write_text(t.rstrip()+'\n\n'+"<!-- WEEK12_PART4_BEGIN -->\n## Part 4: DeepSeek review of root caching and timing excerpts\n\nOn 2026-11-27, ChatGPT sent the actual new root-cache, population-normalization\nand branch-selection snippets plus the timing/call-count contract to DeepSeek:\nhttps://chat.deepseek.com/a/chat/s/fc47b89b-c851-4ed2-a26b-c7409e906862\n\nScope: independent static review of the displayed excerpts, not full-source\ninspection or program execution. DeepSeek reported no definite error, with four\npoints to verify. Their dispositions were checked against the full source:\n\n1. **Finite root features without an incumbent.** feature_rows already assigns\n   gap_ratio=1 when the incumbent is not finite, avoiding infinity/infinity.\n   Network inputs/outputs and standardized root scores are checked for finite\n   values. The feature example explicitly uses an infinite initial incumbent.\n2. **Cache indices.** all_spins is exactly 0..n, feature_rows maps the supplied\n   indices in order, and scores preserves that row order. The cache has length\n   n. The independent Python audit recomputes every root-policy branch choice\n   from the saved cache and current LP fractions; the small Rust test does so\n   independently through actual search traces as well.\n3. **Raw-score evidence.** Full code stores both root_raw_scores and root_scores\n   in the search result. Python recomputes population normalization from raw\n   logits. Creating these records is inside the search; writing final JSON is\n   outside its timer.\n4. **Per-search counters, feasibility and model guards.** Caches, counts and\n   timers are local variables initialized by every search invocation. Model\n   presence is checked before entering either neural policy. For this spin-glass\n   model the initial box is always feasible (for example all x=0,z=0), so every\n   root-policy course instance executes exactly one feature and forward call.\n   This is not a claim for arbitrary infeasible integer programs. Counters and\n   solve accounting are checked from actual evidence.\n\nPre-delivery development validation passed five Rust tests and a separate\ntwo-size n=6/n=8 inference/evaluation/plot pipeline using the prior small-graph\ndevelopment model. Both learned policies retained exactness, zero trial LPs,\nvalid timing components and correct root-cache behavior. No course n=22/n=30\nexperiment was substituted for the user's local run. The final bootstrap uses\nthe user's unchanged trained model, preserves raw results before performance\nchecks, and generates its result tables only from those local files.\n<!-- WEEK12_PART4_END -->\n")
PYDOCS

export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
printf '\n[2/6] Verifying five small-graph tests, root-cache choices and exact optima...\n'
cargo test --release --locked -- --test-threads=1 2>&1 | tee artifacts/eval_transfer/tests.log
cargo install --path . --force --locked 2>&1 | tee artifacts/eval_transfer/build.log
branchopt_bin="${CARGO_HOME:-$HOME/.cargo}/bin/branchopt"
printf '\n[3/6] Comparing all four policies on 16 unseen 22-spin graphs...\n'
"$branchopt_bin" --mode eval --spins 22 --edges 52 --count 16 --seed 3026 \
    --model artifacts/model.json --out artifacts/eval_test \
    2>&1 | tee artifacts/eval_test/stdout.txt
printf '\n[4/6] Testing the SAME model on 8 larger 30-spin, 72-bond graphs...\n'
"$branchopt_bin" --mode eval --spins 30 --edges 72 --count 8 --seed 4026 \
    --model artifacts/model.json --out artifacts/eval_transfer \
    2>&1 | tee artifacts/eval_transfer/stdout.txt
python3 - <<'PYMODELANDENV'
from pathlib import Path
from datetime import datetime
import hashlib,json,platform,subprocess,sys,tomllib
identity=Path('artifacts/part4_model.json')
proof=json.loads(identity.read_text())
proof['after_sha256']=hashlib.sha256(Path('artifacts/model.json').read_bytes()).hexdigest()
identity.write_text(json.dumps(proof,indent=2))
if proof['before_sha256']!=proof['after_sha256']:
    raise SystemExit('Model bytes changed during evaluation. Actual runs were preserved.')
def version(command):
    return subprocess.check_output(command,text=True).strip().splitlines()[0]
packages=tomllib.loads(Path('Cargo.lock').read_text())['package']
model=json.loads(Path('artifacts/model.json').read_text())
cpu=''
if Path('/proc/cpuinfo').is_file():
    for line in Path('/proc/cpuinfo').read_text().splitlines():
        if line.startswith('model name'):
            cpu=line.partition(':')[2].strip();break
environment={'recorded_at':datetime.now().astimezone().isoformat(),'platform':platform.platform(),
    'cpu_model':cpu,'rustc':version(['rustc','--version']),'cargo':version(['cargo','--version']),
    'cmake':version(['cmake','--version']),'gcc':version(['gcc','--version']),
    'uv':version(['uv','--version']),'summary_python':sys.version.split()[0],
    'training_plot_python_requested':'3.12 via uv','training_environment':model['environment'],
    'solver_crates':{p['name']:p['version'] for p in packages if p['name'] in ['highs','highs-sys']},
    'solver_options':{'presolve':'off','threads':1,'mip_rel_gap':0.0,'mip_abs_gap':0.0},
    'model_sha256':proof['after_sha256']}
Path('artifacts/part4_environment.json').write_text(json.dumps(environment,indent=2))
print('Model SHA-256 unchanged:',proof['after_sha256'])
PYMODELANDENV
printf '\n[5/6] Plotting CPU time breakdown, auditing cached choices and collecting actual results...\n'
uv run --no-project --python 3.12 --with matplotlib==3.10.7 \
    python plot_eval.py --test artifacts/eval_test/run.json --transfer artifacts/eval_transfer/run.json \
    --model artifacts/model.json --model-provenance artifacts/part4_model.json \
    --out artifacts/eval.png --require-course-settings \
    2>&1 | tee artifacts/eval_transfer/comparison.txt
python3 summarize_results.py
printf '\n[6/6] Committing the final source and measured results sheet locally...\n'
files=(src/main.rs src/bnb.rs plot_eval.py summarize_results.py branchopt.design.toml README.md DEEPSEEK_REVIEW.md RESULTS.md run_part4.sh)
git add -- "${files[@]}"
if git diff --cached --quiet -- "${files[@]}"; then
    printf 'Part 4 source and results sheet are already committed.\n'
else
    git commit --only -m "week12: evaluate cached neural branching and size transfer" -- "${files[@]}"
fi
printf '\nPART 4 COMPLETE\nFigure: %s/artifacts/eval.png\n' "$PWD"
printf 'Measured results and presentation notes: %s/RESULTS.md\n' "$PWD"
printf 'Raw evidence: artifacts/eval_test/ and artifacts/eval_transfer/\n'
printf 'All four figures and model remain in artifacts/ on your computer.\n'
printf 'Source was committed locally. The remaining publication command is: git push\n'
git status --short --branch
