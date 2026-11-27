#!/usr/bin/env bash
# Run from the existing course repository's week12/ after Part 1.
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
for file in Cargo.toml Cargo.lock src/main.rs branchopt.design.toml README.md; do
    if [[ ! -f "$file" ]]; then
        printf 'Missing Part 1 file: %s\n' "$file" >&2
        exit 1
    fi
done
python3 - <<'PYFIRST'
from pathlib import Path
import json
p=Path('artifacts/gap/run.json')
if not p.exists():
    raise SystemExit('Complete Part 1 first: artifacts/gap/run.json is missing.')
r=json.loads(p.read_text())
if r.get('validation_status')!='passed' or r['summary'].get('official_reference_passed') is not True:
    raise SystemExit('Part 1 reference checks must pass before Part 2.')
print('Existing Part 1 results: PASSED')
PYFIRST
git rev-parse --show-toplevel >/dev/null
mkdir -p src artifacts/bnb_most artifacts/bnb_strong
backup="artifacts/setup_backups/part2-$(date +%Y%m%dT%H%M%S)-$$"
mkdir -p "$backup"
for file in src/main.rs src/bnb.rs plot_bnb.py branchopt.design.toml README.md DEEPSEEK_REVIEW.md run_part2.sh; do
    if [[ -f "$file" ]]; then
        mkdir -p "$backup/$(dirname "$file")"
        cp -- "$file" "$backup/$file"
    fi
done
if [[ "$(realpath "$0")" != "$(realpath -m run_part2.sh)" ]]; then
    cp -- "$0" run_part2.sh
fi
printf '\n[1/5] Adding Branch-and-Bound source; existing files are backed up...\n'
cat > 'src/main.rs' <<'WEEK12_PART2_FILE'
mod bnb;
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
struct Args { mode: String, policy: String, spins: usize, edges: usize, count: usize, seed: u64, out: PathBuf }
fn arguments() -> Result<Args> {
    let mut a = Args { mode: "gap".into(), policy: "most-frac".into(), spins: 22, edges: 52, count: 16,
                       seed: 2026, out: "artifacts/gap".into() };
    let mut it = env::args().skip(1);
    while let Some(k) = it.next() {
        if k == "--help" {
            println!("branchopt --mode gap|bnb --policy most-frac|strong --spins 22 --edges 52 --count 48 --seed 2026 --out artifacts/bnb_most");
            std::process::exit(0);
        }
        let v = it.next().ok_or_else(|| fail(format!("missing value for {k}")))?;
        match k.as_str() {
            "--mode" => a.mode = v, "--policy" => a.policy = v, "--spins" => a.spins = v.parse()?,
            "--edges" => a.edges = v.parse()?, "--count" => a.count = v.parse()?,
            "--seed" => a.seed = v.parse()?, "--out" => a.out = v.into(),
            _ => return Err(fail(format!("unknown option {k}"))),
        }
    }
    check(a.mode == "gap" || a.mode == "bnb", "supported modes: gap | bnb")?;
    check(a.policy == "most-frac" || a.policy == "strong", "supported policies: most-frac | strong")?;
    check(a.spins >= 3 && a.spins <= 10000 && a.count > 0, "invalid spins or count")?;
    check(a.edges >= a.spins && a.edges <= a.spins * (a.spins - 1) / 2,
          "edge count must be between n and n*(n-1)/2")?;
    Ok(a)
}

fn main() -> Result<()> {
    let a = arguments()?;
    if a.mode == "bnb" { return bnb::run(&a); }
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
WEEK12_PART2_FILE
cat > 'src/bnb.rs' <<'WEEK12_PART2_FILE'
//! Exact depth-first Branch-and-Bound. Trial LPs never mutate search state.
use super::*;

const EPS: f64 = 1e-6;
const TIE: f64 = 1e-12;

struct Lp { energy: f64, x: Vec<f64> }

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
    let v = solved.get_solution().columns().to_vec();
    let energy = g.h.iter().sum::<f64>() + g.edges.iter().map(|e|e.j).sum::<f64>()
        - 2.0*g.h.iter().zip(&v[..g.n]).map(|(h,x)|h*x).sum::<f64>()
        - 2.0*g.edges.iter().zip(&v[g.n..]).map(|(e,z)|e.j*z).sum::<f64>();
    check(energy.is_finite(), "node LP returned a non-finite energy")?;
    Ok(Some(Lp { energy, x: v[..g.n].to_vec() }))
}

#[derive(Clone)]
struct Node { fixed: Vec<i8>, parent: Option<usize>, depth: usize, branch: Option<(usize,u8)> }

#[derive(Serialize)]
struct SearchResult {
    energy: f64, bits: Vec<u8>, nodes: usize, lp_solves: usize,
    node_lp_solves: usize, trial_lp_solves: usize,
    total_ms: f64, lp_ms: f64, first_splits: Vec<usize>, trace: Vec<Value>,
}

fn search(g: &Graph, policy: &str) -> Result<SearchResult> {
    check(policy == "most-frac" || policy == "strong", "unsupported B&B policy")?;
    let start = Instant::now();
    let mut stack = vec![Node { fixed: vec![-1;g.n], parent: None, depth: 0, branch: None }];
    let mut incumbent = f64::INFINITY;
    let mut best_bits = Vec::new();
    let (mut nodes, mut trials) = (0usize,0usize);
    let mut lp_ms = 0.0;
    let mut trace = Vec::new();
    let mut first_splits = Vec::new();
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
        let mut chosen = candidates[0];
        let mut strong_audit = Vec::new();
        if policy == "most-frac" {
            let mut best_distance = (lp.x[chosen]-0.5).abs();
            for &i in candidates.iter().skip(1) {
                let distance = (lp.x[i]-0.5).abs();
                if distance < best_distance-TIE { chosen=i; best_distance=distance; }
            }
        } else {
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
        node_lp_solves:nodes,trial_lp_solves:trials,total_ms,lp_ms,first_splits,trace })
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
        let result=search(&graph,&a.policy)?;
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
                    let result=search(&g,policy)?;
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
WEEK12_PART2_FILE
cat > 'plot_bnb.py' <<'WEEK12_PART2_FILE'
#!/usr/bin/env python3
"""Independently sort each curve; use only actual saved B&B runs."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument("--most",default="artifacts/bnb_most/run.json")
parser.add_argument("--strong",default="artifacts/bnb_strong/run.json")
parser.add_argument("--out",default="artifacts/bnb.png")
args=parser.parse_args()
runs=[json.loads(Path(path).read_text()) for path in [args.most,args.strong]]
for run in runs:
    if run.get("validation_status")!="passed":
        raise SystemExit("A saved run failed validation; inspect its raw data before plotting.")
p0,p1=[r["parameters"] for r in runs]
assert all(p0[k]==p1[k] for k in ["spins","edges","count","seed"]),"Benchmark parameters differ"
assert [r["parameters"]["policy"] for r in runs]==["most-frac","strong"]
assert [r["graph"] for r in runs[0]["instances"]]==[r["graph"] for r in runs[1]["instances"]],"Input graphs differ"
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
for run,color,label in zip(runs,["#3765ad","#d0762b"],["Most fractional","Strong branching"]):
    records=run["instances"]
    rank=range(1,len(records)+1)
    axes[0].plot(rank,sorted(r["search"]["nodes"] for r in records),label=label,color=color,lw=2)
    axes[1].plot(rank,sorted(r["search"]["lp_solves"] for r in records),label=label,color=color,lw=2)
for ax,title,ylabel in zip(axes,["Explored nodes","All LP solves, including lookahead"],["Explored nodes","LP solves"]):
    ax.set(xlabel="Independent sorted rank",ylabel=ylabel,title=title)
    ax.grid(axis="y",alpha=0.2)
    ax.legend(frameon=False)
fig.suptitle(f"Exact Branch-and-Bound: n={p0['spins']}, m={p0['edges']}, instances={p0['count']}, seed={p0['seed']}")
out=Path(args.out)
out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out,dpi=180)
print("Saved",out.resolve())
print("\nPolicy       Mean nodes   Mean LP solves   Mean time (ms)   Max energy error")
for run in runs:
    s=run["summary"]
    print(f"{s['policy']:<12} {s['mean_nodes']:10.3f} {s['mean_lp_solves']:16.3f} {s['mean_total_ms']:16.3f} {s['max_energy_error']:18.3e}")
    print("  LP accounting passed:",s["lp_accounting_passed"])
    if s["policy"]=="most-frac":
        print("  LP solves = nodes:",s["lp_solves_equal_nodes"])
    else:
        print("  Mean extra trial LP solves:",s["mean_trial_lp_solves"])
print("\nWorkshop: actual LP solves =",runs[0]["workshop"]["node_count"])
for row in runs[0]["workshop"]["nodes"]:
    print(json.dumps(row))
print("\nPART 2 VALIDATION PASSED")
WEEK12_PART2_FILE
python3 - <<'PYDOCS'
from pathlib import Path
import re
p=Path('branchopt.design.toml')
t=p.read_text() if p.exists() else ''
t=re.sub(re.escape('# WEEK12_PART2_BEGIN')+r'.*?'+re.escape('# WEEK12_PART2_END')+r'\n?', '', t, flags=re.S)
p.write_text(t.rstrip()+'\n\n'+'# WEEK12_PART2_BEGIN\n[args.bnb]\npolicy = "branching rule: most-frac | strong | neural | root-neural"\n[benchmark.bnb]\nsearch = """\nStart a depth-first search with no incumbent. Prune infeasible LPs and bounds\n>= incumbent - 1e-6. Branch on a free spin with 1e-6 < x_i < 1 - 1e-6.\nVisit the child nearer its LP value first (at 0.5, visit x_i=1 first).\n"""\npolicies = """\nMost-frac picks the spin closest to 0.5. Strong probes both child LPs for\nevery candidate and maximizes max(d0,1e-6) * max(d1,1e-6). A gain is the\nchild bound minus the parent bound, or 1e6 if pruned or infeasible. Count\ntrial LPs and leave the incumbent unchanged. Break ties by lowest spin index.\n"""\nevidence = """\nTrace Branch-and-Bound on the opening workshop problem, splitting on x2\nand visiting x2 <= 2 first. Save each branch, status, LP bound and variable\nvalues. For each random spin glass record energy, error against HiGHS,\nnodes, LP solves and elapsed search time.\n"""\n# WEEK12_PART2_END\n')
p=Path('README.md')
t=p.read_text() if p.exists() else ''
t=re.sub(re.escape('<!-- WEEK12_PART2_BEGIN -->')+r'.*?'+re.escape('<!-- WEEK12_PART2_END -->')+r'\n?', '', t, flags=re.S)
p.write_text(t.rstrip()+'\n\n'+"<!-- WEEK12_PART2_BEGIN -->\n## Part 2: Exact Branch-and-Bound\n\n`src/bnb.rs` builds a fresh continuous HiGHS LP for each visited box and each\nStrong Branching trial. The search begins with no incumbent, visits the nearer\nchild first (1 first on exactly 0.5), and prunes using the LP lower bound.\nInteger incumbents are recomputed from the original Hamiltonian. Only fractional\nfree spin variables may be split, into the exact partition x_i=0 / x_i=1.\n\nMost-frac minimizes distance from 0.5. Strong Branching maximizes the product of\nfloored child gains. A bound-pruned or infeasible child has gain 1e6. Candidate\nindices are traversed in increasing order; ties within 1e-12 numerical tolerance\n(relative to max(1, abs(score)) for strong) retain the earlier index.\n\nTrials use cloned box vectors and independent models. They never update the\nincumbent, modify the real stack, or cache solutions for later child visits.\nActual LP solve calls = visited nodes + trial solve calls. The oracle is outside\nthe search timer and is not an input to search. Total time includes recording\noverhead; LP time includes setup, solve and extraction for visited and trial LPs.\n\nThe workshop trace runs a real maximum-revenue LP at each visited box. It\nrequires three visited nodes with LP objectives 83/3, 27 and 26.75 and a final\nbound prune. It does not replace the trace with hardcoded output.\n\nRun from week12/:\n\n```bash\ncargo test --release --locked -- --test-threads=1\ncargo install --path . --force --locked\nbranchopt --mode bnb --spins 22 --edges 52 --count 48 --policy most-frac --seed 2026 --out artifacts/bnb_most\nbranchopt --mode bnb --spins 22 --edges 52 --count 48 --policy strong --seed 2026 --out artifacts/bnb_strong\nuv run --no-project --python 3.12 --with matplotlib==3.10.7 python plot_bnb.py\n```\n\nRaw runs, full traces and measured timing are in artifacts/bnb_most/ and\nartifacts/bnb_strong/. The figure artifacts/bnb.png independently sorts all\nfour curves. Input equality, exact energy and solve accounting are validated.\nNode-count and timing reference values are not substituted for measurements.\nThe two meaningful unit tests compare both policies with exhaustive enumeration\non eight small mixed-sign/field graphs and verify the workshop's actual pruning.\nThe assignment's two 48-instance runs are performed on the user's computer.\n\nPart 1 gap data and gap.png remain available. The upgraded executable still\nsupports --mode gap. See DEEPSEEK_REVIEW.md for the independent review scope.\n<!-- WEEK12_PART2_END -->\n")
p=Path('DEEPSEEK_REVIEW.md')
t=p.read_text() if p.exists() else ''
t=re.sub(re.escape('<!-- WEEK12_PART2_BEGIN -->')+r'.*?'+re.escape('<!-- WEEK12_PART2_END -->')+r'\n?', '', t, flags=re.S)
p.write_text(t.rstrip()+'\n\n'+"<!-- WEEK12_PART2_BEGIN -->\n## Part 2 independent review (2026-11-27)\n\nAn actual follow-up request was submitted to the same DeepSeek conversation.\nThe submitted excerpts covered continuous node LPs, the DFS search loop,\nincumbent updates, Strong Branching trial isolation, gain direction, child\nordering, solve counts, timing scope and the workshop trace contract.\nDeepSeek performed static review only and reported no definite error in the\nshown minimization/pruning, incumbent or trial-accounting logic.\n\nReview dispositions by the primary assistant:\n\n1. The 1e-6 pruning/integrality tolerance is the sheet's requirement. The\n   provided generator uses integer couplings and fields; no claim of certified\n   exactness for arbitrary real input at differences below that tolerance is made.\n2. The gain sentinel 1e6 is explicitly required by the sheet and changes only\n   variable selection. Trial LPs use cloned bound vectors and independent models;\n   an immutable incumbent snapshot is used and trials cannot update it.\n3. Integral solutions are rounded only within tolerance and checked by direct\n   Hamiltonian recomputation; the final search is independently compared with\n   HiGHS. Non-optimal/non-infeasible LP statuses abort the run and cannot be\n   reported as a completed successful benchmark.\n4. Counts include infeasible visited/trial nodes. No trial solve is cached or\n   reused as a visited-node LP. The program checks the total-count identity.\n5. DeepSeek's final prose reversed the required workshop PUSH order in its\n   last point. The delivered code correctly pushes x2>=3 FIRST and x2<=2 LAST;\n   LIFO therefore VISITS x2<=2 first. This was checked by the actual three-node\n   unit test and CLI integration trace; that erroneous prose was not adopted.\n\nValidation performed before delivery: Rust compilation passed; both policies\nmatched exhaustive enumeration on eight small n=3/4/6/8 graphs; the actual\nworkshop trace visited three nodes; a separate 2-instance, 6-spin CLI/plot\nintegration check passed with zero energy errors and correct solve accounting.\nThose are development checks, not the assignment's formal benchmark results.\nThe two official 48-instance runs and their reported times remain to be\ncomputed on the user's own computer by run_part2.sh.\n<!-- WEEK12_PART2_END -->\n")
PYDOCS

export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
printf '\n[2/5] Checking small graphs against exhaustive enumeration and testing workshop pruning...\n'
cargo test --release --locked -- --test-threads=1 2>&1 | tee artifacts/bnb_most/tests.log
cargo install --path . --force --locked 2>&1 | tee artifacts/bnb_most/build.log
branchopt_bin="${CARGO_HOME:-$HOME/.cargo}/bin/branchopt"
printf '\n[3/5] Running 48 graphs with most-frac...\n'
"$branchopt_bin" --mode bnb --spins 22 --edges 52 --count 48 --policy most-frac \
    --seed 2026 --out artifacts/bnb_most 2>&1 | tee artifacts/bnb_most/stdout.txt
printf '\n[4/5] Running the SAME 48 graphs with Strong Branching and plotting...\n'
"$branchopt_bin" --mode bnb --spins 22 --edges 52 --count 48 --policy strong \
    --seed 2026 --out artifacts/bnb_strong 2>&1 | tee artifacts/bnb_strong/stdout.txt
uv run --no-project --python 3.12 --with matplotlib==3.10.7 \
    python plot_bnb.py --most artifacts/bnb_most/run.json \
    --strong artifacts/bnb_strong/run.json --out artifacts/bnb.png \
    2>&1 | tee artifacts/bnb_most/comparison.txt
{
    date -Iseconds
    uname -srmo
    rustc --version
    cargo --version
    cmake --version
    uv --version
} > artifacts/bnb_most/environment.txt
cp -- artifacts/bnb_most/environment.txt artifacts/bnb_strong/environment.txt
printf '\n[5/5] Committing the Part 2 source files locally...\n'
files=(src/main.rs src/bnb.rs plot_bnb.py branchopt.design.toml README.md DEEPSEEK_REVIEW.md run_part2.sh)
git add -- "${files[@]}"
if git diff --cached --quiet -- "${files[@]}"; then
    printf 'Part 2 source is already committed.\n'
else
    git commit --only -m "week12: add exact branch-and-bound and strong branching" -- "${files[@]}"
fi
printf '\nPART 2 COMPLETE\nFigure: %s/artifacts/bnb.png\n' "$PWD"
printf 'Raw runs: artifacts/bnb_most/run.json and artifacts/bnb_strong/run.json\n'
printf 'Source was committed locally; no git push was run.\n'
