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
