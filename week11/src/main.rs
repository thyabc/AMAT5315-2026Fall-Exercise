use clap::Parser;
use omeco::{contraction_complexity, optimize_code, EinCode, GreedyMethod, NestedEinsum, TreeSA};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::io::{self, ErrorKind};
use std::{error::Error, fs, path::PathBuf, time::Instant};

#[derive(Parser, Debug)]
#[command(name = "tnsample", version)]
struct Cli {
    #[arg(long, default_value = "gadgets")]
    mode: String,
    #[arg(long, default_value_t = 2026)]
    seed: u64,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    input: Option<PathBuf>,
    #[arg(long, default_value = "treesa")]
    order: String,
    #[arg(long, default_value_t = 2.0)]
    beta: f64,
    #[arg(long, default_value_t = 2000)]
    samples: usize,
    #[arg(long)]
    l: Option<usize>,
    #[arg(long = "p-af")]
    p_af: Option<f64>,
    #[arg(long)]
    n: Option<usize>,
}

#[derive(Serialize)]
struct GateRow {
    gate: String,
    row: String,
    inputs: Vec<u8>,
    output: u8,
    helper_w: Option<u8>,
    energy: i32,
    valid: bool,
}

#[derive(Serialize)]
struct GateSummary {
    gate: String,
    floor: i32,
    valid_rows: usize,
    total_rows: usize,
}

#[derive(Serialize)]
struct CircuitRow {
    a: u8,
    b: u8,
    t: u8,
    out: u8,
    energy: i32,
    valid: bool,
}

#[derive(Serialize)]
struct Evidence {
    schema_version: u32,
    mode: String,
    seed: u64,
    spin_values: Vec<i32>,
    gates: Vec<GateRow>,
    gate_summaries: Vec<GateSummary>,
    circuit_states: Vec<CircuitRow>,
    circuit_ground_energy: i32,
    circuit_ground_states: usize,
}

fn spin(x: u8) -> i32 {
    2 * x as i32 - 1
}

fn and_energy(a: u8, b: u8, z: u8) -> i32 {
    let (a, b, z) = (spin(a), spin(b), spin(z));
    a * b - 2 * a * z - 2 * b * z - a - b + 2 * z
}

fn or_energy(a: u8, b: u8, z: u8) -> i32 {
    let (a, b, z) = (spin(a), spin(b), spin(z));
    a * b - 2 * a * z - 2 * b * z + a + b - 2 * z
}

fn not_energy(a: u8, z: u8) -> i32 {
    spin(a) * spin(z)
}

fn copy_energy(a: u8, z: u8) -> i32 {
    -spin(a) * spin(z)
}

fn set0_energy(z: u8) -> i32 {
    spin(z)
}

fn set1_energy(z: u8) -> i32 {
    -spin(z)
}

fn xor_energy(a: u8, b: u8, z: u8, w: u8) -> i32 {
    let (a, b, z, w) = (spin(a), spin(b), spin(z), spin(w));

    a * b - a * z - 2 * a * w - b * z - 2 * b * w + 2 * z * w - a - b + z + 2 * w
}

fn run_gadgets(cli: &Cli) -> Result<(), Box<dyn Error>> {
    let start = Instant::now();

    fs::create_dir_all(&cli.out)?;

    let mut gates: Vec<GateRow> = Vec::new();

    // AND
    for a in 0..=1 {
        for b in 0..=1 {
            for z in 0..=1 {
                gates.push(GateRow {
                    gate: "AND".into(),
                    row: format!("{a}{b}{z}"),
                    inputs: vec![a, b],
                    output: z,
                    helper_w: None,
                    energy: and_energy(a, b, z),
                    valid: z == (a & b),
                });
            }
        }
    }

    // OR
    for a in 0..=1 {
        for b in 0..=1 {
            for z in 0..=1 {
                gates.push(GateRow {
                    gate: "OR".into(),
                    row: format!("{a}{b}{z}"),
                    inputs: vec![a, b],
                    output: z,
                    helper_w: None,
                    energy: or_energy(a, b, z),
                    valid: z == (a | b),
                });
            }
        }
    }

    // NOT and COPY
    for a in 0..=1 {
        for z in 0..=1 {
            gates.push(GateRow {
                gate: "NOT".into(),
                row: format!("{a}{z}"),
                inputs: vec![a],
                output: z,
                helper_w: None,
                energy: not_energy(a, z),
                valid: z == (1 - a),
            });

            gates.push(GateRow {
                gate: "COPY".into(),
                row: format!("{a}{z}"),
                inputs: vec![a],
                output: z,
                helper_w: None,
                energy: copy_energy(a, z),
                valid: z == a,
            });
        }
    }

    // SET0 and SET1
    for z in 0..=1 {
        gates.push(GateRow {
            gate: "SET0".into(),
            row: format!("{z}"),
            inputs: vec![],
            output: z,
            helper_w: None,
            energy: set0_energy(z),
            valid: z == 0,
        });

        gates.push(GateRow {
            gate: "SET1".into(),
            row: format!("{z}"),
            inputs: vec![],
            output: z,
            helper_w: None,
            energy: set1_energy(z),
            valid: z == 1,
        });
    }

    // XOR: minimize over the helper bit w
    for a in 0..=1 {
        for b in 0..=1 {
            for z in 0..=1 {
                let mut best_energy = i32::MAX;
                let mut best_w = 0;

                for w in 0..=1 {
                    let e = xor_energy(a, b, z, w);

                    if e < best_energy {
                        best_energy = e;
                        best_w = w;
                    }
                }

                gates.push(GateRow {
                    gate: "XOR".into(),
                    row: format!("{a}{b}{z}"),
                    inputs: vec![a, b],
                    output: z,
                    helper_w: Some(best_w),
                    energy: best_energy,
                    valid: z == (a ^ b),
                });
            }
        }
    }

    // Summarize each gate
    let mut summaries = Vec::new();

    for name in ["AND", "OR", "NOT", "COPY", "SET0", "SET1", "XOR"] {
        let rows: Vec<&GateRow> = gates.iter().filter(|r| r.gate == name).collect();

        let floor = rows.iter().map(|r| r.energy).min().unwrap();

        let valid_rows = rows.iter().filter(|r| r.valid).count();

        summaries.push(GateSummary {
            gate: name.to_string(),
            floor,
            valid_rows,
            total_rows: rows.len(),
        });
    }

    // Circuit: AND(a,b,t) + NOT(t,out) + SET1(out)
    let mut circuit_states = Vec::new();

    for a in 0..=1 {
        for b in 0..=1 {
            for t in 0..=1 {
                for out in 0..=1 {
                    let energy = and_energy(a, b, t) + not_energy(t, out) + set1_energy(out);

                    let valid = t == (a & b) && out == (1 - t) && out == 1;

                    circuit_states.push(CircuitRow {
                        a,
                        b,
                        t,
                        out,
                        energy,
                        valid,
                    });
                }
            }
        }
    }

    let circuit_ground_energy = circuit_states.iter().map(|r| r.energy).min().unwrap();

    let circuit_ground_states = circuit_states
        .iter()
        .filter(|r| r.energy == circuit_ground_energy)
        .count();

    let evidence = Evidence {
        schema_version: 1,
        mode: cli.mode.clone(),
        seed: cli.seed,
        spin_values: vec![-1, 1],
        gates,
        gate_summaries: summaries,
        circuit_states,
        circuit_ground_energy,
        circuit_ground_states,
    };

    let json = serde_json::to_string_pretty(&evidence)?;

    fs::write(cli.out.join("run.json"), json)?;

    let runtime_ms = start.elapsed().as_millis();

    // Required TSV output
    println!("mode\tmetric\tvalue\truntime_ms");

    for s in &evidence.gate_summaries {
        println!("gadgets\t{}_floor\t{}\t{}", s.gate, s.floor, runtime_ms);
        println!(
            "gadgets\t{}_valid_rows\t{}\t{}",
            s.gate, s.valid_rows, runtime_ms
        );
        println!(
            "gadgets\t{}_total_rows\t{}\t{}",
            s.gate, s.total_rows, runtime_ms
        );
    }

    for (i, r) in evidence.gates.iter().enumerate() {
        println!(
            "gadgets\t{}_row_{}_energy\t{}\t{}",
            r.gate, i, r.energy, runtime_ms
        );
    }

    println!(
        "gadgets\tcircuit_ground_energy\t{}\t{}",
        evidence.circuit_ground_energy, runtime_ms
    );

    println!(
        "gadgets\tcircuit_ground_states\t{}\t{}",
        evidence.circuit_ground_states, runtime_ms
    );

    println!("gadgets\tcircuit_total_states\t16\t{}", runtime_ms);

    println!("\nSaved evidence: {}", cli.out.join("run.json").display());

    Ok(())
}

// -----------------------------------------------------------------------------
// Part 3: exact spin-glass contraction, ground-state counting and sampling.
// The omeco crate plans contraction trees; the numerical contractions below
// execute those trees in dense (Boltzmann) and tropical (energy,count) algebra.
// -----------------------------------------------------------------------------

fn ioerr(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(ErrorKind::Other, message.into()))
}

#[derive(Clone)]
struct SpinGlass {
    fields: Vec<i32>,
    edges: Vec<(usize, usize, i32)>,
    // pred bundles use +sum K_ij s_i s_j + sum h_i s_i.
    // The open-lattice magnet uses -sum J_ij s_i s_j.
    bundle_convention: bool,
}

impl SpinGlass {
    fn num_spins(&self) -> usize {
        self.fields.len()
    }

    fn energy(&self, bits: &[u8]) -> i32 {
        let mut e = 0i32;
        for &(i, j, k) in &self.edges {
            let si = 2 * bits[i] as i32 - 1;
            let sj = 2 * bits[j] as i32 - 1;
            e += if self.bundle_convention {
                k * si * sj
            } else {
                -k * si * sj
            };
        }
        for (i, &h) in self.fields.iter().enumerate() {
            if self.bundle_convention {
                e += h * (2 * bits[i] as i32 - 1);
            }
        }
        e
    }
}

fn load_spin_glass(path: &PathBuf) -> Result<SpinGlass, Box<dyn Error>> {
    let root: Value = serde_json::from_slice(&fs::read(path)?)?;
    let data = root
        .get("target")
        .and_then(|v| v.get("data"))
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "bundle missing target.data"))?;
    let fields: Vec<i32> = serde_json::from_value(data["fields"].clone())?;
    let couplings: Vec<i32> = serde_json::from_value(data["couplings"].clone())?;
    let edges: Vec<[usize; 2]> = serde_json::from_value(data["graph"]["edges"].clone())?;
    if fields.is_empty() || edges.len() != couplings.len() {
        return Err(ioerr(
            "invalid SpinGlass bundle: fields/edges/couplings length mismatch",
        ));
    }
    let mut weighted_edges = Vec::with_capacity(edges.len());
    for (edge, k) in edges.into_iter().zip(couplings) {
        if edge[0] >= fields.len() || edge[1] >= fields.len() {
            return Err(ioerr("SpinGlass edge endpoint is outside fields array"));
        }
        weighted_edges.push((edge[0], edge[1], k));
    }
    Ok(SpinGlass {
        fields,
        edges: weighted_edges,
        bundle_convention: true,
    })
}

struct Rng64 {
    state: u64,
}
impl Rng64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9e3779b97f4a7c15 } else { seed },
        }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}

fn make_lattice(l: usize, p_af: f64, seed: u64) -> Result<SpinGlass, Box<dyn Error>> {
    if l < 2 {
        return Err(ioerr("lattice side length --l must be at least 2"));
    }
    if !(0.0..=1.0).contains(&p_af) {
        return Err(ioerr("--p-af must be between 0 and 1"));
    }
    let n = l
        .checked_mul(l)
        .ok_or_else(|| ioerr("lattice size overflow"))?;
    let mut rng = Rng64::new(seed);
    let mut edges = Vec::new();
    for r in 0..l {
        for c in 0..l {
            let i = r * l + c;
            if c + 1 < l {
                let j = i + 1;
                let sign = if rng.next_f64() < p_af { -1 } else { 1 };
                edges.push((i, j, sign));
            }
            if r + 1 < l {
                let j = i + l;
                let sign = if rng.next_f64() < p_af { -1 } else { 1 };
                edges.push((i, j, sign));
            }
        }
    }
    Ok(SpinGlass {
        fields: vec![0; n],
        edges,
        bundle_convention: false,
    })
}

#[derive(Clone)]
struct DenseTensor {
    labels: Vec<usize>,
    data: Vec<f64>,
}
#[derive(Clone, Copy)]
struct TropEntry {
    energy: i32,
    count: u128,
}
#[derive(Clone)]
struct TropTensor {
    labels: Vec<usize>,
    data: Vec<TropEntry>,
}

fn spin_network(sg: &SpinGlass, beta: f64) -> (EinCode<usize>, Vec<DenseTensor>, Vec<TropTensor>) {
    let mut labels = Vec::new();
    let mut dense = Vec::new();
    let mut trop = Vec::new();
    for &(i, j, k) in &sg.edges {
        let ix = vec![i, j];
        let mut weights = Vec::with_capacity(4);
        let mut entries = Vec::with_capacity(4);
        for bi in 0..=1u8 {
            for bj in 0..=1u8 {
                let si = 2 * bi as i32 - 1;
                let sj = 2 * bj as i32 - 1;
                let e = if sg.bundle_convention {
                    k * si * sj
                } else {
                    -k * si * sj
                };
                weights.push((-beta * e as f64).exp());
                entries.push(TropEntry {
                    energy: e,
                    count: 1,
                });
            }
        }
        labels.push(ix.clone());
        dense.push(DenseTensor {
            labels: ix.clone(),
            data: weights,
        });
        trop.push(TropTensor {
            labels: ix,
            data: entries,
        });
    }
    for (i, &h) in sg.fields.iter().enumerate() {
        if h == 0 {
            continue;
        }
        let ix = vec![i];
        let mut weights = Vec::with_capacity(2);
        let mut entries = Vec::with_capacity(2);
        for bit in 0..=1u8 {
            let s = 2 * bit as i32 - 1;
            let e = if sg.bundle_convention { h * s } else { 0 };
            weights.push((-beta * e as f64).exp());
            entries.push(TropEntry {
                energy: e,
                count: 1,
            });
        }
        labels.push(ix.clone());
        dense.push(DenseTensor {
            labels: ix.clone(),
            data: weights,
        });
        trop.push(TropTensor {
            labels: ix,
            data: entries,
        });
    }
    (EinCode::new(labels, Vec::new()), dense, trop)
}

enum PlanNode {
    Leaf {
        index: usize,
        labels: Vec<usize>,
    },
    Node {
        left: Box<PlanNode>,
        right: Box<PlanNode>,
        labels: Vec<usize>,
    },
}
impl PlanNode {
    fn labels(&self) -> &[usize] {
        match self {
            PlanNode::Leaf { labels, .. } => labels,
            PlanNode::Node { labels, .. } => labels,
        }
    }
}

fn convert_plan(
    tree: &NestedEinsum<usize>,
    input_labels: &[Vec<usize>],
) -> Result<PlanNode, Box<dyn Error>> {
    match tree {
        NestedEinsum::Leaf { tensor_index } => {
            let labels = input_labels
                .get(*tensor_index)
                .ok_or_else(|| {
                    io::Error::new(ErrorKind::InvalidData, "omeco tree references missing leaf")
                })?
                .clone();
            Ok(PlanNode::Leaf {
                index: *tensor_index,
                labels,
            })
        }
        NestedEinsum::Node { args, eins } => {
            if args.len() != 2 {
                return Err(ioerr("omeco returned a non-binary contraction node"));
            }
            Ok(PlanNode::Node {
                left: Box::new(convert_plan(&args[0], input_labels)?),
                right: Box::new(convert_plan(&args[1], input_labels)?),
                labels: eins.iy.clone(),
            })
        }
    }
}

fn make_plan(
    code: &EinCode<usize>,
    order: &str,
) -> Result<(PlanNode, f64, f64, f64), Box<dyn Error>> {
    let sizes: HashMap<usize, usize> = code.unique_labels().into_iter().map(|x| (x, 2)).collect();
    let nested = match order {
        "greedy" => optimize_code(code, &sizes, &GreedyMethod::default()),
        "treesa" => optimize_code(code, &sizes, &TreeSA::fast()),
        other => {
            return Err(ioerr(format!(
                "unknown --order {other}; use greedy or treesa"
            )))
        }
    }
    .ok_or_else(|| io::Error::new(ErrorKind::Other, "omeco could not find a contraction tree"))?;
    let c = contraction_complexity(&nested, &sizes, &code.ixs);
    let plan = convert_plan(&nested, &code.ixs)?;
    Ok((plan, c.tc, c.sc, c.rwc))
}

fn unique_union(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = a.to_vec();
    for &x in b {
        if !out.contains(&x) {
            out.push(x);
        }
    }
    out
}
fn positions(labels: &[usize]) -> HashMap<usize, usize> {
    labels.iter().enumerate().map(|(i, &x)| (x, i)).collect()
}
fn flat_index(labels: &[usize], pos: &HashMap<usize, usize>, mask: usize) -> usize {
    let mut idx = 0usize;
    for l in labels {
        idx = (idx << 1) | ((mask >> pos[l]) & 1);
    }
    idx
}
fn checked_states(bits: usize) -> Result<usize, Box<dyn Error>> {
    1usize
        .checked_shl(bits as u32)
        .ok_or_else(|| ioerr(format!("intermediate tensor rank {bits} is too large")))
}

fn contract_dense(
    a: &DenseTensor,
    b: &DenseTensor,
    out_labels: &[usize],
) -> Result<DenseTensor, Box<dyn Error>> {
    let union = unique_union(&a.labels, &b.labels);
    let pos = positions(&union);
    let nstates = checked_states(union.len())?;
    let out_len = checked_states(out_labels.len())?;
    let mut data = vec![0.0; out_len];
    for mask in 0..nstates {
        let ia = flat_index(&a.labels, &pos, mask);
        let ib = flat_index(&b.labels, &pos, mask);
        let io = flat_index(out_labels, &pos, mask);
        data[io] += a.data[ia] * b.data[ib];
    }
    Ok(DenseTensor {
        labels: out_labels.to_vec(),
        data,
    })
}

fn contract_trop(
    a: &TropTensor,
    b: &TropTensor,
    out_labels: &[usize],
) -> Result<TropTensor, Box<dyn Error>> {
    let union = unique_union(&a.labels, &b.labels);
    let pos = positions(&union);
    let nstates = checked_states(union.len())?;
    let out_len = checked_states(out_labels.len())?;
    let mut data = vec![
        TropEntry {
            energy: i32::MAX,
            count: 0
        };
        out_len
    ];
    for mask in 0..nstates {
        let ea = a.data[flat_index(&a.labels, &pos, mask)];
        let eb = b.data[flat_index(&b.labels, &pos, mask)];
        if ea.count == 0 || eb.count == 0 {
            continue;
        }
        let e = ea.energy + eb.energy;
        let count = ea
            .count
            .checked_mul(eb.count)
            .ok_or_else(|| ioerr("tropical state count overflow"))?;
        let slot = &mut data[flat_index(out_labels, &pos, mask)];
        if e < slot.energy {
            slot.energy = e;
            slot.count = count;
        } else if e == slot.energy {
            slot.count = slot
                .count
                .checked_add(count)
                .ok_or_else(|| ioerr("tropical degeneracy overflow"))?;
        }
    }
    Ok(TropTensor {
        labels: out_labels.to_vec(),
        data,
    })
}

fn eval_trop(plan: &PlanNode, leaves: &[TropTensor]) -> Result<TropTensor, Box<dyn Error>> {
    match plan {
        PlanNode::Leaf { index, .. } => leaves
            .get(*index)
            .cloned()
            .ok_or_else(|| ioerr("missing tropical leaf")),
        PlanNode::Node {
            left,
            right,
            labels,
        } => {
            let a = eval_trop(left, leaves)?;
            let b = eval_trop(right, leaves)?;
            contract_trop(&a, &b, labels)
        }
    }
}

struct DenseEval {
    tensor: DenseTensor,
    log_scale: f64,
    kind: DenseEvalKind,
}
enum DenseEvalKind {
    Leaf,
    Node(Box<DenseEval>, Box<DenseEval>),
}
fn eval_dense(plan: &PlanNode, leaves: &[DenseTensor]) -> Result<DenseEval, Box<dyn Error>> {
    match plan {
        PlanNode::Leaf { index, .. } => {
            let mut tensor = leaves
                .get(*index)
                .cloned()
                .ok_or_else(|| ioerr("missing dense leaf"))?;
            let maxv = tensor.data.iter().copied().fold(0.0f64, f64::max);
            if !maxv.is_finite() || maxv <= 0.0 {
                return Err(ioerr("invalid Boltzmann leaf scale"));
            }
            for x in &mut tensor.data {
                *x /= maxv;
            }
            Ok(DenseEval {
                tensor,
                log_scale: maxv.ln(),
                kind: DenseEvalKind::Leaf,
            })
        }
        PlanNode::Node {
            left,
            right,
            labels,
        } => {
            let a = Box::new(eval_dense(left, leaves)?);
            let b = Box::new(eval_dense(right, leaves)?);
            let mut tensor = contract_dense(&a.tensor, &b.tensor, labels)?;
            let maxv = tensor.data.iter().copied().fold(0.0f64, f64::max);
            if !maxv.is_finite() || maxv <= 0.0 {
                return Err(ioerr("invalid Boltzmann intermediate scale"));
            }
            for x in &mut tensor.data {
                *x /= maxv;
            }
            let log_scale = a.log_scale + b.log_scale + maxv.ln();
            Ok(DenseEval {
                tensor,
                log_scale,
                kind: DenseEvalKind::Node(a, b),
            })
        }
    }
}

fn tensor_value(t: &DenseTensor, assignment: &HashMap<usize, u8>) -> Result<f64, Box<dyn Error>> {
    let mut idx = 0usize;
    for l in &t.labels {
        let bit = *assignment.get(l).ok_or_else(|| {
            io::Error::new(ErrorKind::InvalidData, format!("missing sampled index {l}"))
        })? as usize;
        idx = idx * 2 + bit;
    }
    Ok(t.data[idx])
}
fn weighted_pick(weights: &[f64], rng: &mut Rng64) -> Result<usize, Box<dyn Error>> {
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(ioerr("cannot sample from zero total weight"));
    }
    let target = rng.next_f64() * total;
    let mut acc = 0.0;
    for (i, w) in weights.iter().enumerate() {
        acc += *w;
        if target < acc {
            return Ok(i);
        }
    }
    Ok(weights.len() - 1)
}

fn sample_dense(
    node: &DenseEval,
    fixed: &HashMap<usize, u8>,
    assignment: &mut HashMap<usize, u8>,
    rng: &mut Rng64,
) -> Result<(), Box<dyn Error>> {
    match &node.kind {
        DenseEvalKind::Leaf => {
            for &l in &node.tensor.labels {
                let v = *fixed.get(&l).ok_or_else(|| {
                    io::Error::new(
                        ErrorKind::InvalidData,
                        format!("leaf index {l} was not fixed by parent"),
                    )
                })?;
                if let Some(old) = assignment.insert(l, v) {
                    if old != v {
                        return Err(ioerr("inconsistent sampled spin assignment"));
                    }
                }
            }
        }
        DenseEvalKind::Node(left, right) => {
            let union = unique_union(&left.tensor.labels, &right.tensor.labels);
            let free: Vec<usize> = union
                .iter()
                .copied()
                .filter(|l| !node.tensor.labels.contains(l))
                .collect();
            let nstates = checked_states(free.len())?;
            let mut candidates: Vec<HashMap<usize, u8>> = Vec::new();
            let mut weights: Vec<f64> = Vec::new();
            for mask in 0..nstates {
                let mut values = fixed.clone();
                for (i, &l) in free.iter().enumerate() {
                    values.insert(l, ((mask >> i) & 1) as u8);
                }
                let w =
                    tensor_value(&left.tensor, &values)? * tensor_value(&right.tensor, &values)?;
                if w > 0.0 && w.is_finite() {
                    candidates.push(values);
                    weights.push(w);
                }
            }
            let chosen = weighted_pick(&weights, rng)?;
            let values = &candidates[chosen];
            let left_fixed: HashMap<usize, u8> = left
                .tensor
                .labels
                .iter()
                .map(|&l| (l, values[&l]))
                .collect();
            let right_fixed: HashMap<usize, u8> = right
                .tensor
                .labels
                .iter()
                .map(|&l| (l, values[&l]))
                .collect();
            sample_dense(left, &left_fixed, assignment, rng)?;
            sample_dense(right, &right_fixed, assignment, rng)?;
        }
    }
    Ok(())
}

fn plan_json(
    mode: &str,
    seed: u64,
    order: &str,
    code: &EinCode<usize>,
    tc: f64,
    sc: f64,
    rwc: f64,
    runtime_ms: u128,
) -> Value {
    json!({
        "schema_version": 1, "mode": mode, "seed": seed, "order": order,
        "variables": code.unique_labels().len(), "tensors": code.num_tensors(),
        "tc": tc, "sc": sc, "rwc": rwc, "runtime_ms": runtime_ms
    })
}
fn save_run(out: &PathBuf, value: &Value) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(out)?;
    fs::write(out.join("run.json"), serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn tsv(mode: &str, metrics: &[(&str, String)], runtime_ms: u128) {
    println!("mode\tmetric\tvalue\truntime_ms");
    for (name, value) in metrics {
        println!("{mode}\t{name}\t{value}\t{runtime_ms}");
    }
}

// -----------------------------------------------------------------------------
// Part 4: N-queens as a sparse 0/1 carry-tensor network.
// Each square bit is shared by its row, column and two diagonal carry tensors.
// All tensors contain only nonzero entries; contractions join sparse supports.
// -----------------------------------------------------------------------------

#[derive(Clone)]
struct SparseTensor {
    labels: Vec<usize>,
    // Key bits follow `labels` order (label position i is stored in bit i).
    // Every tensor is an indicator/count tensor with nonnegative integer values.
    data: BTreeMap<usize, u128>,
}

struct NQueensNetwork {
    code: EinCode<usize>,
    leaves: Vec<SparseTensor>,
    x_labels: Vec<usize>,
}

struct SparseEval {
    tensor: SparseTensor,
    kind: SparseEvalKind,
}

enum SparseEvalKind {
    Leaf,
    Node(Box<SparseEval>, Box<SparseEval>),
}

fn carry_allowed(cin: u8, x: u8, cout: u8) -> bool {
    matches!((cin, x, cout), (0, 0, 0) | (0, 1, 1) | (1, 0, 1))
}

fn add_carry_line(
    n: usize,
    coords: &[(usize, usize)],
    fixed_end: Option<u8>,
    x_labels: &[usize],
    next_label: &mut usize,
    tensor_labels: &mut Vec<Vec<usize>>,
    leaves: &mut Vec<SparseTensor>,
) -> Result<(), Box<dyn Error>> {
    if coords.is_empty() {
        return Ok(());
    }

    // One binary carry index joins each adjacent pair of squares on this line.
    let mut carries = Vec::with_capacity(coords.len().saturating_sub(1));
    for _ in 0..coords.len().saturating_sub(1) {
        carries.push(*next_label);
        *next_label += 1;
    }

    for (position, &(row, col)) in coords.iter().enumerate() {
        let mut labels = Vec::with_capacity(3);
        if position > 0 {
            labels.push(carries[position - 1]);
        }
        labels.push(x_labels[row * n + col]);
        if position + 1 < coords.len() {
            labels.push(carries[position]);
        }

        let state_count = checked_states(labels.len())?;
        let has_previous = position > 0;
        let has_next = position + 1 < coords.len();
        let x_position = if has_previous { 1 } else { 0 };
        let mut data = BTreeMap::new();

        for key in 0..state_count {
            // Every line starts with carry 0. Later squares read the previous
            // square's outgoing carry from their first local index.
            let cin = if has_previous { (key & 1) as u8 } else { 0 };
            let x = ((key >> x_position) & 1) as u8;

            let allowed = if has_next {
                let cout = ((key >> (labels.len() - 1)) & 1) as u8;
                carry_allowed(cin, x, cout)
            } else if let Some(end) = fixed_end {
                carry_allowed(cin, x, end)
            } else {
                // A diagonal may finish with carry 0 or 1: at most one queen.
                carry_allowed(cin, x, 0) || carry_allowed(cin, x, 1)
            };

            if allowed {
                data.insert(key, 1);
            }
        }

        tensor_labels.push(labels.clone());
        leaves.push(SparseTensor { labels, data });
    }
    Ok(())
}

fn build_nqueens_network(n: usize) -> Result<NQueensNetwork, Box<dyn Error>> {
    if n == 0 {
        return Err(ioerr("N-queens board size --n must be positive"));
    }
    // The assignment requires N=4..10 and mentions 11/12 as optional checks.
    // Keep a guard against accidentally constructing an impractically large graph.
    if n > 12 {
        return Err(ioerr("this N-queens implementation is limited to N <= 12"));
    }
    let square_count = n
        .checked_mul(n)
        .ok_or_else(|| ioerr("N-queens board size overflow"))?;
    let x_labels: Vec<usize> = (0..square_count).collect();
    let mut next_label = square_count;
    let mut tensor_labels = Vec::with_capacity(4 * square_count);
    let mut leaves = Vec::with_capacity(4 * square_count);

    // Rows: carry starts at 0 and must finish at 1 (exactly one queen).
    for row in 0..n {
        let coords: Vec<(usize, usize)> = (0..n).map(|col| (row, col)).collect();
        add_carry_line(
            n,
            &coords,
            Some(1),
            &x_labels,
            &mut next_label,
            &mut tensor_labels,
            &mut leaves,
        )?;
    }

    // Columns: carry starts at 0 and must finish at 1 (exactly one queen).
    for col in 0..n {
        let coords: Vec<(usize, usize)> = (0..n).map(|row| (row, col)).collect();
        add_carry_line(
            n,
            &coords,
            Some(1),
            &x_labels,
            &mut next_label,
            &mut tensor_labels,
            &mut leaves,
        )?;
    }

    // Down-right diagonals, indexed by row - column.
    let ni = isize::try_from(n).map_err(|_| ioerr("N is too large for diagonal indexing"))?;
    for difference in -(ni - 1)..=(ni - 1) {
        let mut coords = Vec::new();
        for row in 0..n {
            let col = row as isize - difference;
            if (0..ni).contains(&col) {
                coords.push((row, col as usize));
            }
        }
        add_carry_line(
            n,
            &coords,
            None,
            &x_labels,
            &mut next_label,
            &mut tensor_labels,
            &mut leaves,
        )?;
    }

    // Up-right diagonals, indexed by row + column.
    for sum in 0..=(2 * ni - 2) {
        let mut coords = Vec::new();
        for row in 0..n {
            let col = sum - row as isize;
            if (0..ni).contains(&col) {
                coords.push((row, col as usize));
            }
        }
        add_carry_line(
            n,
            &coords,
            None,
            &x_labels,
            &mut next_label,
            &mut tensor_labels,
            &mut leaves,
        )?;
    }

    if tensor_labels.len() != 4 * square_count {
        return Err(ioerr(format!(
            "internal N-queens construction error: expected {} carry tensors, built {}",
            4 * square_count,
            tensor_labels.len()
        )));
    }

    let code = EinCode::new(tensor_labels, Vec::new());
    Ok(NQueensNetwork {
        code,
        leaves,
        x_labels,
    })
}

fn project_sparse_key(
    source_positions: &HashMap<usize, usize>,
    source_key: usize,
    target_labels: &[usize],
) -> Result<usize, Box<dyn Error>> {
    let _ = checked_states(target_labels.len())?;
    let mut target_key = 0usize;
    for (target_position, label) in target_labels.iter().enumerate() {
        let source_position = *source_positions.get(label).ok_or_else(|| {
            ioerr(format!(
                "sparse projection references missing label {label}"
            ))
        })?;
        if ((source_key >> source_position) & 1) != 0 {
            target_key |= 1usize << target_position;
        }
    }
    Ok(target_key)
}

fn merge_sparse_keys(
    union_labels: &[usize],
    a_key: usize,
    a_positions: &HashMap<usize, usize>,
    b_key: usize,
    b_positions: &HashMap<usize, usize>,
) -> Result<usize, Box<dyn Error>> {
    let _ = checked_states(union_labels.len())?;
    let mut union_key = 0usize;
    for (position, label) in union_labels.iter().enumerate() {
        let bit = if let Some(source_position) = a_positions.get(label) {
            (a_key >> *source_position) & 1
        } else if let Some(source_position) = b_positions.get(label) {
            (b_key >> *source_position) & 1
        } else {
            return Err(ioerr(format!(
                "sparse union references missing label {label}"
            )));
        };
        if bit != 0 {
            union_key |= 1usize << position;
        }
    }
    Ok(union_key)
}

fn contract_sparse(
    a: &SparseTensor,
    b: &SparseTensor,
    out_labels: &[usize],
) -> Result<SparseTensor, Box<dyn Error>> {
    let union = unique_union(&a.labels, &b.labels);
    let a_positions = positions(&a.labels);
    let b_positions = positions(&b.labels);
    let common: Vec<usize> = a
        .labels
        .iter()
        .copied()
        .filter(|label| b_positions.contains_key(label))
        .collect();
    let union_positions = positions(&union);

    // Hash only the shared-index assignment. This joins nonzero entries without
    // enumerating the zero-filled part of either dense tensor.
    let mut buckets: BTreeMap<usize, Vec<(usize, u128)>> = BTreeMap::new();
    for (&b_key, &b_value) in &b.data {
        let common_key = project_sparse_key(&b_positions, b_key, &common)?;
        buckets
            .entry(common_key)
            .or_default()
            .push((b_key, b_value));
    }

    let mut output = BTreeMap::<usize, u128>::new();
    for (&a_key, &a_value) in &a.data {
        let common_key = project_sparse_key(&a_positions, a_key, &common)?;
        if let Some(matches) = buckets.get(&common_key) {
            for &(b_key, b_value) in matches {
                let union_key =
                    merge_sparse_keys(&union, a_key, &a_positions, b_key, &b_positions)?;
                let output_key = project_sparse_key(&union_positions, union_key, out_labels)?;
                let product = a_value
                    .checked_mul(b_value)
                    .ok_or_else(|| ioerr("N-queens sparse contraction count overflow"))?;
                let slot = output.entry(output_key).or_insert(0);
                *slot = slot
                    .checked_add(product)
                    .ok_or_else(|| ioerr("N-queens sparse contraction sum overflow"))?;
            }
        }
    }

    Ok(SparseTensor {
        labels: out_labels.to_vec(),
        data: output,
    })
}

fn sparse_join_candidates(
    a: &SparseTensor,
    b: &SparseTensor,
) -> Result<(Vec<usize>, Vec<(usize, u128)>), Box<dyn Error>> {
    let union = unique_union(&a.labels, &b.labels);
    let a_positions = positions(&a.labels);
    let b_positions = positions(&b.labels);
    let common: Vec<usize> = a
        .labels
        .iter()
        .copied()
        .filter(|label| b_positions.contains_key(label))
        .collect();
    let mut buckets: BTreeMap<usize, Vec<(usize, u128)>> = BTreeMap::new();
    for (&b_key, &b_value) in &b.data {
        let common_key = project_sparse_key(&b_positions, b_key, &common)?;
        buckets
            .entry(common_key)
            .or_default()
            .push((b_key, b_value));
    }

    let mut candidates = Vec::new();
    for (&a_key, &a_value) in &a.data {
        let common_key = project_sparse_key(&a_positions, a_key, &common)?;
        if let Some(matches) = buckets.get(&common_key) {
            for &(b_key, b_value) in matches {
                let union_key =
                    merge_sparse_keys(&union, a_key, &a_positions, b_key, &b_positions)?;
                let weight = a_value
                    .checked_mul(b_value)
                    .ok_or_else(|| ioerr("N-queens sampling weight overflow"))?;
                if weight > 0 {
                    candidates.push((union_key, weight));
                }
            }
        }
    }
    Ok((union, candidates))
}

fn eval_sparse(plan: &PlanNode, leaves: &[SparseTensor]) -> Result<SparseEval, Box<dyn Error>> {
    match plan {
        PlanNode::Leaf { index, .. } => {
            let tensor = leaves
                .get(*index)
                .cloned()
                .ok_or_else(|| ioerr("missing sparse tensor leaf"))?;
            Ok(SparseEval {
                tensor,
                kind: SparseEvalKind::Leaf,
            })
        }
        PlanNode::Node {
            left,
            right,
            labels,
        } => {
            let a = Box::new(eval_sparse(left, leaves)?);
            let b = Box::new(eval_sparse(right, leaves)?);
            let tensor = contract_sparse(&a.tensor, &b.tensor, labels)?;
            Ok(SparseEval {
                tensor,
                kind: SparseEvalKind::Node(a, b),
            })
        }
    }
}

fn random_u128(rng: &mut Rng64) -> u128 {
    ((rng.next_u64() as u128) << 64) | (rng.next_u64() as u128)
}

fn random_below(rng: &mut Rng64, upper: u128) -> Result<u128, Box<dyn Error>> {
    if upper == 0 {
        return Err(ioerr("cannot draw from zero total sparse weight"));
    }
    // Rejection sampling avoids modulo bias, so all integer-weighted choices
    // remain exactly proportional to their contraction multiplicities.
    let threshold = (!upper).wrapping_add(1) % upper;
    loop {
        let value = random_u128(rng);
        if value >= threshold {
            return Ok(value % upper);
        }
    }
}

fn weighted_sparse_pick(weights: &[u128], rng: &mut Rng64) -> Result<usize, Box<dyn Error>> {
    let total = weights.iter().try_fold(0u128, |sum, &weight| {
        sum.checked_add(weight)
            .ok_or_else(|| ioerr("N-queens sampling weight sum overflow"))
    })?;
    let mut draw = random_below(rng, total)?;
    for (index, &weight) in weights.iter().enumerate() {
        if draw < weight {
            return Ok(index);
        }
        draw -= weight;
    }
    Err(ioerr("weighted sparse selection did not find a candidate"))
}

fn sparse_key_matches_fixed(
    label_positions: &HashMap<usize, usize>,
    key: usize,
    fixed: &HashMap<usize, u8>,
) -> Result<bool, Box<dyn Error>> {
    for (&label, &value) in fixed {
        let position = *label_positions.get(&label).ok_or_else(|| {
            ioerr(format!(
                "fixed N-queens label {label} is absent from contraction node"
            ))
        })?;
        if ((key >> position) & 1) as u8 != value {
            return Ok(false);
        }
    }
    Ok(true)
}

fn decode_sparse_key(labels: &[usize], key: usize) -> HashMap<usize, u8> {
    labels
        .iter()
        .enumerate()
        .map(|(position, &label)| (label, ((key >> position) & 1) as u8))
        .collect()
}

fn sample_sparse(
    node: &SparseEval,
    fixed: &HashMap<usize, u8>,
    assignment: &mut HashMap<usize, u8>,
    rng: &mut Rng64,
) -> Result<(), Box<dyn Error>> {
    match &node.kind {
        SparseEvalKind::Leaf => {
            let mut key = 0usize;
            for (position, label) in node.tensor.labels.iter().enumerate() {
                let value = *fixed.get(label).ok_or_else(|| {
                    ioerr(format!("leaf label {label} was not fixed by its parent"))
                })?;
                if value > 1 {
                    return Err(ioerr("sampled N-queens bit is not binary"));
                }
                if value == 1 {
                    key |= 1usize << position;
                }
            }
            if !node.tensor.data.contains_key(&key) {
                return Err(ioerr(
                    "downward pass selected a forbidden carry-tensor entry",
                ));
            }
            for &label in &node.tensor.labels {
                let value = fixed[&label];
                if let Some(previous) = assignment.insert(label, value) {
                    if previous != value {
                        return Err(ioerr("inconsistent N-queens sample assignment"));
                    }
                }
            }
        }
        SparseEvalKind::Node(left, right) => {
            let (union, candidates) = sparse_join_candidates(&left.tensor, &right.tensor)?;
            let union_positions = positions(&union);
            let mut kept = Vec::new();
            let mut weights = Vec::new();
            for (key, weight) in candidates {
                if sparse_key_matches_fixed(&union_positions, key, fixed)? {
                    kept.push(key);
                    weights.push(weight);
                }
            }
            let chosen = weighted_sparse_pick(&weights, rng)?;
            let values = decode_sparse_key(&union, kept[chosen]);

            let mut left_fixed = HashMap::with_capacity(left.tensor.labels.len());
            for &label in &left.tensor.labels {
                left_fixed.insert(label, values[&label]);
            }
            let mut right_fixed = HashMap::with_capacity(right.tensor.labels.len());
            for &label in &right.tensor.labels {
                right_fixed.insert(label, values[&label]);
            }

            sample_sparse(left, &left_fixed, assignment, rng)?;
            sample_sparse(right, &right_fixed, assignment, rng)?;
        }
    }
    Ok(())
}

fn validate_nqueens_board(n: usize, bits: &[u8]) -> bool {
    if bits.len() != n * n {
        return false;
    }
    let mut columns = vec![false; n];
    let mut down_diagonals = vec![false; 2 * n - 1];
    let mut up_diagonals = vec![false; 2 * n - 1];

    for row in 0..n {
        let mut row_count = 0usize;
        for col in 0..n {
            if bits[row * n + col] == 1 {
                row_count += 1;
                let down = row + (n - 1 - col);
                let up = row + col;
                if columns[col] || down_diagonals[down] || up_diagonals[up] {
                    return false;
                }
                columns[col] = true;
                down_diagonals[down] = true;
                up_diagonals[up] = true;
            } else if bits[row * n + col] != 0 {
                return false;
            }
        }
        if row_count != 1 {
            return false;
        }
    }
    columns.into_iter().all(|occupied| occupied)
}

fn check_nqueens_cli(cli: &Cli) -> Result<(), Box<dyn Error>> {
    if cli.input.is_some() || cli.l.is_some() || cli.p_af.is_some() {
        return Err(ioerr("--n cannot be combined with --input, --l or --p-af"));
    }
    Ok(())
}

fn run_nqueens_plan(cli: &Cli, n: usize) -> Result<(), Box<dyn Error>> {
    check_nqueens_cli(cli)?;
    let start = Instant::now();
    let network = build_nqueens_network(n)?;
    let (_plan, tc, sc, rwc) = make_plan(&network.code, &cli.order)?;
    let runtime_ms = start.elapsed().as_millis();
    let mut value = plan_json(
        "plan",
        cli.seed,
        &cli.order,
        &network.code,
        tc,
        sc,
        rwc,
        runtime_ms,
    );
    value["problem"] = json!("nqueens");
    value["n"] = json!(n);
    save_run(&cli.out, &value)?;
    tsv(
        "plan",
        &[
            ("n", n.to_string()),
            ("variables", network.code.unique_labels().len().to_string()),
            ("tensors", network.code.num_tensors().to_string()),
            ("tc", tc.to_string()),
            ("sc", sc.to_string()),
            ("rwc", rwc.to_string()),
        ],
        runtime_ms,
    );
    Ok(())
}

fn run_nqueens_count(cli: &Cli, n: usize) -> Result<(), Box<dyn Error>> {
    check_nqueens_cli(cli)?;
    let start = Instant::now();
    let network = build_nqueens_network(n)?;
    let (_plan, tc, sc, rwc) = make_plan(&network.code, &cli.order)?;
    let plan = _plan;
    let eval = eval_sparse(&plan, &network.leaves)?;
    if !eval.tensor.labels.is_empty() {
        return Err(ioerr("N-queens contraction did not produce a scalar"));
    }
    let board_count = eval.tensor.data.get(&0).copied().unwrap_or(0);
    let runtime_ms = start.elapsed().as_millis();
    let value = json!({
        "schema_version": 1,
        "mode": "count",
        "problem": "nqueens",
        "n": n,
        "seed": cli.seed,
        "order": cli.order,
        "variables": network.code.unique_labels().len(),
        "tensors": network.code.num_tensors(),
        "tc": tc,
        "sc": sc,
        "rwc": rwc,
        "valid_boards": board_count,
        "runtime_ms": runtime_ms
    });
    save_run(&cli.out, &value)?;
    tsv(
        "count",
        &[
            ("n", n.to_string()),
            ("valid_boards", board_count.to_string()),
            ("variables", network.code.unique_labels().len().to_string()),
            ("tensors", network.code.num_tensors().to_string()),
            ("tc", tc.to_string()),
            ("sc", sc.to_string()),
            ("rwc", rwc.to_string()),
        ],
        runtime_ms,
    );
    Ok(())
}

fn run_nqueens_sample(cli: &Cli, n: usize) -> Result<(), Box<dyn Error>> {
    check_nqueens_cli(cli)?;
    if cli.samples == 0 {
        return Err(ioerr("--samples must be positive"));
    }
    let start = Instant::now();
    let network = build_nqueens_network(n)?;
    let (plan, tc, sc, rwc) = make_plan(&network.code, &cli.order)?;
    let eval = eval_sparse(&plan, &network.leaves)?;
    if !eval.tensor.labels.is_empty() {
        return Err(ioerr("N-queens contraction did not produce a scalar"));
    }
    let exact_count = eval.tensor.data.get(&0).copied().unwrap_or(0);
    if exact_count == 0 {
        return Err(ioerr(format!(
            "there are no valid N-queens boards for N={n}"
        )));
    }

    let mut rng = Rng64::new(cli.seed);
    let mut histogram: BTreeMap<Vec<usize>, usize> = BTreeMap::new();
    let mut records = Vec::with_capacity(cli.samples);

    for sample_id in 0..cli.samples {
        let mut assignment = HashMap::new();
        sample_sparse(&eval, &HashMap::new(), &mut assignment, &mut rng)?;

        let mut bits = Vec::with_capacity(network.x_labels.len());
        for &label in &network.x_labels {
            bits.push(
                *assignment
                    .get(&label)
                    .ok_or_else(|| ioerr(format!("sample did not assign board bit {label}")))?,
            );
        }
        let valid = validate_nqueens_board(n, &bits);
        if !valid {
            return Err(ioerr(format!(
                "sample {sample_id} failed the N-queens validity check"
            )));
        }

        let mut columns = Vec::with_capacity(n);
        for row in 0..n {
            let queen_col = (0..n)
                .find(|&col| bits[row * n + col] == 1)
                .ok_or_else(|| ioerr("valid sample is missing a queen in one row"))?;
            columns.push(queen_col);
        }
        *histogram.entry(columns.clone()).or_insert(0) += 1;
        let board: Vec<Vec<u8>> = bits.chunks(n).map(|row| row.to_vec()).collect();
        records.push(json!({
            "sample": sample_id,
            "columns": columns,
            "board": board,
            "valid": valid
        }));
    }

    let unique_boards = histogram.len();
    let histogram_json: Vec<Value> = histogram
        .iter()
        .map(|(columns, count)| json!({"columns": columns, "count": count}))
        .collect();
    let runtime_ms = start.elapsed().as_millis();
    let value = json!({
        "schema_version": 1,
        "mode": "sample",
        "problem": "nqueens",
        "n": n,
        "seed": cli.seed,
        "order": cli.order,
        "samples_requested": cli.samples,
        "variables": network.code.unique_labels().len(),
        "tensors": network.code.num_tensors(),
        "tc": tc,
        "sc": sc,
        "rwc": rwc,
        "exact_board_count": exact_count,
        "unique_boards": unique_boards,
        "all_samples_valid": true,
        "histogram": histogram_json,
        "samples": records,
        "runtime_ms": runtime_ms
    });
    save_run(&cli.out, &value)?;
    tsv(
        "sample",
        &[
            ("n", n.to_string()),
            ("samples", cli.samples.to_string()),
            ("exact_board_count", exact_count.to_string()),
            ("unique_boards", unique_boards.to_string()),
            ("all_samples_valid", "true".to_string()),
            ("tc", tc.to_string()),
            ("sc", sc.to_string()),
            ("rwc", rwc.to_string()),
        ],
        runtime_ms,
    );
    Ok(())
}

fn run_plan(cli: &Cli) -> Result<(), Box<dyn Error>> {
    if let Some(n) = cli.n {
        return run_nqueens_plan(cli, n);
    }
    let start = Instant::now();
    let path = cli.input.as_ref().ok_or_else(|| {
        io::Error::new(
            ErrorKind::InvalidInput,
            "--mode plan requires --input bundle.json",
        )
    })?;
    let sg = load_spin_glass(path)?;
    let (code, _, _) = spin_network(&sg, cli.beta);
    let (_plan, tc, sc, rwc) = make_plan(&code, &cli.order)?;
    let runtime_ms = start.elapsed().as_millis();
    let value = plan_json("plan", cli.seed, &cli.order, &code, tc, sc, rwc, runtime_ms);
    save_run(&cli.out, &value)?;
    tsv(
        "plan",
        &[
            ("variables", code.unique_labels().len().to_string()),
            ("tensors", code.num_tensors().to_string()),
            ("tc", tc.to_string()),
            ("sc", sc.to_string()),
            ("rwc", rwc.to_string()),
        ],
        runtime_ms,
    );
    Ok(())
}

fn run_count(cli: &Cli) -> Result<(), Box<dyn Error>> {
    if let Some(n) = cli.n {
        return run_nqueens_count(cli, n);
    }
    let start = Instant::now();
    let sg = if let Some(path) = &cli.input {
        load_spin_glass(path)?
    } else if let Some(l) = cli.l {
        make_lattice(l, cli.p_af.unwrap_or(0.0), cli.seed)?
    } else {
        return Err(ioerr("--mode count requires --input bundle.json or --l L"));
    };
    let (code, dense_leaves, trop_leaves) = spin_network(&sg, cli.beta);
    let (plan, tc, sc, rwc) = make_plan(&code, &cli.order)?;
    let tropical = eval_trop(&plan, &trop_leaves)?;
    if !tropical.labels.is_empty() || tropical.data.len() != 1 || tropical.data[0].count == 0 {
        return Err(ioerr(
            "contraction did not produce a nonzero scalar tropical result",
        ));
    }
    let dense = eval_dense(&plan, &dense_leaves)?;
    if !dense.tensor.labels.is_empty() || dense.tensor.data.len() != 1 {
        return Err(ioerr("Boltzmann contraction did not produce a scalar"));
    }
    let log_z = dense.tensor.data[0].ln() + dense.log_scale;
    let e_min = tropical.data[0].energy;
    let g0 = tropical.data[0].count;
    let runtime_ms = start.elapsed().as_millis();
    let value = json!({
        "schema_version":1, "mode":"count", "seed":cli.seed,
        "input":cli.input.as_ref().map(|p|p.display().to_string()),
        "l":cli.l, "p_af":cli.p_af, "beta":cli.beta, "order":cli.order,
        "variables":sg.num_spins(), "edges":sg.edges.len(), "tensors":code.num_tensors(),
        "tc":tc, "sc":sc, "rwc":rwc, "log_z":log_z,
        "energy_min":e_min, "ground_state_degeneracy":g0, "runtime_ms":runtime_ms
    });
    save_run(&cli.out, &value)?;
    tsv(
        "count",
        &[
            ("energy_min", e_min.to_string()),
            ("ground_state_degeneracy", g0.to_string()),
            ("log_z", format!("{log_z:.10}")),
            ("tc", tc.to_string()),
            ("sc", sc.to_string()),
            ("rwc", rwc.to_string()),
        ],
        runtime_ms,
    );
    Ok(())
}

fn run_sample(cli: &Cli) -> Result<(), Box<dyn Error>> {
    if let Some(n) = cli.n {
        return run_nqueens_sample(cli, n);
    }
    let start = Instant::now();
    let path = cli.input.as_ref().ok_or_else(|| {
        io::Error::new(
            ErrorKind::InvalidInput,
            "--mode sample currently requires --input bundle.json",
        )
    })?;
    if cli.samples == 0 {
        return Err(ioerr("--samples must be positive"));
    }
    let sg = load_spin_glass(path)?;
    let (code, dense_leaves, trop_leaves) = spin_network(&sg, cli.beta);
    let (plan, tc, sc, rwc) = make_plan(&code, &cli.order)?;
    let tropical = eval_trop(&plan, &trop_leaves)?;
    let e_min = tropical.data[0].energy;
    let g0 = tropical.data[0].count;
    let eval = eval_dense(&plan, &dense_leaves)?;
    let log_z = eval.tensor.data[0].ln() + eval.log_scale;
    let p0 = ((g0 as f64).ln() - cli.beta * e_min as f64 - log_z)
        .exp()
        .clamp(0.0, 1.0);
    let mut rng = Rng64::new(cli.seed);
    let mut records = Vec::with_capacity(cli.samples);
    let mut histogram: BTreeMap<i32, usize> = BTreeMap::new();
    let mut best_energy = i32::MAX;
    let mut best_config = String::new();
    for sample_id in 0..cli.samples {
        let mut assignment = HashMap::new();
        sample_dense(&eval, &HashMap::new(), &mut assignment, &mut rng)?;
        let mut bits = Vec::with_capacity(sg.num_spins());
        for i in 0..sg.num_spins() {
            bits.push(*assignment.get(&i).ok_or_else(|| {
                io::Error::new(
                    ErrorKind::InvalidData,
                    format!("sample did not assign spin {i}"),
                )
            })?);
        }
        let energy = sg.energy(&bits);
        *histogram.entry(energy).or_insert(0) += 1;
        let config = bits
            .iter()
            .map(|b| b.to_string())
            .collect::<Vec<_>>()
            .join(",");
        if energy < best_energy {
            best_energy = energy;
            best_config = config.clone();
        }
        records.push(json!({"sample":sample_id,"energy":energy,"bits":bits}));
    }
    let runtime_ms = start.elapsed().as_millis();
    fs::create_dir_all(&cli.out)?;
    fs::write(cli.out.join("best_config.txt"), format!("{best_config}\n"))?;
    let hist_json: Vec<Value> = histogram
        .iter()
        .map(|(e, c)| json!({"energy":e,"count":c}))
        .collect();
    let value = json!({
        "schema_version":1, "mode":"sample", "seed":cli.seed, "input":path.display().to_string(),
        "beta":cli.beta, "order":cli.order, "samples_requested":cli.samples,
        "variables":sg.num_spins(), "tc":tc, "sc":sc, "rwc":rwc,
        "energy_min":e_min, "ground_state_degeneracy":g0, "log_z":log_z,
        "ground_state_probability":p0, "predicted_ground_state_samples":cli.samples as f64*p0,
        "best_sample_energy":best_energy, "best_config_csv":best_config,
        "histogram":hist_json, "samples":records, "runtime_ms":runtime_ms
    });
    save_run(&cli.out, &value)?;
    tsv(
        "sample",
        &[
            ("samples", cli.samples.to_string()),
            ("energy_min", e_min.to_string()),
            ("ground_state_degeneracy", g0.to_string()),
            ("log_z", format!("{log_z:.10}")),
            (
                "predicted_ground_state_samples",
                format!("{:.4}", cli.samples as f64 * p0),
            ),
            ("best_sample_energy", best_energy.to_string()),
        ],
        runtime_ms,
    );
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    match cli.mode.as_str() {
        "gadgets" => run_gadgets(&cli),
        "plan" => run_plan(&cli),
        "count" => run_count(&cli),
        "sample" => run_sample(&cli),
        other => Err(ioerr(format!("unsupported mode: {other}"))),
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
