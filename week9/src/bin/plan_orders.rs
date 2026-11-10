use anyhow::{ensure, Context, Result};
use omeco::{optimize_code, EinCode, GreedyMethod, NestedEinsum};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::fs;

#[derive(Serialize)]
struct PlanRecord {
    version: u32,
    model: String,
    evidence: BTreeMap<String, usize>,
    method: String,
    seed: Option<u64>,
    input_scopes: Vec<String>,
    tree: NestedEinsum<char>,
}

#[derive(Serialize)]
struct Step {
    step: usize,
    left_labels: String,
    right_labels: String,
    retained_labels: String,
    work: usize,
    output_elements: usize,
}

#[derive(Serialize)]
struct Complexity {
    method: String,
    work: usize,
    largest_tensor_elements: usize,
    steps: Vec<Step>,
}

struct Active {
    tree: NestedEinsum<char>,
    labels: Vec<char>,
}

fn read_scopes(path: &str) -> Result<(Vec<Vec<char>>, HashMap<char, usize>)> {
    let content = fs::read_to_string(path)?;
    let mut it = content.split_whitespace();

    ensure!(it.next() == Some("MARKOV"), "Expected MARKOV");

    let n: usize = it.next().context("Missing variable count")?.parse()?;

    let mut dims = Vec::new();

    for _ in 0..n {
        dims.push(it.next().context("Missing dimension")?.parse::<usize>()?);
    }

    let nf: usize = it.next().context("Missing factor count")?.parse()?;

    let mut scopes = Vec::new();

    for _ in 0..nf {
        let rank: usize = it.next().context("Missing factor rank")?.parse()?;
        let mut scope = Vec::new();

        for _ in 0..rank {
            let v: usize = it.next().context("Missing variable ID")?.parse()?;
            ensure!(v < n, "Invalid variable ID");
            scope.push((b'a' + v as u8) as char);
        }

        scopes.push(scope);
    }

    let mut sizes = HashMap::new();

    for (v, &dim) in dims.iter().enumerate() {
        sizes.insert((b'a' + v as u8) as char, dim);
    }

    Ok((scopes, sizes))
}

fn tensor_size(labels: &[char], sizes: &HashMap<char, usize>) -> usize {
    labels.iter().map(|x| sizes[x]).product()
}

// Generate a reproducible random binary contraction tree.
fn random_tree(scopes: &[Vec<char>], seed: u64) -> NestedEinsum<char> {
    let mut rng = StdRng::seed_from_u64(seed);

    let mut active: Vec<Active> = scopes
        .iter()
        .enumerate()
        .map(|(i, labels)| Active {
            tree: NestedEinsum::leaf(i),
            labels: labels.clone(),
        })
        .collect();

    while active.len() > 1 {
        let i = rng.random_range(0..active.len());
        let mut j = rng.random_range(0..active.len() - 1);

        if j >= i {
            j += 1;
        }

        let hi = i.max(j);
        let lo = i.min(j);

        let right = active.remove(hi);
        let left = active.remove(lo);

        let mut union = left.labels.clone();

        for &label in &right.labels {
            if !union.contains(&label) {
                union.push(label);
            }
        }

        // Retain labels that still occur in another active tensor.
        let output: Vec<char> = union
            .into_iter()
            .filter(|label| active.iter().any(|x| x.labels.contains(label)))
            .collect();

        let code = EinCode::new(
            vec![left.labels.clone(), right.labels.clone()],
            output.clone(),
        );

        let tree = NestedEinsum::node(vec![left.tree, right.tree], code);

        active.push(Active {
            tree,
            labels: output,
        });
    }

    active.pop().unwrap().tree
}

// Independently recount work and largest intermediate size.
fn analyze_tree(
    tree: &NestedEinsum<char>,
    input_scopes: &[Vec<char>],
    sizes: &HashMap<char, usize>,
    steps: &mut Vec<Step>,
    largest: &mut usize,
) -> Result<Vec<char>> {
    match tree {
        NestedEinsum::Leaf { tensor_index } => {
            let labels = input_scopes
                .get(*tensor_index)
                .context("Invalid leaf tensor index")?
                .clone();

            *largest = (*largest).max(tensor_size(&labels, sizes));

            Ok(labels)
        }

        NestedEinsum::Node { args, eins } => {
            ensure!(args.len() == 2, "Expected binary contraction");

            let left = analyze_tree(&args[0], input_scopes, sizes, steps, largest)?;

            let right = analyze_tree(&args[1], input_scopes, sizes, steps, largest)?;

            let mut union = left.clone();

            for &label in &right {
                if !union.contains(&label) {
                    union.push(label);
                }
            }

            let work = tensor_size(&union, sizes);
            let output = eins.iy.clone();
            let output_elements = tensor_size(&output, sizes);

            *largest = (*largest).max(output_elements);

            steps.push(Step {
                step: steps.len() + 1,
                left_labels: left.iter().collect(),
                right_labels: right.iter().collect(),
                retained_labels: output.iter().collect(),
                work,
                output_elements,
            });

            Ok(output)
        }
    }
}

fn save_order(
    method: &str,
    seed: Option<u64>,
    tree: NestedEinsum<char>,
    scopes: &[Vec<char>],
    sizes: &HashMap<char, usize>,
) -> Result<()> {
    let mut steps = Vec::new();
    let mut largest = 0;

    let final_labels = analyze_tree(&tree, scopes, sizes, &mut steps, &mut largest)?;

    ensure!(final_labels.is_empty(), "Final output must be scalar");

    let work: usize = steps.iter().map(|s| s.work).sum();

    let dir = format!("runs/{}", method);
    fs::create_dir_all(&dir)?;

    let mut evidence = BTreeMap::new();
    evidence.insert("A".to_string(), 0);
    evidence.insert("D".to_string(), 0);

    let record = PlanRecord {
        version: 1,
        model: "asia.uai".to_string(),
        evidence,
        method: method.to_string(),
        seed,
        input_scopes: scopes.iter().map(|s| s.iter().collect()).collect(),
        tree,
    };

    let complexity = Complexity {
        method: method.to_string(),
        work,
        largest_tensor_elements: largest,
        steps,
    };

    fs::write(
        format!("{}/order.json", dir),
        serde_json::to_string_pretty(&record)?,
    )?;

    fs::write(
        format!("{}/complexity.json", dir),
        serde_json::to_string_pretty(&complexity)?,
    )?;

    println!("\n===== {} ORDER =====", method.to_uppercase());
    println!("Work = {}", work);
    println!("Largest tensor = {}", largest);
    println!("Number of contraction steps = {}", complexity.steps.len());
    println!("Saved {}/order.json", dir);
    println!("Saved {}/complexity.json", dir);

    Ok(())
}

fn main() -> Result<()> {
    println!("===== ASIA AD CONTRACTION ORDER =====");

    let (mut scopes, sizes) = read_scopes("asia.uai")?;

    // AD evidence: A=0 and D=0.
    // Each observation contributes a unary indicator tensor.
    scopes.push(vec!['a']);
    scopes.push(vec!['h']);

    println!("Input tensors: {}", scopes.len());
    println!("Output: scalar evidence mass");

    // 1. Random order, fixed seed.
    let random = random_tree(&scopes, 42);

    save_order("random", Some(42), random, &scopes, &sizes)?;

    // 2. OMECO GreedyMethod.
    let code = EinCode::new(scopes.clone(), vec![]);

    let greedy = optimize_code(&code, &sizes, &GreedyMethod::default())
        .expect("OMECO greedy optimization failed");

    ensure!(greedy.is_binary(), "Greedy tree is not binary");

    save_order("greedy", None, greedy, &scopes, &sizes)?;

    println!("\nALL ORDER PLANS GENERATED");

    Ok(())
}
