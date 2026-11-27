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
