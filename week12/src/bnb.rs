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
