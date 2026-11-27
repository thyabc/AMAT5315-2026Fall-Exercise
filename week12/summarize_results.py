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
