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
