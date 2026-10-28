#!/usr/bin/env python3
"""Week 8 Part 6 — reproduce Chinchilla scaling-law fit and compute allocation.

Use from the week8_1 directory:
  uv run --no-project --with numpy --with scipy --with matplotlib \
      python -u part6_scaling.py --bootstrap 400

Input: Epoch AI / Besiroglu digitization, pinned commit 92258837425e1b5f2851d624287f0120583a3d0e.
Full data must have 245 rows; the five largest losses are excluded. Fits the
Huber (delta=.001) penalty on log predicted-minus-observed loss with L-BFGS-B
and a grid of starting values, then 400 bootstrap refits using the best fit.
No models are trained; NumPy/SciPy/Matplotlib on CPU are sufficient.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, nnls

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
ART = ROOT / 'artifacts'
EVID = ROOT / 'evidence'
REV = '92258837425e1b5f2851d624287f0120583a3d0e'
SOURCE = f'https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla/{REV}/data/svg_extracted_data.csv'
CSV_PATH = DATA / 'chinchilla-runs.csv'

# N (parameters) and D (tokens) are normalized to avoid badly scaled optimizer gradients.
N_SCALE = 1e9
D_SCALE = 1e10
HUBER_DELTA = 0.001
BOUNDS = [(0.5, 2.3), (1e-6, 5.0), (1e-6, 5.0), (0.08, 0.80), (0.08, 0.80)]
PRINTED = {'E':1.69, 'A':406.4, 'B':410.7, 'alpha':0.34, 'beta':0.28}
CHINCHILLA_C = 5.76e23   # The budget explicitly used in the Week 8 VERIFY question.


def log(*xs):
    print(*xs, flush=True)


def ensure_input():
    if CSV_PATH.is_file() and CSV_PATH.stat().st_size > 1000:
        log('Using existing data:', CSV_PATH)
        return
    DATA.mkdir(parents=True, exist_ok=True)
    log('Downloading fixed Chinchilla dataset from Epoch AI:')
    log(SOURCE)
    req = urllib.request.Request(SOURCE, headers={'User-Agent':'week8-learning-sheet/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            blob = r.read()
        if b'Model Size' not in blob[:500] or len(blob) < 1000:
            raise ValueError('Downloaded data does not appear to be the expected CSV.')
        CSV_PATH.write_bytes(blob)
        log('Downloaded', len(blob), 'bytes to', CSV_PATH)
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        raise RuntimeError(
            'Could not download Chinchilla CSV. Try this in your week8_1 terminal:\n'
            '  mkdir -p data\n'
            f'  curl -fL --retry 4 --connect-timeout 25 -o data/chinchilla-runs.csv "{SOURCE}"\n'
            'Then rerun this script; your earlier results do not need to be repeated.\n'
            f'Underlying download error: {e}'
        ) from e


def load_runs():
    with CSV_PATH.open('r', encoding='utf-8-sig', newline='') as fh:
        reader=csv.DictReader(fh)
        required={'Model Size','Training FLOP','loss'}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f'Unexpected columns {reader.fieldnames}; expected {sorted(required)}')
        rows=list(reader)
    if len(rows) != 245:
        raise ValueError(f'Expected exactly 245 original runs; got {len(rows)}. Please check the input CSV.')
    n=np.array([float(row['Model Size']) for row in rows],dtype=np.float64)
    c=np.array([float(row['Training FLOP']) for row in rows],dtype=np.float64)
    loss=np.array([float(row['loss']) for row in rows],dtype=np.float64)
    if not (np.isfinite(n).all() and np.isfinite(c).all() and np.isfinite(loss).all()):
        raise ValueError('Nonfinite values in input CSV')
    if not (np.all(n>0) and np.all(c>0) and np.all(loss>0)):
        raise ValueError('All N, C, and losses must be positive')
    d=c/(6*n)
    retained=np.argsort(loss)[:-5]
    dropped=np.argsort(loss)[-5:]
    log('Original runs:',len(rows),'| fitted runs:',len(retained),'| excluded 5 highest losses:',np.round(loss[dropped],4).tolist())
    log('Model sizes: %.3g – %.3g | tokens: %.3g – %.3g'%(np.min(n),np.max(n),np.min(d),np.max(d)))
    return n[retained],c[retained],d[retained],loss[retained]


def to_theta(p):
    return np.array([p['E'],p['A']/N_SCALE**p['alpha'],p['B']/D_SCALE**p['beta'],p['alpha'],p['beta']],dtype=np.float64)


def from_theta(th):
    E,a,b,alpha,beta=map(float,th)
    return {'E':E,'A':a*N_SCALE**alpha,'B':b*D_SCALE**beta,'alpha':alpha,'beta':beta}


def predict(theta,n,d):
    E,a,b,alpha,beta=theta
    return E+a*np.exp(-alpha*np.log(np.asarray(n)/N_SCALE))+b*np.exp(-beta*np.log(np.asarray(d)/D_SCALE))


def huber_loss_and_grad(theta,logn,logd,log_loss):
    E,a,b,alpha,beta=theta
    fn=np.exp(-alpha*logn)
    fd=np.exp(-beta*logd)
    aterm=a*fn
    bterm=b*fd
    estimate=E+aterm+bterm
    if np.any(estimate<=0):
        return 1e10, np.zeros(5)
    residual=np.log(estimate)-log_loss
    abs_r=np.abs(residual)
    inner=abs_r<=HUBER_DELTA
    # Standard Huber convention (0.5*r^2 inside); factor 2 does not alter optimum.
    penalty=np.where(inner, 0.5*residual**2,
                     HUBER_DELTA*(abs_r-0.5*HUBER_DELTA))
    grad_r=np.where(inner,residual,HUBER_DELTA*np.sign(residual)) / estimate
    grad=np.array([
        grad_r.mean(),
        np.mean(grad_r*fn),
        np.mean(grad_r*fd),
        np.mean(grad_r*(-aterm*logn)),
        np.mean(grad_r*(-bterm*logd))
    ])
    return float(penalty.mean()),grad


def inputs(n,d,loss):
    return np.log(n/N_SCALE),np.log(d/D_SCALE),np.log(loss)


def optimize(start, inputs_tuple, maxiter=700):
    args=inputs_tuple
    return minimize(huber_loss_and_grad,start,args=args,method='L-BFGS-B',jac=True,
                    bounds=BOUNDS,options={'maxiter':maxiter,'ftol':1e-14,'gtol':1e-12,'maxls':35})


def multistart_fit(n,d,loss):
    stuff=inputs(n,d,loss)
    starts=[to_theta({'E':1.817,'A':478,'B':2143,'alpha':0.347,'beta':0.367}),to_theta(PRINTED)]
    # Grid of E, alpha, beta; solve positive linear coefficients for each start.
    for E in (1.65,1.82,1.97):
        for alpha in (.25,.35,.45):
            for beta in (.25,.37,.49):
                xn=np.exp(-alpha*stuff[0]); xd=np.exp(-beta*stuff[1])
                coeff,_=nnls(np.column_stack((xn,xd)),loss-E)
                starts.append(np.array([E,np.clip(coeff[0],1e-5,4.9),np.clip(coeff[1],1e-5,4.9),alpha,beta]))
    results=[]
    t0=time.time()
    for k,x in enumerate(starts,1):
        fit=optimize(x,stuff)
        if np.isfinite(fit.fun): results.append(fit)
        if k%10==0: log(f'  L-BFGS multistart: {k}/{len(starts)}; best Huber={min(r.fun for r in results):.9g}')
    if not results: raise RuntimeError('All optimization starts failed')
    best=min(results,key=lambda r:r.fun)
    best=optimize(best.x,stuff,1000)
    log(f'Best Huber objective={best.fun:.9g} from {len(starts)} starts in {time.time()-t0:.1f}s; optimizer: {best.message}')
    return best.x


def bootstrap_fit(n,d,loss,center,num,seed):
    rng=np.random.default_rng(seed)
    arr=[]
    failed=0
    t0=time.time()
    for b in range(num):
        idx=rng.integers(0,len(loss),size=len(loss))
        sampled=inputs(n[idx],d[idx],loss[idx])
        trial=optimize(center,sampled,maxiter=450)
        if not trial.success or not np.all(np.isfinite(trial.x)):
            trial=optimize(np.clip(trial.x, [v[0] for v in BOUNDS], [v[1] for v in BOUNDS]),sampled,maxiter=900)
        if not np.all(np.isfinite(trial.x)):
            failed+=1
            continue
        arr.append(trial.x)
        if (b+1)%50==0 or b+1==num:
            log(f'  bootstrap: {b+1}/{num}, valid={len(arr)}, elapsed={time.time()-t0:.1f}s')
    if len(arr)!=num:
        raise RuntimeError(f'Only {len(arr)}/{num} valid bootstrap refits; failed {failed}')
    return np.array(arr,dtype=np.float64)


def optimal(theta,c):
    p=from_theta(theta)
    alpha,beta=p['alpha'],p['beta']
    logG=(math.log(alpha*p['A'])-math.log(beta*p['B']))/(alpha+beta)
    lnC=np.log(np.asarray(c,dtype=np.float64)/6.0)
    nstar=np.exp(logG+beta/(alpha+beta)*lnC)
    dstar=np.exp(-logG+alpha/(alpha+beta)*lnC)
    return nstar,dstar


def curves(n,c,d,y,theta):
    # Figure 19: loss vs size by compute, and measured vs predicted.
    ctarget=(1e19,1e20,1e21)
    colors=('#2c7fb8','#e69f00','#31a354')
    fig,ax=plt.subplots(1,2,figsize=(12.4,5.0),constrained_layout=True)
    ax[0].scatter(n,y,c='#b9bdc3',s=13,alpha=.5,label='Other training runs')
    for budget,color in zip(ctarget,colors):
        mask=np.abs(c/budget-1)<=.15
        ax[0].scatter(n[mask],y[mask],s=20,color=color,alpha=.9,label=f'Within 15% of {budget:.0e} FLOP')
        x=np.logspace(np.log10(n.min()*.85),np.log10(n.max()*1.2),220)
        yhat=predict(theta,x,budget/(6*x))
        ax[0].plot(x,yhat,color=color,lw=1.9)
    ax[0].set(xscale='log',xlabel='Model parameters N',ylabel='Final loss (nats/token)',title='Chinchilla runs and fixed-compute fit')
    ax[0].legend(loc='best',fontsize=7.8)
    yhat=predict(theta,n,d)
    sc=ax[1].scatter(y,yhat,c=np.log10(c),cmap='viridis',s=19,alpha=.8)
    bounds=(min(y.min(),yhat.min())-.1,max(y.max(),yhat.max())+.1)
    ax[1].plot(bounds,bounds,'--',color='black',lw=1)
    ax[1].set(xlim=bounds,ylim=bounds,xlabel='Measured loss',ylabel='Fitted loss',title='Measured vs fitted, 240 runs')
    fig.colorbar(sc,ax=ax[1],label='log10(compute FLOP)',shrink=.8)
    fig.savefig(EVID/'part6_chinchilla_fit.png',dpi=190)
    plt.close(fig)
    return yhat


def residual_plot(n,refit,printed):
    fig,ax=plt.subplots(1,2,figsize=(12.2,4.8),constrained_layout=True)
    ax[0].scatter(n,refit,s=13,color='#2c7fb8',alpha=.65,label=f'Refit, mean={refit.mean():+.4f}')
    ax[0].scatter(n,printed,s=13,color='#e08a33',alpha=.5,label=f'Printed, mean={printed.mean():+.4f}')
    ax[0].axhline(0,color='black',ls='--',lw=1)
    ax[0].set(xscale='log',xlabel='Parameters N',ylabel='log(predicted) - log(measured)',title='Residuals vs model size')
    ax[0].legend(fontsize=9)
    bins=np.linspace(min(refit.min(),printed.min())-.01,max(refit.max(),printed.max())+.01,35)
    ax[1].hist(refit,bins=bins,color='#2c7fb8',alpha=.65,label='Refit')
    ax[1].hist(printed,bins=bins,color='#e08a33',alpha=.60,label='Printed')
    ax[1].axvline(0,color='black',ls='--',lw=1)
    ax[1].set(xlabel='log(predicted) - log(measured)',ylabel='Runs',title='Residual distributions')
    ax[1].legend()
    fig.savefig(EVID/'part6_chinchilla_residuals.png',dpi=190)
    plt.close(fig)


def allocation_plot(theta):
    budgets=np.logspace(18,25,220)
    nopt,dopt=optimal(theta,budgets)
    nprint,dprint=optimal(to_theta(PRINTED),budgets)
    fig,ax=plt.subplots(1,2,figsize=(12.3,4.7),constrained_layout=True)
    ax[0].plot(budgets,nopt,color='#2c7fb8',lw=2,label='Refit')
    ax[0].plot(budgets,nprint,'--',color='#e08a33',lw=2,label='Printed parameters')
    ax[0].plot(CHINCHILLA_C,70e9,marker='*',color='#2ca25f',markersize=14,label='Chinchilla: 70B')
    ax[0].set(xscale='log',yscale='log',xlabel='Compute C (FLOP)',ylabel='Optimal parameters N*',title='Compute-optimal model size')
    ax[0].legend(fontsize=8)
    ax[1].plot(budgets,dopt,color='#2c7fb8',lw=2,label='Refit')
    ax[1].plot(budgets,dprint,'--',color='#e08a33',lw=2,label='Printed parameters')
    ax[1].plot(CHINCHILLA_C,1.4e12,marker='*',color='#2ca25f',markersize=14,label='Chinchilla: 1.4T')
    ax[1].set(xscale='log',yscale='log',xlabel='Compute C (FLOP)',ylabel='Optimal training tokens D*',title='Compute-optimal training data')
    ax[1].legend(fontsize=8)
    fig.savefig(EVID/'part6_chinchilla_allocation.png',dpi=190)
    plt.close(fig)
    with (ART/'part6_optimal_allocation.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['compute_FLOP','refit_parameters_N','refit_tokens_D','printed_parameters_N','printed_tokens_D'])
        for vals in zip(budgets,nopt,dopt,nprint,dprint):writer.writerow(vals)


def update_readme(results):
    readme=ROOT/'README.md'
    original=readme.read_text(encoding='utf-8') if readme.exists() else '# Week 8\n'
    start='<!-- BEGIN WEEK8 PART6 -->'
    end='<!-- END WEEK8 PART6 -->'
    section=f'''{start}
## Part 6 — Scaling laws: how to spend a compute budget

Dataset: Epoch AI's Chinchilla Figure 4 reconstruction, fixed revision `{REV}`:
`{SOURCE}`. Data file `data/chinchilla-runs.csv` has 245 rows, and the 5 highest-loss
runs are excluded (240 fitted runs). Keep `data/` and `artifacts/` out of Git.

Reproduce inside the `week8_1/` folder:

```bash
uv run --no-project --with numpy --with scipy --with matplotlib python -u part6_scaling.py --bootstrap 400
```

Fit `L(N,D)=E+A/N^alpha+B/D^beta`, with `D=C/(6N)`, via Huber loss
(`delta=0.001`) applied to log-loss residuals; fit L-BFGS-B from a grid,
then bootstrap 400 times (seed 1730) for standard errors and a 95% interval.

| Parameter | Refit | Bootstrap standard error |
|---|---:|---:|
| E | {results['fit']['E']:.5g} | {results['standard_errors']['E']:.4g} |
| A | {results['fit']['A']:.5g} | {results['standard_errors']['A']:.4g} |
| B | {results['fit']['B']:.5g} | {results['standard_errors']['B']:.4g} |
| alpha | {results['fit']['alpha']:.5g} | {results['standard_errors']['alpha']:.4g} |
| beta | {results['fit']['beta']:.5g} | {results['standard_errors']['beta']:.4g} |

- Figure 19, fit and measured-vs-predicted: `evidence/part6_chinchilla_fit.png`.
- Figure 20, residual comparison: `evidence/part6_chinchilla_residuals.png`.
- Figure 21, optimal allocation: `evidence/part6_chinchilla_allocation.png`.
- The mean log-loss residual for our refit is {results['residuals']['refit_mean']:+.5f}; for printed parameters {results['residuals']['printed_mean']:+.5f}.
- At 5.76e23 FLOP, our `D*/N*` is {results['chinchilla_budget']['ratio_tokens_per_parameter']:.3f} with 95% bootstrap interval [{results['chinchilla_budget']['ratio_95pct_interval'][0]:.2f}, {results['chinchilla_budget']['ratio_95pct_interval'][1]:.2f}].
- Raw numeric results are `artifacts/part6_summary.json` and `artifacts/part6_optimal_allocation.csv`.
{end}
'''
    if start in original and end in original:
        original=original[:original.index(start)]+section+original[original.index(end)+len(end):]
    else:
        original=original.rstrip()+'\n\n'+section
    readme.write_text(original,encoding='utf-8')


def run(boot,seed):
    for d in (DATA,ART,EVID):d.mkdir(parents=True,exist_ok=True)
    ensure_input()
    n,c,d,loss=load_runs()
    fit=multistart_fit(n,d,loss)
    p=from_theta(fit)
    log('Best fit:', ' '.join(f'{k}={v:.7g}' for k,v in p.items()))
    log('Running',boot,'bootstrap refits; this may take longer than the fit...')
    boots=bootstrap_fit(n,d,loss,fit,boot,seed)
    original_param=[from_theta(row) for row in boots]
    errs={key:float(np.std([row[key] for row in original_param],ddof=1)) for key in p}
    log('Bootstrap standard errors:', ' '.join(f'{k}={v:.5g}' for k,v in errs.items()))
    fitted=predict(fit,n,d)
    printed=predict(to_theta(PRINTED),n,d)
    resid_fit=np.log(fitted)-np.log(loss)
    resid_printed=np.log(printed)-np.log(loss)
    opt_n,opt_d=optimal(fit,CHINCHILLA_C)
    opt_np,opt_dp=optimal(to_theta(PRINTED),CHINCHILLA_C)
    ratios=[]
    for row in boots:
        ni,di=optimal(row,CHINCHILLA_C)
        ratios.append(float(di/ni))
    ci=np.percentile(ratios,[2.5,97.5])
    ratio=float(opt_d/opt_n)
    checks={
       'alpha_in_range': bool(.32<=p['alpha']<=.38),
       'beta_in_range': bool(.32<=p['beta']<=.42),
       'E_in_range': bool(1.75<=p['E']<=1.90),
       'refit_mean_residual_within_0.005': bool(abs(np.mean(resid_fit))<=.005),
       'printed_mean_residual_above_0.01': bool(np.mean(resid_printed)>.01),
       'optimal_D_over_N_10_to_30': bool(10<=ratio<=30),
       'ratio_95pct_interval_includes_20': bool(ci[0]<=20<=ci[1]),
       'bootstrap_count_400': bool(boot==400)
    }
    log('\nRefit mean log residual: %+.6f | printed mean log residual: %+.6f'%(np.mean(resid_fit),np.mean(resid_printed)))
    log('At C=5.76e23: N*=%.4g, D*=%.4g, D*/N*=%.3f [95%% CI %.3f, %.3f]'%(opt_n,opt_d,ratio,ci[0],ci[1]))
    log('Printed parameters give D*/N*=%.3f'%(opt_dp/opt_np))
    curves(n,c,d,loss,fit)
    residual_plot(n,resid_fit,resid_printed)
    allocation_plot(fit)
    summary={
       'data_source':SOURCE,
       'total_runs':245,'fitted_runs':len(loss),'excluded_highest_loss':5,
       'Huber_delta':HUBER_DELTA,'bootstrap_refits':boot,'bootstrap_seed':seed,
       'fit':p,'standard_errors':errs,'printed_parameters':PRINTED,
       'residuals':{'refit_mean':float(np.mean(resid_fit)),'printed_mean':float(np.mean(resid_printed))},
       'chinchilla_budget':{'FLOP':CHINCHILLA_C,'optimal_parameters_N':float(opt_n),
             'optimal_training_tokens_D':float(opt_d),'ratio_tokens_per_parameter':ratio,
             'ratio_95pct_interval':[float(ci[0]),float(ci[1])],
             'printed_ratio_tokens_per_parameter':float(opt_dp/opt_np)},
       'compute_scaling_exponents':{'model_N':p['beta']/(p['alpha']+p['beta']),
                                     'data_D':p['alpha']/(p['alpha']+p['beta'])},
       'checks':checks,'all_checks_pass':bool(all(checks.values()))
    }
    (ART/'part6_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    update_readme(summary)
    log('\n=== Part 6 VERIFY ===')
    for k,v in checks.items():log(f'{k}: {"PASS" if v else "CHECK"}')
    for fn in ('part6_chinchilla_fit.png','part6_chinchilla_residuals.png','part6_chinchilla_allocation.png'):
        log('Evidence:',EVID/fn)
    log('Saved metrics:',ART/'part6_summary.json')
    log('Updated README:',ROOT/'README.md')
    log('Overall:', 'PASS' if all(checks.values()) else 'CHECK')
    return 0 if all(checks.values()) else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap',type=int,default=400,help='Number of bootstrap refits (Week 8 requires 400).')
    parser.add_argument('--seed',type=int,default=1730)
    args=parser.parse_args()
    if args.bootstrap<2:parser.error('--bootstrap must be >= 2')
    return run(args.bootstrap,args.seed)

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:
        log('ERROR:',str(e))
        sys.exit(1)
