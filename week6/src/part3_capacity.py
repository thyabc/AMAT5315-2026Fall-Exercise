from pathlib import Path
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import argparse
import csv
import json
import math
import time

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts'
EVIDENCE = ROOT / 'evidence'
ARTIFACTS.mkdir(exist_ok=True)
EVIDENCE.mkdir(exist_ok=True)
SEED = 20261014
MAX_SWEEPS = 100
FIELDS = ['N','P','alpha','trial','seed','unstable_fraction','mean_overlap',
          'recall_success','unconverged','sweeps','runtime_sec']


def make_recaller(n):
    rows = jnp.arange(10)
    @jax.jit
    def recall(J, starts, key):
        initial = (starts, starts @ J, key,
                   jnp.array(0, dtype=jnp.int32),
                   jnp.ones(10, dtype=jnp.int32))
        def condition(carry):
            return (carry[3] < MAX_SWEEPS) & jnp.any(carry[4] != 0)
        def sweep(carry):
            states, fields, rng, sweeps, _ = carry
            rng, subkey = jax.random.split(rng)
            keys = jax.random.split(subkey, 10)
            orders = jax.vmap(lambda k: jax.random.permutation(k, n))(keys)
            def update_one(state, idx):
                spins, h, flips = state
                old = spins[rows, idx]
                field = h[rows, idx]
                new = jnp.where(field > 0, 1., jnp.where(field < 0, -1., old))
                delta = new - old
                spins = spins.at[rows, idx].set(new)
                h = h + delta[:, None] * J[idx, :]
                flips = flips + (delta != 0).astype(jnp.int32)
                return (spins, h, flips), None
            (states, fields, flips), _ = jax.lax.scan(
                update_one,
                (states, fields, jnp.zeros(10, dtype=jnp.int32)),
                orders.T
            )
            return states, fields, rng, sweeps + 1, flips
        result = jax.lax.while_loop(condition, sweep, initial)
        return result[0], result[3], result[4]
    return recall


def one_trial(n, pct, trial, recall):
    P = n * pct // 100
    seed = SEED + n * 10000 + pct * 100 + trial
    rng = np.random.default_rng(seed)
    patterns = (2 * rng.integers(0, 2, size=(P,n)) - 1).astype(np.float32)
    t = time.perf_counter()
    X = jnp.asarray(patterns)
    J = (X.T @ X) / n
    J = J.at[jnp.diag_indices(n)].set(0.)
    unstable = float(jnp.mean((X * (X @ J)) < 0))
    n_query = min(P,10)
    if n_query < 10:
        padded = np.concatenate([patterns[:n_query],
             np.repeat(patterns[:1], 10-n_query, axis=0)],axis=0)
    else:
        padded = patterns[:10]
    out, sweeps, flips = recall(J,jnp.asarray(padded),jax.random.PRNGKey(seed))
    recovered = np.asarray(out)[:n_query]
    overlaps = np.mean(recovered * patterns[:n_query], axis=1)
    flips_np = np.asarray(flips)[:n_query]
    return dict(N=n,P=P,alpha=P/n,trial=trial,seed=seed,
        unstable_fraction=unstable,mean_overlap=float(np.mean(overlaps)),
        recall_success=float(np.mean(overlaps >= .9)),
        unconverged=int(np.sum(flips_np != 0)),sweeps=int(sweeps),
        runtime_sec=round(time.perf_counter()-t,4))


def summarize(records, n_values, pct_values, trials):
    summaries = {}
    for n in n_values:
        rows=[]
        for pct in pct_values:
            block=[r for r in records if int(r['N'])==n and int(round(float(r['alpha'])*100))==pct]
            if len(block)!=trials:
                raise RuntimeError(f'Incomplete: N={n}, alpha={pct/100}: {len(block)}/{trials} trials')
            entry={'alpha': pct/100, 'N': n, 'P':n*pct//100,
                   'n_trials':len(block)}
            for key in ('unstable_fraction','mean_overlap','recall_success'):
                values=np.array([float(r[key]) for r in block])
                entry[key]=float(values.mean())
                entry[key+'_sem']=float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else 0.
            entry['unconverged_total']=int(sum(int(r['unconverged']) for r in block))
            rows.append(entry)
        summaries[n]=rows
    return summaries


def make_plots(summary):
    fig, axs=plt.subplots(1,2,figsize=(13,5))
    a=np.linspace(.05,.25,300)
    theory=np.array([.5*math.erfc(1/math.sqrt(2*x)) for x in a])
    for ax in axs:
        ax.plot(a,theory,'k--',linewidth=2,label='Gaussian large-N prediction')
        for n,rows in summary.items():
            x=[r['alpha'] for r in rows]
            y=[r['unstable_fraction'] for r in rows]
            sem=[r['unstable_fraction_sem'] for r in rows]
            ax.errorbar(x,y,yerr=sem,marker='o',markersize=3,capsize=2,label=f'N={n}')
        ax.set_xlabel('Load alpha = P/N')
        ax.set_ylabel('Fraction of unstable bits')
        ax.grid(alpha=.25)
    axs[0].set_title('Unstable fraction (linear scale)')
    axs[1].set_yscale('log')
    axs[1].set_ylim(1e-6,.1)
    axs[1].set_title('Unstable fraction (log scale)')
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(EVIDENCE/'part3_unstable_fraction.png',dpi=200,bbox_inches='tight')
    plt.close(fig)
    for measure,filename,title,ylabel in [
        ('mean_overlap','part3_final_overlap.png','Recall from stored patterns','Mean final overlap'),
        ('recall_success','part3_recall_success.png','Recall success rate','Fraction of recalls with overlap >= 0.9')]:
        fig,ax=plt.subplots(figsize=(9,5.5))
        for n,rows in summary.items():
            ax.errorbar([r['alpha'] for r in rows],[r[measure] for r in rows],
                 yerr=[r[measure+'_sem'] for r in rows],marker='o',markersize=4,
                 capsize=2,label=f'N={n}')
        ax.axvline(.138,color='black',linestyle='--',label='Critical load = 0.138')
        ax.set(xlabel='Load alpha = P/N',ylabel=ylabel,title=title,ylim=(0,1.05))
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(EVIDENCE/filename,dpi=200,bbox_inches='tight')
        plt.close(fig)
    print('Saved 3 plots in evidence/',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preview',action='store_true')
    args=parser.parse_args()
    n_values=(100,400) if args.preview else (100,200,400,800)
    pct_values=(5,15,25) if args.preview else tuple(range(5,26))
    trials=2 if args.preview else 20
    suffix='_preview' if args.preview else ''
    csv_path=ARTIFACTS/f'part3_capacity_runs{suffix}.csv'
    records=[]
    if csv_path.exists():
        with csv_path.open(newline='') as f:
            records=list(csv.DictReader(f))
        print(f'Resuming from {len(records)} saved trials: {csv_path}',flush=True)
    seen={(int(r['N']),int(round(float(r['alpha'])*100)),int(r['trial'])) for r in records}
    print(f'Plan: N={n_values}, alpha={pct_values}, repetitions={trials}',flush=True)
    all_start=time.perf_counter()
    with csv_path.open('a',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        if not records:
            writer.writeheader()
        for n in n_values:
            recall=make_recaller(n)
            for pct in pct_values:
                for trial in range(trials):
                    if (n,pct,trial) in seen:
                        continue
                    result=one_trial(n,pct,trial,recall)
                    writer.writerow(result)
                    f.flush()
                    records.append(result)
                block=[r for r in records if int(r['N'])==n and int(round(float(r['alpha'])*100))==pct]
                print(f'N={n:3d}, alpha={pct/100:.2f}, P={n*pct//100:3d}, '
                    f'overlap={np.mean([float(r["mean_overlap"]) for r in block]):.3f}, '
                    f'completed={len(block)}/{trials}',flush=True)
    summary=summarize(records,n_values,pct_values,trials)
    out=ARTIFACTS/f'part3_capacity_summary{suffix}.json'
    out.write_text(json.dumps({'seed_formula':'20261014 + N*10000 + 100*alpha_percent + trial',
        'max_sweeps':MAX_SWEEPS,'replicates':trials,'results':summary},indent=2))
    print('Total nonconverged recalls:',sum(int(r['unconverged']) for r in records),flush=True)
    print('Elapsed seconds:',round(time.perf_counter()-all_start,1),flush=True)
    if not args.preview:
        make_plots(summary)
    print('Saved summary:',out,flush=True)

if __name__=='__main__':
    main()
