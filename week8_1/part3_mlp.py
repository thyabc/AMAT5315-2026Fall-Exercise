#!/usr/bin/env python3
"""Week 8 Part 3: key-value memory capacity + one-token JAX MLP.

From inside week8_1/:
 uv run --no-project --with numpy --with matplotlib \
     python -u part3_mlp.py --only memory
 uv run --no-project --with numpy --with matplotlib --with 'jax[cpu]' --with sentencepiece \
     python -u part3_mlp.py --only mlp

Required existing inputs from Part 2: artifacts/part2_train_tokens.npy,
artifacts/part2_val_tokens.npy, artifacts/part2_summary.json.

The very large nonlinear memory stores approximate the median across ALL queries
using a random subset of stored queries to keep CPU work finite. All stored
memories participate in each selected query. This approximation is clearly
identified in reports; it is NOT a literal exhaustive query for M>1024.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

# Environment BEFORE loading NumPy/JAX avoids oversubscribing WSL CPU.
os.environ.setdefault('JAX_PLATFORM_NAME','cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
EVIDENCE = BASE / 'evidence'
ARTIFACTS = BASE / 'artifacts'
MEMORY_STORES = ('Outer product', 'Least squares', 'ReLU', 'ReLU^2', 'ReLU^3')
COLORS = {'Outer product':'#31688e','Least squares':'#35b779',
          'ReLU':'#f59e0b', 'ReLU^2':'#9467bd', 'ReLU^3':'#d54f58'}


def save_json(name, obj):
    p = ARTIFACTS / name
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    return p


def unit_vectors(rng, n, d):
    x = rng.normal(size=(n,d)).astype(np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def memory_one_draw(d, m, seed, max_queries=128):
    """Reproducible, direct recall with full store and sampled query keys for big M."""
    rng = np.random.default_rng(seed)
    k = unit_vectors(rng, m, d)
    v = unit_vectors(rng, m, d)
    # Exact over all stored keys for M <= 1024, representative key subset above.
    qi = (np.arange(m) if m <= 1024
          else np.sort(rng.choice(m, size=min(max_queries, m), replace=False)))
    q = k[qi]
    expected = v[qi]
    norm = np.linalg.norm(expected, axis=1)

    # Linear outer-product memory, exact direct formula.
    out = q @ (k.T @ v)
    r_outer = float(np.median(np.linalg.norm(out-expected,axis=1)/norm))

    # Least-squares linear memory, min_W ||K W - V||_F^2.
    # Using normal equations only in the overdetermined case reduces memory use.
    if m <= d:
        # H=K K^+ is the identity on linearly independent training keys.
        r_ls = 0.0
    else:
        inverse_gram = np.linalg.solve(k.T @ k, np.eye(d))
        weights = inverse_gram @ (k.T @ v)
        out = q @ weights
        # LS recall contains h_jj*v_j plus cross-talk from all other values.
        # Normalize by the *self coefficient* h_jj, not by an assumed 1.
        # This produces the theory R = sqrt(M/d - 1), capacity near 2d.
        h_self = np.einsum('id,id->i', q @ inverse_gram, q)
        leak = out - h_self[:,None]*expected
        r_ls = float(np.median(np.linalg.norm(leak,axis=1)/(h_self*norm)))

    # The query's own contribution has coefficient 1, so exclude it exactly.
    r_nonlin = {1:[], 2:[], 3:[]}
    # Chunk query blocks to bound temporary memory for M=65536.
    for start in range(0, len(qi), 64):
        stop = min(start+64,len(qi))
        sims = (q[start:stop] @ k.T)
        relu = np.maximum(sims,0.0)
        rowids = np.arange(stop-start)
        relu[rowids,qi[start:stop]]=0.0
        for power in (1,2,3):
            weights = relu if power == 1 else relu ** power
            leakage = weights @ v
            ratios = np.linalg.norm(leakage,axis=1) / norm[start:stop]
            r_nonlin[power].extend(ratios.tolist())
    del k,v
    return {
        'Outer product':r_outer, 'Least squares':r_ls,
        'ReLU':float(np.median(r_nonlin[1])),
        'ReLU^2':float(np.median(r_nonlin[2])),
        'ReLU^3':float(np.median(r_nonlin[3])),
        '_query_count':int(len(qi)), '_query_sampling':bool(len(qi)!=m),
    }


def capacity_crossing(ms, ratios):
    ms=np.asarray(ms,dtype=np.float64)
    rr=np.asarray(ratios,dtype=np.float64)
    idx=np.where(rr>=1)[0]
    if len(idx)==0:
        return None
    i=int(idx[0])
    if i==0:
        return None
    # Fit a straight segment on log M vs. log interference.
    y0,y1=np.log(max(rr[i-1],1e-30)),np.log(max(rr[i],1e-30))
    x0,x1=np.log(ms[i-1]),np.log(ms[i])
    if not np.isfinite(y1-y0) or abs(y1-y0)<1e-13:
        return None
    return float(np.exp(x0+(0-y0)*(x1-x0)/(y1-y0)))


def run_memory(seed, max_queries=128, max_m=65536, draws=3):
    ms=[2**i for i in range(3,17) if 2**i<=max_m]
    if not ms:
        raise ValueError('Set --max-m >= 8')
    dimensions=[16,32,64]
    curves={}
    all_caps={name:[] for name in MEMORY_STORES}
    for d in dimensions:
        rows=[]
        print(f'\n=== Memory stores d={d} ===',flush=True)
        for m in ms:
            started=time.monotonic()
            run=[memory_one_draw(d,m,seed+10000*d+100*m+rep,
                                 max_queries=max_queries) for rep in range(draws)]
            entry={'d':d,'M':m,'num_draws':draws,
                   'queries_per_draw':run[0]['_query_count'],
                   'queries_are_subset':run[0]['_query_sampling']}
            for name in MEMORY_STORES:
                entry[name]=float(np.median([r[name] for r in run]))
            rows.append(entry)
            print(f"  M={m:6d}, queries={entry['queries_per_draw']:4d}; "
                  +'; '.join(f'{n}={entry[n]:.3g}' for n in MEMORY_STORES)
                  +f' [{time.monotonic()-started:.1f}s]',flush=True)
        curves[str(d)]=rows
        for name in MEMORY_STORES:
            cap=capacity_crossing(ms,[r[name] for r in rows])
            all_caps[name].append(cap)
            print(f'  capacity {name:16s}: {cap if cap is not None else "outside measured range"}',flush=True)

    # Graph 1: all measured R curves for d=64, plus theoretical predictions.
    fig,ax=plt.subplots(figsize=(9.5,6))
    vals=curves['64']
    m_arr=np.asarray(ms,dtype=float)
    for name in MEMORY_STORES:
        rr=np.asarray([p[name] for p in vals])
        ax.plot(m_arr, np.maximum(rr,1e-12), 'o-', ms=4, label=name, color=COLORS[name])
    theories={'Outer product':np.sqrt(m_arr/64),
              'Least squares':np.sqrt(np.maximum(m_arr/64-1,0)),
              'ReLU':np.sqrt(m_arr/(2*64)),
              'ReLU^2':np.sqrt(3*m_arr/(2*64**2)),
              'ReLU^3':np.sqrt(15*m_arr/(2*64**3))}
    for name,y in theories.items():
        ax.plot(m_arr,np.maximum(y,1e-12),'--',lw=1,alpha=.48,color=COLORS[name])
    ax.axhline(1,color='black',linestyle=':',lw=1,label='Capacity threshold R=1')
    ax.set(xscale='log',yscale='log',xlabel='Stored memories M',
           ylabel='Interference-to-signal ratio R (median)',
           title='Part 3: key–value memory interference, d=64')
    ax.set_ylim(1e-2,None)
    ax.legend(fontsize=8,ncol=2)
    ax.grid(alpha=.2,which='both')
    fig.tight_layout()
    img1=EVIDENCE/'part3_memory_interference.png'
    fig.savefig(img1,dpi=180)
    plt.close(fig)

    slopes={}
    fig,ax=plt.subplots(figsize=(9.5,6))
    for name in MEMORY_STORES:
        cap=all_caps[name]
        xs=np.array([d for d,c in zip(dimensions,cap) if c is not None],dtype=float)
        ys=np.array([c for c in cap if c is not None],dtype=float)
        if len(xs)<2:
            slopes[name]=None
            continue
        slope, intercept=np.polyfit(np.log(xs),np.log(ys),1)
        slopes[name]=float(slope)
        ax.loglog(xs,ys,'o-',color=COLORS[name],label=f'{name}: slope={slope:.2f}')
        ax.plot(xs,np.exp(intercept)*xs**slope,':',lw=1,color=COLORS[name])
    ax.set(xlabel='Key dimension d',ylabel='Capacity (R crosses 1)',
           title='Part 3: memory capacity scaling with dimension')
    ax.grid(alpha=.25,which='both')
    ax.legend(fontsize=9)
    fig.tight_layout()
    img2=EVIDENCE/'part3_capacity_scaling.png'
    fig.savefig(img2,dpi=180)
    plt.close(fig)

    summary={'dimensions':dimensions,'M_values':ms,'draws':draws,
             'max_queries':max_queries,'query_sampling_note':
             'Every stored key queried for M <= 1024; for larger M, a random subset '
             'of stored keys is queried against ALL M memories. This saves CPU time '
             'but is an approximation to the sheet\'s exhaustive querying requirement.',
             'curves':curves,
             'capacity':{name:{str(d):cap for d,cap in zip(dimensions,all_caps[name])}
                         for name in MEMORY_STORES},
             'slopes':slopes,
             'checks':{
                  'outer_R_crosses_d64': (all_caps['Outer product'][2] is not None and
                                           50<all_caps['Outer product'][2]<85),
                  'ls_R_crosses_d64': (all_caps['Least squares'][2] is not None and
                                       100<all_caps['Least squares'][2]<170),
                  'linear_slopes': all(slopes[name] is not None and abs(slopes[name]-1)<.1 for name in ('Outer product','Least squares','ReLU')),
                  'relu2_slope':(slopes['ReLU^2'] is not None and abs(slopes['ReLU^2']-2)<.15),
                  'relu3_slope':(slopes['ReLU^3'] is not None and slopes['ReLU^3']>2.5),
              }}
    save_json('part3_memory_summary.json',summary)
    print('\n=== Memory acceptance checks (approximate for large M) ===',flush=True)
    for name,status in summary['checks'].items():
        print(f'  {name}: {"PASS" if status else "CHECK"}',flush=True)
    print('Evidence:',img1,'and',img2,flush=True)
    return summary


def run_mlp(seed, steps=4000, batch_size=16384):
    import jax
    import jax.numpy as jnp

    t0=time.monotonic()
    for fname in ('part2_train_tokens.npy','part2_val_tokens.npy','part2_summary.json'):
        if not (ARTIFACTS/fname).exists():
            raise FileNotFoundError(
                f'Missing {ARTIFACTS/fname}. Please run Part 2 before Part 3.')
    train=np.load(ARTIFACTS/'part2_train_tokens.npy',mmap_mode='r')
    valid=np.load(ARTIFACTS/'part2_val_tokens.npy',mmap_mode='r')
    p2=json.loads((ARTIFACTS/'part2_summary.json').read_text(encoding='utf-8'))
    bigram_nats=float(next(row['nats_per_token'] for row in p2['n_grams'] if row['n']==2))
    eos=2  # detected from the pinned tok512.model on the original runs
    # Use the actual EOS id in original tokenizer, not an assumption.
    import sentencepiece as spm
    model_path=BASE/'data'/'stories260K'/'tok512.model'
    if not model_path.exists():
        raise FileNotFoundError(f'Missing tokenizer from Part 2: {model_path}')
    tokenizer=spm.SentencePieceProcessor(model_file=str(model_path))
    eos=tokenizer.eos_id()
    if not (0<=eos<512): raise ValueError('Invalid EOS token')

    train=np.asarray(train,dtype=np.int32)
    valid=np.asarray(valid,dtype=np.int32)
    prev_train=np.concatenate([[eos],train[:-1]]).astype(np.int32)
    prev_val=np.concatenate([[eos],valid[:-1]]).astype(np.int32)
    vocab=512
    # Dense 512x512 held-out joint counts, allowing evaluation without allocating
    # large batches of 512-vocab logits. Equivalent to evaluating all tokens.
    va_counts=np.bincount(prev_val*vocab+valid,minlength=vocab*vocab).reshape(vocab,vocab)
    val_count_sum=float(va_counts.sum())
    val_counts_j=jnp.asarray(va_counts,dtype=jnp.float32)
    xp=jnp.asarray(prev_train)
    yp=jnp.asarray(train)
    max_index=len(prev_train)
    print(f'Train pairs {max_index:,}; validation {len(valid):,}; '
          f'bigram baseline {bigram_nats:.5f} nats/token',flush=True)

    rng=np.random.default_rng(seed)
    params={
      'E':jnp.asarray(rng.normal(0,.11,(512,64)),dtype=jnp.float32),
      'W1':jnp.asarray(rng.normal(0,np.sqrt(2/576),(64,512)),dtype=jnp.float32),
      'b1':jnp.zeros((512,),dtype=jnp.float32),
      'W2':jnp.asarray(rng.normal(0,.025,(512,512)),dtype=jnp.float32),
      'b2':jnp.zeros((512,),dtype=jnp.float32),
    }

    def logits_all(parameters):
        hidden=jnp.tanh(parameters['E']@parameters['W1']+parameters['b1'])
        return hidden@parameters['W2']+parameters['b2']

    def loss_from_joint(p,joint):
        logp=jax.nn.log_softmax(logits_all(p),axis=1)
        return -jnp.sum(joint*logp) / jnp.maximum(jnp.sum(joint),1.)

    # AdamW with 200-step warmup + cosine decay implemented directly in JAX.
    # This avoids adding optax as a required dependency.
    beta1,beta2,eps,weight_decay=.9,.999,1e-8,.01
    m1=jax.tree_util.tree_map(jnp.zeros_like,params)
    m2=jax.tree_util.tree_map(jnp.zeros_like,params)
    state=(m1,m2,jnp.asarray(0,dtype=jnp.int32))

    def learning_rate(step):
        step=jnp.asarray(step,dtype=jnp.float32)
        warm=jnp.minimum(1.,step/200.)
        progress=jnp.clip((step-200.)/max(steps-200,1),0.,1.)
        cosine=.5*(1.+jnp.cos(jnp.pi*progress))
        return jnp.where(step<=200,.006*warm,
                         .00006+(.006-.00006)*cosine)

    # The random minibatch contains exactly 16,384 sample pairs. Collapsing
    # equal (previous,next) pairs to joint counts gives the IDENTICAL average
    # cross entropy and gradient at far lower CPU cost.
    @jax.jit
    def train_step(p,optim_state,random_key):
        old_m,old_v,t=optim_state
        ids=jax.random.randint(random_key,(batch_size,),minval=0,maxval=max_index)
        packed=xp[ids]*vocab+yp[ids]
        pair_hist=jnp.bincount(packed,length=vocab*vocab).reshape((vocab,vocab))
        loss,grads=jax.value_and_grad(loss_from_joint)(p,pair_hist)
        next_t=t+1
        new_m=jax.tree_util.tree_map(lambda a,g:beta1*a+(1-beta1)*g,old_m,grads)
        new_v=jax.tree_util.tree_map(lambda a,g:beta2*a+(1-beta2)*g*g,old_v,grads)
        corr1=1.-beta1**next_t
        corr2=1.-beta2**next_t
        lr=learning_rate(next_t)
        new_p=jax.tree_util.tree_map(
            lambda a,ma,va:a-lr*((ma/corr1)/(jnp.sqrt(va/corr2)+eps)+weight_decay*a),
            p,new_m,new_v)
        return new_p,(new_m,new_v,next_t),loss

    @jax.jit
    def eval_loss(p):
        return loss_from_joint(p,val_counts_j)

    key=jax.random.PRNGKey(seed)
    history=[]
    print('JAX device:',jax.devices(),flush=True)
    val0=float(eval_loss(params))
    history.append({'step':0,'validation_nats':val0})
    print(f'  step=0 validation_nats={val0:.5f}',flush=True)
    for step in range(1,steps+1):
        key,subkey=jax.random.split(key)
        params,state,loss=train_step(params,state,subkey)
        if step%200==0 or step==steps:
            vl=float(eval_loss(params))
            tl=float(loss)
            history.append({'step':step,'validation_nats':vl,'batch_nats':tl})
            print(f'  step={step:4d}/{steps} train_batch={tl:.5f} '
                  f'validation={vl:.5f}; elapsed={time.monotonic()-t0:.1f}s',flush=True)
    final_nats=history[-1]['validation_nats']
    diff=final_nats-bigram_nats
    accepted=abs(diff)<.01

    fig,ax=plt.subplots(figsize=(8.6,5.1))
    ax.plot([h['step'] for h in history],[h['validation_nats'] for h in history],
            'o-',ms=3,label='One-token MLP validation')
    ax.axhline(bigram_nats,ls='--',c='grey',label=f'Part 2 bigram baseline: {bigram_nats:.4f}')
    ax.set(xlabel='AdamW training step',ylabel='Held-out cross entropy (nats/token)',
           title='Part 3: one-token MLP vs. bigram')
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    plot=EVIDENCE/'part3_mlp_training_curve.png'
    fig.savefig(plot,dpi=180)
    plt.close(fig)

    summary={'architecture':{'embedding_dim':64,'hidden_units':512,
                              'activation':'tanh','vocab_size':512},
             'optimizer':'AdamW','peak_lr':.006,'warmup_steps':200,
             'steps':steps,'batch_size':batch_size,'seed':seed,
             'validation_nats_per_token':final_nats,'bigram_nats_per_token':bigram_nats,
             'difference_vs_bigram':diff,'within_0.01':bool(accepted),
             'history':history,'training_time_seconds':time.monotonic()-t0,
             'note':'Minibatch sampled with replacement; equal (previous,next) samples '
                    'are exactly aggregated into joint histograms for efficient CPU training.'}
    save_json('part3_mlp_summary.json',summary)
    np.savez_compressed(ARTIFACTS/'part3_mlp_params.npz',
                        **{name:np.asarray(arr) for name,arr in params.items()})
    print(f'\nMLP final held-out nats/token: {final_nats:.5f}; '
          f'Part 2 bigram: {bigram_nats:.5f}; delta: {diff:+.5f}; '
          f'{"PASS" if accepted else "CHECK"}',flush=True)
    print('Evidence:',plot,flush=True)
    return summary


def update_readme(memory, mlp):
    start='<!-- BEGIN WEEK8 PART3 -->'
    end='<!-- END WEEK8 PART3 -->'
    content=[start,'## Part 3 — The MLP: a memory in the weights','',
             'Re-run inside `week8_1/` using Python/uv and CPU JAX:',
             '```bash',
             "uv run --no-project --with numpy --with matplotlib --with 'jax[cpu]' --with sentencepiece --with sentencepiece python -u part3_mlp.py --only memory",
             "uv run --no-project --with numpy --with matplotlib --with 'jax[cpu]' --with sentencepiece --with sentencepiece python -u part3_mlp.py --only mlp",
             '```','']
    if memory:
        content.extend(['**Memory stores:** Three random draws for each d=16,32,64 '
                        'and M=8,...,65536 (powers of two). For M>1024 only a '
                        'random subset of stored query keys is evaluated against all '
                        'stored memories, an explicit CPU-saving approximation to '
                        'the sheet\'s exhaustive query requirement. Results at smaller M '
                        'query all keys. Interference R is the norm of the leakage '
                        'from other memories divided by signal.','',
                        '| Store | d=64 capacity | Scaling exponent |',
                        '|---|---:|---:|'])
        for name in MEMORY_STORES:
            capacity=memory['capacity'][name]['64']
            slope=memory['slopes'][name]
            content.append(f'| {name} | {capacity:.1f} | {slope:.3f} |' if capacity is not None and slope is not None else f'| {name} | N/A | N/A |')
        content.extend(['','Evidence: `evidence/part3_memory_interference.png`, '
                        '`evidence/part3_capacity_scaling.png`.'])
    if mlp:
        content.extend(['',f"**One-token MLP:** 512-token vocabulary, embedding 64, "
                        f"tanh hidden layer 512; {mlp['steps']} AdamW steps, "
                        f"batch size {mlp['batch_size']}, learning rate peak 0.006, "
                        f"cosine decay and warmup.",
                        f"Held-out loss: **{mlp['validation_nats_per_token']:.5f}** "
                        f"nats/token, Bigram baseline: **{mlp['bigram_nats_per_token']:.5f}**, "
                        f"difference: {mlp['difference_vs_bigram']:+.5f}. "
                        f"Threshold ±0.01: {'PASS' if mlp['within_0.01'] else 'CHECK'}.",
                        'Evidence: `evidence/part3_mlp_training_curve.png`.'])
    content.extend(['','Raw metrics are in `artifacts/part3_*.json` (not for Git).',end])
    block='\n'.join(content)
    f=BASE/'README.md'
    old=f.read_text(encoding='utf-8') if f.exists() else '# Week 8\n'
    if start in old and end in old:
        pre, tail=old.split(start,1)
        _,post=tail.split(end,1)
        new=pre+block+post
    else:
        new=old.rstrip()+'\n\n'+block+'\n'
    f.write_text(new,encoding='utf-8')
    print('Updated README:',f,flush=True)


def main():
    p=argparse.ArgumentParser(description='Week 8 Part 3 key-value memory and JAX MLP')
    p.add_argument('--only',choices=['memory','mlp','both'],default='both')
    p.add_argument('--seed',type=int,default=1730)
    p.add_argument('--max-queries',type=int,default=128,
                   help='Number of stored keys sampled as queries for M>1024; default 128')
    p.add_argument('--max-m',type=int,default=65536)
    p.add_argument('--draws',type=int,default=3)
    p.add_argument('--steps',type=int,default=4000)
    p.add_argument('--batch-size',type=int,default=16384)
    args=p.parse_args()
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    memo=run_memory(args.seed,args.max_queries,args.max_m,args.draws) if args.only in ('memory','both') else None
    mlp=run_mlp(args.seed,args.steps,args.batch_size) if args.only in ('mlp','both') else None
    # Preserve already computed portions from earlier --only calls.
    if memo is None and (ARTIFACTS/'part3_memory_summary.json').exists():
        memo=json.loads((ARTIFACTS/'part3_memory_summary.json').read_text(encoding='utf-8'))
    if mlp is None and (ARTIFACTS/'part3_mlp_summary.json').exists():
        mlp=json.loads((ARTIFACTS/'part3_mlp_summary.json').read_text(encoding='utf-8'))
    update_readme(memo,mlp)
    print('Part 3 tasks completed. Inspect PASS/CHECK results above.',flush=True)

if __name__=='__main__':
    main()
