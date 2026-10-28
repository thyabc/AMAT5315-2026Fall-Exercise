"""Regenerate evidence from saved weights; full training is a separate command."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import jax
import jax.numpy as jnp
import numpy as np
from eval import metrics, load_model, save_evaluation
from model import initialize, forward
from prepare import prepare
from train import atomic_npz

HERE=Path(__file__).resolve().parent


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def baselines(out, data):
    training,validation,meta=prepare(data)
    probabilities=(np.bincount(training,minlength=65)+1)/(len(training)+65)
    counts=np.bincount(validation[1:],minlength=65)
    uniform=metrics((len(validation)-1)*np.log(65),len(validation)-1)
    frequency=metrics(-np.dot(counts,np.log(probabilities)),len(validation)-1)
    example=metrics(-np.log(.5)-np.log(.25),2)
    assert np.isclose(uniform['ppl'],65)
    assert np.isclose(example['ppl'],np.sqrt(8))
    # Unequal blocks must be combined by token count before exponentiating.
    assert np.isclose(metrics(np.log(2)+2*np.log(4),3)['ppl'],32**(1/3))
    result=dict(vocabulary=meta['vocabulary'],train_probabilities=probabilities.tolist(),
                validation_counts=counts.tolist(), uniform=uniform, frequency=frequency,
                two_target_example=example, corpus_sha256=meta['corpus_sha256'],
                decoded_input=''.join(meta['vocabulary'][i] for i in training[:24]),
                decoded_target=''.join(meta['vocabulary'][i] for i in training[1:25]))
    write_json(out/'baselines.json',result)
    return result


def probe(out, data):
    _,_,meta=prepare(data)
    if (out/'model.npz').exists():
        params, model_meta=load_model(out)
        config=model_meta['architecture']
    else:
        config=json.loads((HERE/'config-tiny.json').read_text())
        params=initialize(config,jax.random.PRNGKey(7))
    text='To be, or not to be'
    ids=np.array([[meta['vocabulary'].index(ch) for ch in text]],np.int32)
    changed=ids.copy()
    prefix=7
    changed[:,prefix:]=(changed[:,prefix:]+11)%65
    logits,attention=forward(params,ids,config,return_attention=True)
    logits_b=forward(params,changed,config)
    error=float(jnp.max(jnp.abs(logits[:,:prefix]-logits_b[:,:prefix])))
    probabilities=np.asarray(jax.nn.softmax(logits,axis=-1))
    atomic_npz(out/'causality.npz',dict(inputs_a=ids,inputs_b=changed,logits_a=logits,logits_b=logits_b))
    result=dict(input=text,attention=np.asarray(attention[0,0]).tolist(),
                next_probabilities=probabilities[0,-1].tolist(),vocabulary=meta['vocabulary'],
                prefix_length=prefix,max_prefix_error=error,prefix_error=error,
                max_normalization_error=float(np.max(np.abs(probabilities.sum(axis=-1)-1))),
                max_future_attention=float(np.max(np.triu(np.asarray(attention),1))))
    result.update(checkpoint_sha256=model_meta['checkpoint_sha256'] if (out/'model.npz').exists() else None,
                  initialization_seed=None if (out/'model.npz').exists() else 7)
    assert error < 1e-5 and result['max_future_attention']==0
    write_json(out/'causality.json',result)
    return result


def resume_check(out, data, negative_controls=False):
    directory=out/'resume'
    directory.mkdir(exist_ok=True)
    config=json.loads((HERE/'config-tiny.json').read_text())
    config.update(max_steps=12,eval_interval=12,warmup_steps=2)
    write_json(directory/'config.json',config)
    def run(name,*extra):
        subprocess.run([sys.executable,str(HERE/'train.py'),str(directory/'config.json'),
            '--out',str(directory/name),'--data-dir',str(data),'--no-eval',*extra],
            check=True,env=dict(os.environ,JAX_PLATFORMS='cpu'))
    run('uninterrupted')
    run('interrupted','--stop-step','6')
    checkpoint=str(directory/'interrupted/train-state/state.npz')
    run('resumed','--resume',checkpoint)
    names=['uninterrupted','resumed']
    if negative_controls:
        run('missing-optimizer','--resume',checkpoint,'--reset','optimizer')
        run('missing-key','--resume',checkpoint,'--reset','key')
        names += ['missing-optimizer','missing-key']
    losses={}
    arrays={}
    for name in names:
        source=directory/name/'train-state/state.npz'
        shutil.copyfile(source,directory/(name+'.npz'))
        with np.load(source,allow_pickle=False) as archive:
            arrays[name]={k:archive[k] for k in archive.files}
        losses[name.replace('-','_')]=np.array(json.loads((directory/name/'trace.json').read_text())['losses'][-6:])
    atomic_npz(directory/'losses.npz',losses)
    def error(name,prefix):
        return max(float(np.max(np.abs(arrays['uninterrupted'][k]-arrays[name][k])))
                   for k in arrays[name] if k.startswith(prefix))
    result=dict(steps=list(range(7,13)),uninterrupted_losses=losses['uninterrupted'].tolist(),
        resumed_losses=losses['resumed'].tolist(),
        max_loss_error=float(np.max(np.abs(losses['uninterrupted']-losses['resumed']))),
        max_parameter_error=error('resumed','p::'),
        max_optimizer_error=max(error('resumed','mu::'),error('resumed','nu::')),
        negative_controls={name:dict(max_parameter_error=error(name,'p::'),
            max_loss_error=float(np.max(np.abs(losses['uninterrupted']-losses[name.replace('-','_')]))))
            for name in names[2:]},
        backend='cpu',configuration=config,split_step=6,total_steps=12)
    assert max(result['max_loss_error'],result['max_parameter_error'],result['max_optimizer_error']) < 1e-6
    assert all(control['max_parameter_error']>1e-4 for control in result['negative_controls'].values())
    write_json(out/'resume.json',result)
    return result


def compare(out,data):
    baseline=out/'comparison/baseline'
    baseline.mkdir(parents=True,exist_ok=True)
    for filename in ['model.npz','model.json','config.json','train_log.csv','eval.json']:
        shutil.copyfile(out/filename,baseline/filename)
    for directory in sorted((out/'comparison').iterdir()):
        if directory.is_dir() and (directory/'model.npz').exists():
            save_evaluation(directory,data)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['baselines','probe','compare','resume','all'],nargs='?',default='all')
    parser.add_argument('--artifacts',default='artifacts')
    parser.add_argument('--data-dir',default='data')
    parser.add_argument('--negative-controls',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    out,data=Path(args.artifacts).resolve(),Path(args.data_dir).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if args.command in ['baselines','all']: baselines(out,data)
    if args.command in ['probe','all']: probe(out,data)
    if args.command=='all':
        save_evaluation(out,data)
        from sample import sample
        sample(out)
    if args.command in ['compare','all']: compare(out,data)
    if args.command in ['resume','all']: resume_check(out,data,args.negative_controls)
    from plots import render_all
    render_all(out)


if __name__=='__main__':
    main()
