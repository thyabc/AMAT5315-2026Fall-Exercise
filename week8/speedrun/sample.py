"""Generate two temperatures from one unchanged exported checkpoint."""
import argparse
import json
from pathlib import Path
import jax
import numpy as np
from eval import load_model
from model import forward


def sample(directory='artifacts', prompt='To be', seed=7, length=256, temperatures=(0.7, 1.1)):
    params, meta = load_model(directory)
    vocabulary = meta['vocabulary']
    if not prompt or any(ch not in vocabulary for ch in prompt):
        raise ValueError('Prompt must be nonempty and contain only corpus characters')
    if length < 0 or not 0 <= seed < 2**32:
        raise ValueError('Length must be nonnegative and seed must fit uint32')
    if not temperatures or any(not np.isfinite(t) or t <= 0 for t in temperatures):
        raise ValueError('Temperatures must be finite and positive')
    @jax.jit
    def predict(ids, index):
        return forward(params, ids, meta['architecture'])[0, index]
    records=[]
    for temperature in temperatures:
        ids=[vocabulary.index(ch) for ch in prompt]
        key=jax.random.PRNGKey(seed)
        for _ in range(length):
            context=ids[-128:]
            padded=np.zeros((1,128),np.int32)
            padded[0,:len(context)]=context
            logits=predict(padded,len(context)-1)
            key, subkey=jax.random.split(key)
            ids.append(int(jax.random.categorical(subkey, logits/temperature)))
        settings=dict(prompt=prompt, seed=seed, temperature=temperature, length=length,
                      prompt_length=len(prompt), total_length=len(ids),
                      checkpoint_sha256=meta['checkpoint_sha256'])
        records.append(dict(settings=settings, text=''.join(vocabulary[i] for i in ids)))
    (Path(directory)/'sample.txt').write_text('\n\n'.join(
        json.dumps(record['settings'])+'\n'+record['text'] for record in records)+'\n')
    return records


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--artifacts',default='artifacts')
    parser.add_argument('--data-dir',default='data')
    parser.add_argument('--prompt',default='To be')
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--length','--num',dest='length',type=int,default=256)
    parser.add_argument('--temperature',type=float,nargs='+',default=[0.7,1.1])
    args=parser.parse_args()
    sample(args.artifacts,prompt=args.prompt,seed=args.seed,length=args.length,temperatures=args.temperature)
