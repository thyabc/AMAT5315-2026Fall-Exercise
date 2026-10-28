"""JAX/Optax training with a numeric, atomic full-state checkpoint."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from model import initialize, forward
from prepare import prepare
from eval import evaluate


def atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **{k: np.asarray(v) for k, v in arrays.items()})
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_log(path, rows):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['step','train_loss','val_loss','tokens','wall_seconds','lr'])
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def learning_rate(config):
    return optax.warmup_cosine_decay_schedule(0., config['learning_rate'],
        config['warmup_steps'], config['max_steps'], config['min_lr'])


def optimizer(config):
    schedule = learning_rate(config)
    return optax.chain(optax.clip_by_global_norm(1.), optax.adamw(schedule,
                      b1=0.9, b2=0.99, weight_decay=config['weight_decay']))


def save_state(path, params, opt_state, key, step, elapsed, config):
    adam = opt_state[1][0]
    arrays = {f'p::{name}': value for name, value in params.items()}
    arrays.update({f'mu::{name}': value for name, value in adam.mu.items()})
    arrays.update({f'nu::{name}': value for name, value in adam.nu.items()})
    arrays.update(step=np.array(step), key=key, elapsed_seconds=np.array(elapsed),
                  config_json=np.frombuffer(json.dumps(config,sort_keys=True).encode(),dtype=np.uint8))
    atomic_npz(path, arrays)


def load_state(path, tx, config):
    with np.load(path, allow_pickle=False) as source:
        saved_config=json.loads(source['config_json'].tobytes())
        if saved_config != config:
            raise ValueError('Resume configuration differs from saved training configuration')
        params = {name[3:]: jnp.asarray(source[name]) for name in source.files if name.startswith('p::')}
        state = tx.init(params)
        step = int(source['step'])
        adam = state[1][0]._replace(count=jnp.asarray(step, dtype=jnp.int32),
            mu={name: jnp.asarray(source['mu::'+name]) for name in params},
            nu={name: jnp.asarray(source['nu::'+name]) for name in params})
        schedule = state[1][2]._replace(count=jnp.asarray(step, dtype=jnp.int32))
        state = (state[0], (adam, state[1][1], schedule))
        return params, state, jnp.asarray(source['key']), step, float(source['elapsed_seconds'])


def export_model(out, params, config, meta, step):
    atomic_npz(out / 'model.npz', params)
    architecture = {k: config[k] for k in ['vocab_size','block_size','n_layer','n_head','n_embd','dropout']}
    architecture.update(layer_norm_epsilon=1e-5, gelu='exact')
    info = dict(schema_version=1, architecture=architecture, vocabulary=meta['vocabulary'],
                corpus_sha256=meta['corpus_sha256'], step=step,
                checkpoint_sha256=hashlib.sha256((out/'model.npz').read_bytes()).hexdigest(),
                parameter_count=sum(p.size for p in params.values()))
    (out/'model.json').write_text(json.dumps(info, indent=2)+'\n')


def train(config, out, data='data', stop_step=None, resume=None, reset=None, evaluate_model=True):
    started = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    training, validation, meta = prepare(data)
    tx = optimizer(config)
    key = jax.random.PRNGKey(config['seed'])
    key, init_key = jax.random.split(key)
    params = initialize(config, init_key)
    state = tx.init(params)
    step, previous_elapsed = 0, 0.
    if resume:
        params, state, key, step, previous_elapsed = load_state(resume, tx, config)
        if reset == 'optimizer':
            # Negative control preserves schedule position, but loses Adam's moments.
            fresh = tx.init(params)
            state = (state[0], (fresh[1][0], state[1][1], state[1][2]))
        elif reset == 'key':
            key = jax.random.PRNGKey(config['seed'])
    training = jnp.asarray(training, dtype=jnp.int32)
    @jax.jit
    def update(params, state, key):
        key, batch_key = jax.random.split(key)
        starts = jax.random.randint(batch_key, (config['batch_size'],), 0, len(training)-128)
        x = training[starts[:, None]+jnp.arange(128)]
        y = training[starts[:, None]+jnp.arange(1,129)]
        def objective(p):
            return optax.softmax_cross_entropy_with_integer_labels(forward(p, x, config), y).mean()
        loss, grad = jax.value_and_grad(objective)(params)
        changes, state = tx.update(grad, state, params)
        return optax.apply_updates(params, changes), state, key, loss
    log_path = out/'train_log.csv'
    rows = []
    if resume and log_path.exists():
        with log_path.open() as stream:
            rows = [row for row in csv.DictReader(stream) if int(row['step'])<=step]
    stop = config['max_steps'] if stop_step is None else stop_step
    if not step <= stop <= config['max_steps']:
        raise ValueError('stop-step must lie between resumed step and max_steps')
    first_seconds = None
    schedule = learning_rate(config)
    losses = []
    for step in range(step+1, stop+1):
        before = time.perf_counter()
        params, state, key, loss = update(params, state, key)
        loss = float(loss)  # Synchronizes the device before recording elapsed time.
        losses.append(loss)
        if first_seconds is None:
            first_seconds = time.perf_counter()-before
        if step % config['eval_interval'] == 0 or step == stop or step == 1:
            score = evaluate(params, config, validation) if evaluate_model else {'loss': float('nan')}
            elapsed = previous_elapsed + time.perf_counter()-started
            rows.append(dict(step=step,train_loss=loss,val_loss=score['loss'],
                             tokens=step*config['batch_size']*128,wall_seconds=elapsed,lr=float(schedule(step-1))))
            # Commit the log first. Resume drops rows beyond the last saved step.
            write_log(log_path, rows)
            save_state(out/'train-state/state.npz',params,state,key,step,elapsed,config)
            print(f"step {step}: train={loss:.5f}, val={score['loss']:.5f}, wall={elapsed:.1f}s", flush=True)
    write_log(log_path, rows)
    export_model(out, params, config, meta, stop)
    elapsed = previous_elapsed + time.perf_counter()-started
    save_state(out/'train-state/state.npz', params, state, key, stop, elapsed, config)
    elapsed = previous_elapsed + time.perf_counter()-started
    # The reported full run cost includes final checkpoint serialization.
    info = dict(config, device=str(jax.devices()[0]), device_kind=jax.devices()[0].device_kind,
                memory_stats=jax.devices()[0].memory_stats(), precision='float32',
                matmul_precision='highest', jax_version=jax.__version__,
                optax_version=optax.__version__, elapsed_seconds=elapsed,
                first_update_seconds=first_seconds)
    (out/'config.json').write_text(json.dumps(info, indent=2)+'\n')
    (out/'trace.json').write_text(json.dumps(dict(losses=losses, stop_step=stop), indent=2)+'\n')
    return info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--out', default='artifacts')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--stop-step',type=int)
    parser.add_argument('--resume')
    parser.add_argument('--reset',choices=['optimizer','key'])
    parser.add_argument('--no-eval',action='store_true')
    args=parser.parse_args()
    print(train(json.loads(Path(args.config).read_text()),args.out,args.data_dir,args.stop_step,args.resume,args.reset,not args.no_eval))
