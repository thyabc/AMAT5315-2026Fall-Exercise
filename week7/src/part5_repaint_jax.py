import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from part4_cifar10_ddpm_jax import (
    ROOT, EVIDENCE, ARTIFACTS, STEPS, BETA_START, BETA_END,
    load_predictor, forward_unet, to_uint8, save_montage,
)

INPUTS = ROOT / "inputs"
SEEDS = [7201, 7202, 7203, 7204]
H0, H1 = 6, 26
W0, W1 = 6, 26


def alpha_bar(level, alpha_bars):
    if level == 0:
        return jnp.array(1.0, dtype=jnp.float32)
    return alpha_bars[level - 1]


def predict_eps_jit(cfg):
    def fn(x, t, params):
        return forward_unet(x, t, params, cfg)
    return jax.jit(fn)


def repaint_single(obs, truth, mask, seed, predict_eps, params, alpha_bars):
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(key, truth.shape, dtype=jnp.float32)

    hidden = (1.0 - mask)
    traces = {1000: np.asarray(x[0])}
    evals = 0

    def merge_with_observation(x_prop, tprime, key):
        abar_tp = alpha_bar(tprime, alpha_bars)
        if tprime == 0:
            y_tp = obs
        else:
            key, noise_key = jax.random.split(key)
            noise = jax.random.normal(noise_key, obs.shape, dtype=jnp.float32)
            y_tp = jnp.sqrt(abar_tp) * obs + jnp.sqrt(1.0 - abar_tp) * noise
        merged = mask * y_tp + hidden * x_prop
        return merged, key

    for t in range(1000, 20, -20):  # 1000,980,...,40
        tprime = t - 20
        abar_t = alpha_bar(t, alpha_bars)
        abar_tp = alpha_bar(tprime, alpha_bars)
        a = abar_t / abar_tp
        for attempt in (1, 2):
            if attempt == 2:
                key, noise_key = jax.random.split(key)
                f = jax.random.normal(noise_key, x.shape, dtype=jnp.float32)
                x = jnp.sqrt(a) * x + jnp.sqrt(1.0 - a) * f
            eps = predict_eps(x, jnp.array([t - 1], dtype=jnp.float32), params)
            evals += 1
            x0_hat = (x - jnp.sqrt(1.0 - abar_t) * eps) / jnp.sqrt(abar_t)
            x0_hat = jnp.clip(x0_hat, -1.0, 1.0)
            coef_x0 = jnp.sqrt(abar_tp) * (1.0 - a) / (1.0 - abar_t)
            coef_xt = jnp.sqrt(a) * (1.0 - abar_tp) / (1.0 - abar_t)
            mu = coef_x0 * x0_hat + coef_xt * x
            var = (1.0 - abar_tp) * (1.0 - a) / (1.0 - abar_t)
            key, noise_key = jax.random.split(key)
            z = jax.random.normal(noise_key, x.shape, dtype=jnp.float32)
            x_prop = mu + jnp.sqrt(jnp.maximum(var, 0.0)) * z
            x, key = merge_with_observation(x_prop, tprime, key)
        if tprime in (800, 600, 400, 200, 100, 20):
            x.block_until_ready()
            traces[tprime] = np.asarray(x[0])

    # final jump 20 -> 0, once only
    t, tprime = 20, 0
    abar_t = alpha_bar(t, alpha_bars)
    abar_tp = alpha_bar(tprime, alpha_bars)
    a = abar_t / abar_tp  # here abar_tp = 1, so a = abar_t
    eps = predict_eps(x, jnp.array([t - 1], dtype=jnp.float32), params)
    evals += 1
    x0_hat = (x - jnp.sqrt(1.0 - abar_t) * eps) / jnp.sqrt(abar_t)
    x0_hat = jnp.clip(x0_hat, -1.0, 1.0)
    coef_x0 = jnp.sqrt(abar_tp) * (1.0 - a) / (1.0 - abar_t)
    coef_xt = jnp.sqrt(a) * (1.0 - abar_tp) / (1.0 - abar_t)
    mu = coef_x0 * x0_hat + coef_xt * x
    x = mask * obs + hidden * mu
    x.block_until_ready()
    traces[0] = np.asarray(x[0])

    completed = np.asarray(x[0], dtype=np.float32)
    truth_np = np.asarray(truth[0], dtype=np.float32)
    hidden_mask_np = np.asarray(hidden[0], dtype=np.float32)
    mse = float(np.mean(((completed - truth_np) * hidden_mask_np) ** 2) / np.mean(hidden_mask_np))
    return completed, mse, traces, evals


def save_trace(traces, out_path):
    levels = sorted(traces.keys(), reverse=True)
    pics = [to_uint8(np.expand_dims(traces[level], 0))[0] for level in levels]
    strip = np.concatenate(pics, axis=1)
    Image.fromarray(strip).resize((strip.shape[1]*2, strip.shape[0]*2), Image.Resampling.NEAREST).save(out_path)


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    data = np.load(INPUTS / 'images.npz')
    images = data['test'].astype(np.float32)

    mask = np.ones_like(images, dtype=np.float32)
    mask[:, :, H0:H1, W0:W1] = 0.0
    observed = images * mask

    params, cfg = load_predictor()
    predict_eps = predict_eps_jit(cfg)
    # one compile
    dummy = jnp.asarray(images[:1])
    _ = predict_eps(dummy, jnp.array([999], dtype=jnp.float32), params).block_until_ready()

    betas = jnp.linspace(BETA_START, BETA_END, STEPS, dtype=jnp.float32)
    alphas = 1.0 - betas
    alpha_bars = jnp.cumprod(alphas)

    # Save masked observations grid
    obs_grid = to_uint8(observed)
    save_montage(obs_grid, EVIDENCE / 'part5_observed_grid.png', columns=2)

    records = []
    all_outputs = []
    per_image_outputs = {i: [] for i in range(4)}
    started = time.time()

    for img_idx in range(4):
        print(f'Image {img_idx+1}/4 ...', flush=True)
        obs = jnp.asarray(observed[img_idx:img_idx+1])
        truth = jnp.asarray(images[img_idx:img_idx+1])
        m = jnp.asarray(mask[img_idx:img_idx+1])
        for seed in SEEDS:
            print(f'  seed={seed} ...', flush=True)
            completed, mse, traces, evals = repaint_single(obs, truth, m, seed, predict_eps, params, alpha_bars)
            all_outputs.append(completed)
            per_image_outputs[img_idx].append(completed)
            out_png = EVIDENCE / f'part5_img{img_idx+1}_seed{seed}_completion.png'
            Image.fromarray(to_uint8(np.expand_dims(completed,0))[0]).save(out_png)
            save_trace(traces, EVIDENCE / f'part5_img{img_idx+1}_seed{seed}_trace.png')
            records.append({
                'image_index': img_idx + 1,
                'seed': seed,
                'hidden_pixel_mse': mse,
                'network_evaluations': evals,
                'output_png': str(out_png.relative_to(ROOT)),
            })

    all_outputs = np.stack(all_outputs)
    save_montage(to_uint8(all_outputs), EVIDENCE / 'part5_repaint_all_completions_grid.png', columns=4)

    # Pixelwise variance across 4 completions per image
    variance_summaries = []
    for img_idx in range(4):
        arr = np.stack(per_image_outputs[img_idx], axis=0)  # 4,3,32,32
        varmap = np.var(arr, axis=0).mean(axis=0)  # 32,32 average over channels
        vmax = float(max(varmap.max(), 1e-8))
        norm = np.clip(varmap / vmax, 0.0, 1.0)
        heat = np.round(norm * 255.0).astype(np.uint8)
        Image.fromarray(heat).resize((32*8, 32*8), Image.Resampling.NEAREST).save(
            EVIDENCE / f'part5_img{img_idx+1}_variance_heatmap.png')
        variance_summaries.append({
            'image_index': img_idx + 1,
            'mean_pixel_variance': float(varmap.mean()),
            'max_pixel_variance': float(varmap.max()),
        })

    summary = {
        'task': 'Week7 Part5 RePaint inpainting',
        'mask_rows': [H0, H1],
        'mask_cols': [W0, W1],
        'seeds': SEEDS,
        'jump_levels': list(range(1000, -1, -20)),
        'network_evaluations_per_completion': 99,
        'records': records,
        'variance_summary': variance_summaries,
        'elapsed_minutes': round((time.time() - started) / 60.0, 2),
    }
    (ARTIFACTS / 'part5_repaint_metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    np.savez_compressed(ARTIFACTS / 'part5_repaint_outputs.npz', outputs=all_outputs, observed=observed, truth=images, mask=mask)
    print('PART 5 FINISHED.')
    print('Main grid:', EVIDENCE / 'part5_repaint_all_completions_grid.png')
    print('Metrics:', ARTIFACTS / 'part5_repaint_metrics.json')


if __name__ == '__main__':
    main()
