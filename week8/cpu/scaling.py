"""Part 3: a scaling law that predicts a held-out model size before it is trained.

    python scaling.py train-fit  configs/scaling.json --jobs 2   # fit-set widths x seeds
    python scaling.py predict    configs/scaling.json            # fit, write prediction (timestamped)
    python scaling.py train-held configs/scaling.json --jobs 2   # refuses without a prediction
    python scaling.py compare    configs/scaling.json            # prediction vs actual, controls

Every run uses the same token budget and schedule; only the width changes.
N is the non-embedding parameter count (Kaplan et al. 2020).
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'work' / 'scaling'
N0 = 1e4                 # reference scale inside the fit, for conditioning only
BOOTSTRAP = 4000


def run_config(spec, width, seed):
    sys.path.insert(0, str(HERE))
    import lm
    return lm.make_config(spec['n_layer'], width, spec['n_head'], seed=seed, **spec['training'])


def run_dir(width, seed):
    return RUNS / f'w{width:03d}-s{seed}'


# ---------------------------------------------------------------- one training run
def run_one(spec, width, seed):
    """Train, export, rescore independently; write result.json. Called in a subprocess."""
    import lm
    config = run_config(spec, width, seed)
    out = run_dir(width, seed)
    params, rows, final, timing = lm.train(config, log_every=spec['log_every'], verbose=False)
    meta = lm.export(out, params, config, config['max_steps'])
    independent = lm.independent_score(out)
    counts = lm.parameter_counts(config)
    result = dict(width=width, seed=seed, config=config, parameters=counts,
                  loss=final['loss'], independent_loss=independent['loss'],
                  token_count=final['token_count'], curve=rows, timing=timing,
                  checkpoint_sha256=meta['checkpoint_sha256'],
                  finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"width {width} seed {seed}: N={counts['non_embedding']} loss={final['loss']:.4f} "
          f"(independent {independent['loss']:.4f}) {timing['steps_per_second']:.1f} steps/s "
          f"{timing['wall_seconds']:.0f}s", flush=True)


def launch(config_path, pairs, jobs):
    """Run (width, seed) pairs as separate single-threaded processes, `jobs` at a time."""
    env = dict(os.environ)
    if jobs > 1:
        env['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
    todo = [(w, s) for w, s in pairs if not (run_dir(w, s) / 'result.json').exists()]
    # Largest first so the pool's tail is short.
    todo.sort(key=lambda p: -p[0])

    def go(pair):
        width, seed = pair
        subprocess.run([sys.executable, __file__, 'run', config_path,
                        '--width', str(width), '--seed', str(seed)], check=True, env=env)
    with ThreadPoolExecutor(jobs) as pool:
        list(pool.map(go, todo))


def collect(widths, seeds):
    table = []
    for width in widths:
        results = [json.loads((run_dir(width, s) / 'result.json').read_text()) for s in seeds]
        losses = np.array([r['loss'] for r in results])
        table.append(dict(width=width, N=results[0]['parameters']['non_embedding'],
                          N_total=results[0]['parameters']['total'],
                          seeds=list(seeds), losses=losses.tolist(),
                          independent_losses=[r['independent_loss'] for r in results],
                          mean=float(losses.mean()),
                          std=float(losses.std(ddof=1)) if len(losses) > 1 else None,
                          wall_seconds=[r['timing']['wall_seconds'] for r in results],
                          steps_per_second=[r['timing']['steps_per_second'] for r in results]))
    return table


# ---------------------------------------------------------------- fitting
def law(N, a, alpha, L_inf):
    return a * (N / N0) ** (-alpha) + L_inf


def fit_law(N, y, sigma):
    p0 = (y[0] - y[-1] + 0.1, 0.3, max(y[-1] - 0.3, 0.5))
    params, cov = curve_fit(law, N, y, p0=p0, sigma=sigma, absolute_sigma=True, maxfev=20000,
                            bounds=([0, 0, 0], [50, 3, 5]))
    return params, cov


def fit_pure(N, y, sigma):
    """L = a N^-alpha (no irreducible term): weighted line in log-log."""
    w = (y / sigma) ** 2                     # var(log y) ~ (sigma/y)^2
    X = np.vstack([np.ones_like(N), -np.log(N / N0)]).T
    beta = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * np.log(y)))
    return np.exp(beta[0]), beta[1]


def pooled_std(table):
    dev = [np.array(r['losses']) - r['mean'] for r in table]
    dof = sum(len(d) - 1 for d in dev)
    return float(np.sqrt(sum((d ** 2).sum() for d in dev) / dof)), dof


def analyse(table, N_target, n_target_seeds, s_pool, rng, key='N', grid=None):
    """WLS fit of the means, parametric bootstrap, prediction interval at N_target."""
    N = np.array([r[key] for r in table], float)
    y = np.array([r['mean'] for r in table])
    k = np.array([len(r['losses']) for r in table])
    sigma = s_pool / np.sqrt(k)
    params, cov = fit_law(N, y, sigma)
    fitted = law(N, *params)
    chi2 = float(np.sum(((y - fitted) / sigma) ** 2))
    # Seed noise alone underestimates the scatter about the law when chi2/dof > 1
    # (width-dependent effects at a fixed budget). Inflate by the Birge ratio.
    birge = float(np.sqrt(max(1.0, chi2 / max(len(N) - 3, 1))))
    sigma_eff = sigma * birge
    draws, preds = [], []
    for _ in range(BOOTSTRAP):
        y_star = fitted + rng.normal(0, sigma_eff)
        try:
            p_star, _ = fit_law(N, y_star, sigma)
        except RuntimeError:
            continue
        draws.append(p_star)
        preds.append(law(N_target, *p_star))
    draws, preds = np.array(draws), np.array(preds)
    # Predicting a mean of n_target_seeds runs adds that mean's seed noise.
    # The held-out mean scatters about the law like a fit point: seed noise x Birge ratio.
    observed = preds + rng.normal(0, birge * s_pool / np.sqrt(n_target_seeds), len(preds))
    a_pure, alpha_pure = fit_pure(N, y, sigma)
    band = None
    if grid is not None:
        curves = np.array([law(np.asarray(grid), *d) for d in draws])
        band = dict(N=list(map(float, grid)), lo=np.percentile(curves, 2.5, axis=0).tolist(),
                    hi=np.percentile(curves, 97.5, axis=0).tolist())
    # Pure power law (L_inf = 0): its own Birge ratio and parametric bootstrap.
    pure = lambda n, a, al: a * (np.asarray(n) / N0) ** (-al)
    fitted_pure = pure(N, a_pure, alpha_pure)
    chi2_pure = float(np.sum(((y - fitted_pure) / sigma) ** 2))
    birge_pure = float(np.sqrt(max(1.0, chi2_pure / max(len(N) - 2, 1))))
    pure_draws = np.array([fit_pure(N, fitted_pure + rng.normal(0, sigma * birge_pure), sigma)
                           for _ in range(BOOTSTRAP)])
    pure_preds = pure(N_target, pure_draws[:, 0], pure_draws[:, 1])
    pure_observed = pure_preds + rng.normal(0, birge_pure * s_pool / np.sqrt(n_target_seeds),
                                            len(pure_preds))
    pure_band = None
    if grid is not None:
        curves = np.array([pure(grid, a, al) for a, al in pure_draws])
        pure_band = dict(N=list(map(float, grid)), lo=np.percentile(curves, 2.5, axis=0).tolist(),
                         hi=np.percentile(curves, 97.5, axis=0).tolist())
    return dict(band=band,
        N_key=key, a=float(params[0]), alpha=float(params[1]), L_inf=float(params[2]),
        a_std=float(draws[:, 0].std()), alpha_std=float(draws[:, 1].std()),
        L_inf_std=float(draws[:, 2].std()),
        alpha_interval_95=np.percentile(draws[:, 1], [2.5, 97.5]).tolist(),
        L_inf_interval_95=np.percentile(draws[:, 2], [2.5, 97.5]).tolist(),
        covariance_std=np.sqrt(np.diag(cov)).tolist(),
        chi2=chi2, dof=len(N) - 3, birge_ratio=birge, bootstrap_draws=len(preds), N0=N0,
        N_target=float(N_target), prediction=float(law(N_target, *params)),
        prediction_mean_interval_95=np.percentile(preds, [2.5, 97.5]).tolist(),
        prediction_interval_95=np.percentile(observed, [2.5, 97.5]).tolist(),
        pure_power_law=dict(a=float(a_pure), alpha=float(alpha_pure),
                            prediction=float(a_pure * (N_target / N0) ** (-alpha_pure)),
                            alpha_std=float(pure_draws[:, 1].std()),
                            alpha_interval_95=np.percentile(pure_draws[:, 1], [2.5, 97.5]).tolist(),
                            chi2=chi2_pure, dof=len(N) - 2, birge_ratio=birge_pure,
                            prediction_interval_95=np.percentile(pure_observed, [2.5, 97.5]).tolist(),
                            band=pure_band),
        sigma_seed=sigma.tolist(), sigma_used=sigma_eff.tolist())


# ---------------------------------------------------------------- stages
def chosen(fit, primary):
    """The pre-registered prediction and 95% interval: 'law' (with L_inf) or 'pure'."""
    if primary == 'pure':
        return fit['pure_power_law']['prediction'], fit['pure_power_law']['prediction_interval_95']
    return fit['prediction'], fit['prediction_interval_95']


def stage_predict(spec):
    import lm
    section = spec.get('section', 'part3')
    store = json.loads(lm.MEASUREMENTS.read_text()) if lm.MEASUREMENTS.exists() else {}
    held = spec['heldout_width']
    held_done = [s for s in spec['seeds'] if (run_dir(held, s) / 'result.json').exists()]
    if held_done:
        raise SystemExit(f'held-out runs already exist for seeds {held_done}: a prediction now would not be blind')
    table = collect(spec['fit_widths'], spec['seeds'])
    s_pool, dof = pooled_std(table)
    N_held = lm.parameter_counts(run_config(spec, held, 0))
    rng = np.random.default_rng(2026)
    fit = analyse(table, N_held['non_embedding'], len(spec['seeds']), s_pool, rng)
    # Control: count embeddings and unembedding as parameters.
    total = analyse(table, N_held['total'], len(spec['seeds']), s_pool, rng, key='N_total')
    prediction = dict(
        written=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        heldout_runs_present_when_written=held_done,
        spec=spec, fit_table=table, pooled_seed_std=s_pool, pooled_dof=dof,
        excluded_table=collect(spec.get('excluded_widths', []), spec['seeds']),
        heldout=dict(width=held, N=N_held['non_embedding'], N_total=N_held['total']),
        fit=fit, control_total_parameter_count=total)
    store_part = store.get(section, {})
    store_part.update(prediction=prediction)
    store_part.pop('result', None)
    lm.record(section, store_part)
    value, (lo, hi) = chosen(fit, spec.get('primary', 'law'))
    print(f"alpha={fit['alpha']:.3f}+-{fit['alpha_std']:.3f}  L_inf={fit['L_inf']:.3f}+-{fit['L_inf_std']:.3f}"
          f"  chi2={fit['chi2']:.2f}/{fit['dof']}")
    print(f"pure power law: alpha={fit['pure_power_law']['alpha']:.4f}+-{fit['pure_power_law']['alpha_std']:.4f} "
          f"chi2={fit['pure_power_law']['chi2']:.2f}/{fit['pure_power_law']['dof']}")
    print(f"PREDICTION ({spec.get('primary', 'law')}) width {held} (N={N_held['non_embedding']}): "
          f"{value:.4f}  95% [{lo:.4f}, {hi:.4f}]")


def stage_compare(spec):
    import lm
    section = spec.get('section', 'part3')
    store = json.loads(lm.MEASUREMENTS.read_text())
    part = store[section]
    prediction = part['prediction']
    held = spec['heldout_width']
    held_table = collect([held], spec['seeds'])[0]
    results = [json.loads((run_dir(held, s) / 'result.json').read_text()) for s in spec['seeds']]
    first_finish = min(r['finished'] for r in results)
    fit = prediction['fit']
    primary = spec.get('primary', 'law')
    predicted, (lo, hi) = chosen(fit, primary)
    actual = held_table['mean']
    rng = np.random.default_rng(7)
    fit_table = prediction['fit_table']
    s_pool = prediction['pooled_seed_std']
    # Regenerate the 95% band of the law with the prediction's own seed (identical draws).
    grid = np.geomspace(min(r['N'] for r in fit_table) / 1.5, held_table['N'] * 1.5, 60)
    band = analyse(fit_table, held_table['N'], len(spec['seeds']), s_pool,
                   np.random.default_rng(2026), grid=grid)
    assert abs(chosen(band, primary)[1][0] - lo) < 1e-12, 'band does not reproduce prediction'
    # Control: the held-out point leaks into the fit.
    leak = analyse(fit_table + [held_table], held_table['N'], len(spec['seeds']), s_pool, rng)
    result = dict(
        prediction_written=prediction['written'], first_heldout_run_finished=first_finish,
        prediction_before_training=prediction['written'] < min(
            r['finished'] for r in results),
        primary=primary, heldout=held_table, predicted=predicted, interval_95=[lo, hi],
        law_prediction=fit['prediction'], law_interval_95=fit['prediction_interval_95'],
        fit_band_95=band['band'], pure_band_95=band['pure_power_law']['band'],
        actual=actual, residual=actual - predicted,
        residual_in_bootstrap_sd=(actual - predicted) / ((hi - lo) / 3.92),
        within_interval=bool(lo <= actual <= hi),
        pure_power_law_prediction=fit['pure_power_law']['prediction'],
        pure_power_law_within_interval=bool(lo <= fit['pure_power_law']['prediction'] <= hi),
        control_total_parameter_count=dict(
            alpha=prediction['control_total_parameter_count']['alpha'],
            alpha_std=prediction['control_total_parameter_count']['alpha_std'],
            prediction=prediction['control_total_parameter_count']['prediction'],
            interval_95=prediction['control_total_parameter_count']['prediction_interval_95']),
        control_leaked_fit=dict(alpha=leak['alpha'], alpha_std=leak['alpha_std'],
                                L_inf=leak['L_inf'], prediction=leak['prediction'],
                                interval_95=leak['prediction_interval_95'],
                                chi2=leak['chi2'], dof=leak['dof']))
    part['result'] = result
    lm.record(section, part)
    if section == 'part3':
        lm.save_arrays(
        part3_N=np.array([r['N'] for r in fit_table] + [held_table['N']], float),
        part3_N_total=np.array([r['N_total'] for r in fit_table] + [held_table['N_total']], float),
        part3_losses=np.array([r['losses'] for r in fit_table] + [held_table['losses']]))
    print(f"predicted ({primary}) {predicted:.4f} [{lo:.4f}, {hi:.4f}]  actual {actual:.4f} "
          f"(seeds {held_table['losses']})  within={result['within_interval']}")
    print(f"order: prediction {prediction['written']} < first held-out finish {first_finish}: "
          f"{result['prediction_before_training']}")


def stage_predict_extra(spec, width):
    """A second blind test of the SAME recorded fit at another width (no refit, no new choices)."""
    import lm
    store = json.loads(lm.MEASUREMENTS.read_text())
    part = store['part3']
    done = [s for s in spec['seeds'] if (run_dir(width, s) / 'result.json').exists()]
    if done:
        raise SystemExit(f'width {width} already trained for seeds {done}: not blind')
    prediction = part['prediction']
    N = lm.parameter_counts(run_config(spec, width, 0))['non_embedding']
    again = analyse(prediction['fit_table'], N, len(spec['seeds']), prediction['pooled_seed_std'],
                    np.random.default_rng(2026))
    assert abs(again['alpha'] - prediction['fit']['alpha']) < 1e-12, 'fit not reproduced'
    extra = part.setdefault('extra_heldout', {})
    extra[str(width)] = dict(written=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                             width=width, N=N, prediction=again['prediction'],
                             interval_95=again['prediction_interval_95'],
                             pure_power_law_prediction=again['pure_power_law']['prediction'])
    lm.record('part3', part)
    lo, hi = again['prediction_interval_95']
    print(f"PREDICTION width {width} (N={N}): {again['prediction']:.4f}  95% [{lo:.4f}, {hi:.4f}]")


def stage_compare_extra(spec, width):
    import lm
    part = json.loads(lm.MEASUREMENTS.read_text())['part3']
    entry = part['extra_heldout'][str(width)]
    table = collect([width], spec['seeds'])[0]
    finished = min(json.loads((run_dir(width, s) / 'result.json').read_text())['finished']
                   for s in spec['seeds'])
    lo, hi = entry['interval_95']
    entry.update(actual=table['mean'], losses=table['losses'], seed_std=table['std'],
                 independent_losses=table['independent_losses'],
                 wall_seconds=table['wall_seconds'], first_run_finished=finished,
                 prediction_before_training=entry['written'] < finished,
                 residual=table['mean'] - entry['prediction'],
                 within_interval=bool(lo <= table['mean'] <= hi))
    lm.record('part3', part)
    print(f"width {width}: predicted {entry['prediction']:.4f} [{lo:.4f}, {hi:.4f}] actual "
          f"{table['mean']:.4f} within={entry['within_interval']} order_ok={entry['prediction_before_training']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['train-fit', 'predict', 'train-held', 'compare', 'run',
                                          'predict-extra', 'train-extra', 'compare-extra'])
    parser.add_argument('config')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--width', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    sys.path.insert(0, str(HERE))
    spec = json.loads(Path(args.config).read_text())
    if args.stage == 'run':
        run_one(spec, args.width, args.seed)
    elif args.stage == 'train-fit':
        widths = spec['fit_widths'] + spec.get('excluded_widths', [])
        launch(args.config, [(w, s) for w in widths for s in spec['seeds']], args.jobs)
    elif args.stage == 'predict':
        stage_predict(spec)
    elif args.stage == 'train-held':
        import lm
        store = json.loads(lm.MEASUREMENTS.read_text()) if lm.MEASUREMENTS.exists() else {}
        if 'prediction' not in store.get(spec.get('section', 'part3'), {}):
            raise SystemExit('write the prediction first: python scaling.py predict ...')
        launch(args.config, [(spec['heldout_width'], s) for s in spec['seeds']], args.jobs)
    elif args.stage == 'compare':
        stage_compare(spec)
    elif args.stage == 'predict-extra':
        stage_predict_extra(spec, args.width)
    elif args.stage == 'train-extra':
        import lm
        part = json.loads(lm.MEASUREMENTS.read_text()).get('part3', {})
        if str(args.width) not in part.get('extra_heldout', {}):
            raise SystemExit('write the prediction first: python scaling.py predict-extra ...')
        launch(args.config, [(args.width, s) for s in spec['seeds']], args.jobs)
    elif args.stage == 'compare-extra':
        stage_compare_extra(spec, args.width)
