"""Figures for the Week 8 CPU answer key (Parts 1-3).

Reads week8/data/cpu-measurements.json and week8/data/cpu-evidence.npz written by
week8/cpu/*.py and writes PNGs into week8/results/.

    python week8/data/cpu-figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DATA = Path(__file__).resolve().parent
OUT = DATA.parent / 'results'
M = json.loads((DATA / 'cpu-measurements.json').read_text())
E = dict(np.load(DATA / 'cpu-evidence.npz'))
VOCAB = sorted(set((DATA.parent / 'speedrun' / 'input.txt').read_text()))

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 150, 'savefig.bbox': 'tight'})
BLUE, ORANGE, GREEN, GREY, RED = '#2b6cb0', '#dd6b20', '#2f855a', '#718096', '#c53030'


def label(c):
    return {'\n': r'\n', ' ': '␣'}.get(c, c)


def part1():
    uni, bi = E['part1_unigram'], E['part1_bigram']
    context = 't'
    row = bi[VOCAB.index(context)]
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
    x = np.arange(len(VOCAB))
    axes[0].bar(x, uni, color=BLUE, width=0.8)
    axes[0].axhline(1 / 65, color=GREY, ls='--', lw=0.8)
    axes[0].text(30, 1 / 65 + 0.004, 'uniform 1/65', color=GREY, ha='center', va='bottom', fontsize=8)
    axes[0].set_ylabel('p(x)')
    axes[0].set_title('Unigram (add-one, training split): p(next character)', loc='left')
    axes[1].bar(x, row, color=ORANGE, width=0.8)
    axes[1].set_ylabel(f"p(x | '{context}')")
    axes[1].set_title(f"Bigram (add-one) row for the previous character '{context}'", loc='left')
    top = int(np.argmax(row))
    axes[1].annotate(f"'{VOCAB[top]}': {row[top]:.2f}", (top, row[top]), xytext=(8, -4),
                     textcoords='offset points', fontsize=8)
    axes[1].set_xticks(x, [label(c) for c in VOCAB], fontsize=6.5, family='monospace')
    axes[1].set_xlim(-1, 65)
    fig.align_ylabels(axes)
    fig.savefig(OUT / 'cpu-part1-probabilities.png')
    plt.close(fig)

    p1 = M['part1']
    names = ['uniform', 'unigram', 'bigram']
    losses = [p1[n]['loss'] for n in names]
    ppls = [p1[n]['ppl'] for n in names]
    colors = [GREY, BLUE, ORANGE]
    if 'part2' in M:
        names.append('transformer\n(Part 2)')
        losses.append(M['part2']['independent_score']['loss'])
        ppls.append(M['part2']['independent_score']['ppl'])
        colors.append(GREEN)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.bar(range(len(names)), losses, color=colors, width=0.6)
    for b, l, p in zip(bars, losses, ppls):
        ax.text(b.get_x() + b.get_width() / 2, l + 0.05, f'loss {l:.4f}\nPPL {p:.3f}',
                ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel('validation loss (nats / character)')
    ax.set_ylim(0, max(losses) * 1.25)
    ax.set_title(f"Held-out cross-entropy, {p1['scored_tokens']} scored characters", loc='left')
    fig.savefig(OUT / 'cpu-part1-baselines.png')
    plt.close(fig)


def part2():
    p2 = M['part2']
    curve = p2['curve']
    steps = [r['step'] for r in curve]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(steps, [r['train_loss'] for r in curve], color=BLUE, label='training (mean over interval)')
    ax.plot(steps, [r['val_loss_subset'] for r in curve], color=ORANGE,
            label='validation, first 96 blocks')
    final = p2['independent_score']['loss']
    ax.scatter([steps[-1]], [final], color=GREEN, zorder=3,
               label=f'full validation, independent: {final:.3f}')
    for name, color in [('unigram', BLUE), ('bigram', ORANGE)]:
        value = M['part1'][name]['loss']
        ax.axhline(value, color=GREY, ls=':', lw=0.8)
        ax.text(steps[-1], value, f'{name} {value:.3f}', ha='right', va='bottom', fontsize=8, color=GREY)
    ax.set_xlabel('optimizer step')
    ax.set_ylabel('loss (nats / character)')
    ax.set_ylim(1.3, 3.6)
    ax.legend(frameon=False, fontsize=8, loc='upper right', bbox_to_anchor=(1, 0.88))
    cfg = p2['config']
    ax.set_title(f"{cfg['n_layer']} layers, width {cfg['n_embd']}, {cfg['n_head']} heads, "
                 f"{p2['parameters']['total']:,} parameters", loc='left')
    fig.savefig(OUT / 'cpu-part2-loss.png')
    plt.close(fig)

    attention = E['part2_attention']
    prompt = bytes(E['part2_prompt']).decode()
    heads = attention.shape[0]
    cols = 2 if heads > 2 else heads
    rows = int(np.ceil(heads / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols + 0.8, 3.4 * rows + 0.4),
                             layout='constrained')
    ticks = [label(c) for c in prompt]
    for h, ax in enumerate(np.ravel(axes)):
        image = ax.imshow(attention[h], cmap='viridis', vmin=0, vmax=1)
        ax.set_xticks(range(len(prompt)), ticks, fontsize=5.5, family='monospace')
        ax.set_yticks(range(len(prompt)), ticks, fontsize=5.5, family='monospace')
        ax.set_title(f'layer 1, head {h + 1}', fontsize=8)
        ax.set_xlabel('key (earlier character)', fontsize=7)
        if h % cols == 0:
            ax.set_ylabel('query (predicting position)', fontsize=7)
    fig.colorbar(image, ax=axes, shrink=0.7, label='attention weight')
    fig.suptitle(f'Prompt {prompt!r}: every weight above the diagonal is exactly zero',
                 fontsize=9)
    fig.savefig(OUT / 'cpu-part2-attention.png')
    plt.close(fig)


def part3(section, filename, title):
    p3 = M[section]
    pred = p3['prediction']
    fit = pred['fit']
    res = p3.get('result', {})
    primary = res.get('primary', pred['spec'].get('primary', 'law'))
    table = pred['fit_table']
    N = np.array([r['N'] for r in table], float)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for r in table:
        ax.scatter([r['N']] * len(r['losses']), r['losses'], color=BLUE, s=10, alpha=0.5, lw=0)
    ax.errorbar(N, [r['mean'] for r in table], yerr=fit['sigma_seed'], fmt='o', color=BLUE,
                ms=5, capsize=2, label='fit set: seed mean ± seed s.e.')
    held = pred['heldout']['N']
    grid = np.geomspace(N.min() / 1.5, held * 1.5, 200)
    pure = fit['pure_power_law']
    pure_curve = pure['a'] * (grid / fit['N0']) ** (-pure['alpha'])
    law_curve = fit['a'] * (grid / fit['N0']) ** (-fit['alpha']) + fit['L_inf']
    if primary == 'law':
        ax.plot(grid, law_curve, color=BLUE, lw=1.2,
                label=rf"$aN^{{-\alpha}}+L_\infty$: $\alpha$={fit['alpha']:.2f}±{fit['alpha_std']:.2f}, "
                      rf"$L_\infty$={fit['L_inf']:.2f}±{fit['L_inf_std']:.2f}")
        band = res.get('fit_band_95')
        ax.plot(grid, pure_curve, color=GREY, ls='--', lw=1,
                label=rf"pure power law (comparison): $\alpha$={pure['alpha']:.3f}")
    else:
        ax.plot(grid, pure_curve, color=BLUE, lw=1.2,
                label=rf"$aN^{{-\alpha}}$: $\alpha$={pure['alpha']:.3f}±{pure['alpha_std']:.3f}")
        band = res.get('pure_band_95')
    if band:
        ax.fill_between(band['N'], band['lo'], band['hi'], color=BLUE, alpha=0.15, lw=0,
                        label=f"95% band of the fit (Birge ratio {fit['birge_ratio'] if primary == 'law' else pure['birge_ratio']:.1f})")
    for r in pred.get('excluded_table', []):
        ax.scatter([r['N']], [r['mean']], marker='o', facecolor='white', edgecolor=GREY, s=25,
                   label=f"excluded (width {r['width']}, bigram plateau)")
    value, (lo, hi) = (res['predicted'], res['interval_95']) if res else (fit['prediction'], fit['prediction_interval_95'])
    ax.errorbar([held], [value], yerr=[[value - lo], [hi - value]],
                fmt='D', mfc='white', color=ORANGE, capsize=4, ms=6,
                label=f"prediction, width {pred['heldout']['width']}: {value:.3f} [{lo:.3f}, {hi:.3f}]")
    if res:
        ax.scatter([held] * len(res['heldout']['losses']), res['heldout']['losses'], color=RED,
                   s=10, alpha=0.6, lw=0)
        ax.scatter([held], [res['actual']], color=RED, marker='*', s=90, zorder=4,
                   label=f"trained afterwards: {res['actual']:.3f}")
    for w, e in p3.get('extra_heldout', {}).items():
        el, eh = e['interval_95']
        ax.errorbar([e['N']], [e['prediction']], yerr=[[e['prediction'] - el], [eh - e['prediction']]],
                    fmt='s', mfc='white', color=ORANGE, capsize=3, ms=4, alpha=0.8,
                    label=f"2nd prediction, width {w}: {e['prediction']:.3f} [{el:.3f}, {eh:.3f}]")
        if 'actual' in e:
            ax.scatter([e['N']], [e['actual']], color=RED, marker='P', s=50, zorder=4,
                       label=f"width {w} trained afterwards: {e['actual']:.3f}")
    ax.axhline(M['part1']['bigram']['loss'], color=GREY, ls=':', lw=0.8)
    ax.text(grid[-1], M['part1']['bigram']['loss'], 'bigram ', va='bottom', ha='right', fontsize=8, color=GREY)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ticks = [1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4]
    ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('non-embedding parameters N')
    ax.set_ylabel('validation loss (nats / character)')
    spec = pred['spec']
    tokens = spec['training']['max_steps'] * spec['training']['batch_size'] * 128
    ax.set_title(f"{title}: same {tokens / 1e6:.1f}M-token budget, {spec['n_layer']} layers, "
                 f"{len(spec['seeds'])} seeds per width", loc='left', fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc='lower left')
    fig.savefig(OUT / filename)
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    part1()
    if 'part2' in M:
        part2()
    if 'part3' in M:
        part3('part3', 'cpu-part3-scaling-protocol-a.png', 'Protocol A (law with L_inf)')
    if 'part3b' in M:
        part3('part3b', 'cpu-part3-scaling.png', 'Protocol B (pure power law)')
    print('figures in', OUT)
