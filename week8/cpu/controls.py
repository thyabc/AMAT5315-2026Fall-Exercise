"""Negative controls for Parts 2 and 3 (measure only).

    python controls.py configs/part2.json     # needs work/part2 from part2.py

1. Non-causal mask: the same model trained and scored without the causal mask
   sees the target; its self-reported validation loss is implausibly low, the
   causality probe is large, and the checker's causal rescoring exposes it.
2. Averaging batch PPLs instead of token-weighting the summed NLL.
(The leaked fit and the total-parameter count are in `scaling.py compare`.)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lm  # noqa: E402


def noncausal(config, steps):
    config = dict(config, max_steps=steps, warmup_steps=min(config['warmup_steps'], steps // 4))
    params, rows, final, timing = lm.train(config, causal=False, log_every=steps // 4)
    out = lm.HERE / 'work' / 'control-noncausal'
    lm.export(out, params, config, steps)
    return dict(steps=steps, self_reported_noncausal_loss=final['loss'],
                causal_rescore_same_weights=lm.score(params, config, lm.corpus()[1])['loss'],
                independent_checker_loss=lm.independent_score(out)['loss'],
                causality_probe=lm.causality_probe(params, config, causal=False),
                causality_probe_with_mask=lm.causality_probe(params, config, causal=True),
                wall_seconds=timing['wall_seconds'])


def averaging(part2_dir, batch=32):
    params, meta = lm.load(part2_dir)
    config = meta['architecture']
    s = lm.score(params, config, lm.corpus()[1])
    nll, tokens = np.array(s['block_nll']), np.array(s['block_tokens'])
    correct = float(np.exp(nll.sum() / tokens.sum()))
    batches = [(nll[i:i + batch].sum(), tokens[i:i + batch].sum()) for i in range(0, len(nll), batch)]
    batch_ppl = float(np.mean([np.exp(a / b) for a, b in batches]))
    batch_loss_mean = float(np.mean([a / b for a, b in batches]))
    block_ppl = float(np.mean(np.exp(nll / tokens)))
    return dict(token_weighted_ppl=correct, token_weighted_loss=float(nll.sum() / tokens.sum()),
                mean_of_batch_ppls=batch_ppl, exp_of_mean_batch_loss=float(np.exp(batch_loss_mean)),
                mean_of_block_ppls=block_ppl, batch_blocks=batch, blocks=len(nll),
                final_block_tokens=int(tokens[-1]),
                two_target_wrong=(2 + 4) / 2, two_target_right=float(np.sqrt(8)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--steps', type=int, default=300)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    result = dict(averaging=averaging(lm.HERE / 'work' / 'part2'))
    print(json.dumps(result['averaging'], indent=2), flush=True)
    result['noncausal'] = noncausal(config, args.steps)
    print(json.dumps(result['noncausal'], indent=2))
    lm.record('controls', result)
