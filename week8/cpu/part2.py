"""Part 2: train one small causal transformer on CPU, rescore it independently,
probe causality, sample at two temperatures and record a first-layer attention map.

    python part2.py configs/part2.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lm  # noqa: E402

PROMPT = 'ROMEO:\nBut soft, what light'
SAMPLE_PROMPT = 'ROMEO:\n'


def main(config_path, out):
    config = json.loads(Path(config_path).read_text())
    counts = lm.parameter_counts(config)
    print(f"config {config} params={counts}", flush=True)
    params, rows, final, timing = lm.train(config, log_every=config.get('log_every', 100))
    meta = lm.export(out, params, config, config['max_steps'])
    independent = lm.independent_score(out)
    probe = lm.causality_probe(params, config)
    _, _, corpus = lm.corpus()
    vocabulary = corpus['vocabulary']
    samples = [dict(prompt=SAMPLE_PROMPT, temperature=t, seed=7, length=500,
                    checkpoint_sha256=meta['checkpoint_sha256'],
                    text=lm.generate(params, config, vocabulary, SAMPLE_PROMPT, t, 500, 7))
               for t in (0.5, 1.0)]
    ids = np.array([[vocabulary.index(c) for c in PROMPT]], np.int32)
    logits, attention = lm.forward(params, ids, config, return_attention=True)
    attention = np.asarray(attention)[0]          # heads x t x t, first layer
    upper = float(np.max(np.abs(np.triu(attention, k=1))))
    probabilities = np.asarray(lm.jax.nn.softmax(logits[0, -1]))
    top = np.argsort(probabilities)[::-1][:5]
    lm.save_arrays(part2_attention=attention, part2_prompt=np.frombuffer(PROMPT.encode(), np.uint8),
                   part2_next_probabilities=probabilities,
                   part2_steps=np.array([r['step'] for r in rows]),
                   part2_train=np.array([r['train_loss'] for r in rows]),
                   part2_val=np.array([r['val_loss_subset'] for r in rows]))
    result = dict(config=config, parameters=counts, timing=timing, curve=rows,
                  self_score={k: final[k] for k in ('loss', 'ppl', 'bpc', 'token_count', 'summed_nll')},
                  independent_score=independent,
                  self_vs_independent_abs_diff=abs(final['loss'] - independent['loss']),
                  causality=probe, attention_prompt=PROMPT,
                  attention_upper_triangle_max=upper,
                  attention_row_sum_max_error=float(np.max(np.abs(attention.sum(-1) - 1))),
                  next_char_top5=[(vocabulary[i], float(probabilities[i])) for i in top],
                  samples=samples, checkpoint_sha256=meta['checkpoint_sha256'])
    lm.record('part2', result)
    print(json.dumps({k: result[k] for k in ('self_score', 'independent_score', 'causality',
                                              'attention_upper_triangle_max')}, indent=2))
    print(f"steps/s={timing['steps_per_second']:.2f} wall={timing['wall_seconds']:.0f}s")
    for s in samples:
        print(f"--- T={s['temperature']}\n{s['text']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--out', default=str(lm.HERE / 'work' / 'part2'))
    args = parser.parse_args()
    main(args.config, args.out)
