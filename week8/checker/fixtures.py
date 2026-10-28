#!/usr/bin/env python3
"""Course-side negative controls, derived from the measured reference artifacts."""
import json
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from evaluator import corpus, evaluate, forward, load_model

COURSE = Path(__file__).resolve().parents[2]


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2)+'\n')


def generate(directory):
    directory = Path(directory).resolve()
    kind = directory.name.removeprefix('week8-')
    reference = COURSE/'fixtures'/'week8-reference'
    if not (reference/'artifacts'/'resume'/'losses.npz').exists():
        subprocess.run(['make', 'reproduce'], cwd=reference, check=True)
    artifacts = directory/'artifacts'
    if artifacts.exists(): shutil.rmtree(artifacts)
    shutil.copytree(reference/'artifacts', artifacts)
    if kind in ('missing-optimizer', 'missing-key'):
        shutil.copyfile(artifacts/'resume'/f'{kind}.npz', artifacts/'resume'/'resumed.npz')
        with np.load(artifacts/'resume'/'losses.npz', allow_pickle=False) as raw:
            traces = {k: raw[k] for k in raw.files}
        traces['resumed'] = traces[kind.replace('-', '_')]
        np.savez(artifacts/'resume'/'losses.npz', **traces)
    elif kind == 'noncausal':
        _, _, vocabulary = corpus()
        params, config, _ = load_model(artifacts, vocabulary)
        with np.load(artifacts/'causality.npz', allow_pickle=False) as raw:
            probe = {k: raw[k] for k in raw.files}
        for name in ('a', 'b'):
            probe['logits_'+name], _ = forward(params, config, probe['inputs_'+name], causal=False)
        np.savez(artifacts/'causality.npz', **probe)
    elif kind == 'wrong-targets':
        _, val, vocabulary = corpus()
        params, config, _ = load_model(artifacts, vocabulary)
        report = json.loads((artifacts/'eval.json').read_text())
        report.update(evaluate(params, config, val, target_shift=0))
        write_json(artifacts/'eval.json', report)
    elif kind == 'wrong-weighting':
        report = json.loads((artifacts/'eval.json').read_text())
        nll, counts = report['block_nll'], report['block_tokens']
        # Real aggregation bug: average two batch means despite unequal token counts.
        report['loss'] = (nll[0]/counts[0]+sum(nll[1:])/sum(counts[1:]))/2
        report['ppl'] = float(np.exp(report['loss']))
        write_json(artifacts/'eval.json', report)
    elif kind == 'broken':
        early = reference/'checkpoints'/'cpu-smoke'
        for filename in ('model.npz', 'model.json'):
            shutil.copyfile(early/filename, artifacts/filename)
        _, _, vocabulary = corpus()
        params, config, meta = load_model(artifacts, vocabulary)
        with np.load(artifacts/'causality.npz', allow_pickle=False) as raw:
            probe = {k: raw[k] for k in raw.files}
        for name in ('a', 'b'):
            probe['logits_'+name], _ = forward(params, config, probe['inputs_'+name])
        np.savez(artifacts/'causality.npz', **probe)
        # Keep the trained report and log: the early checkpoint cannot reproduce them.
        report = json.loads((artifacts/'eval.json').read_text())
        report['checkpoint_sha256'] = meta['checkpoint_sha256']
        write_json(artifacts/'eval.json', report)
    else:
        raise ValueError(f'unknown fixture {kind}')
    print(f'generated {directory.name}: {kind}')


if __name__ == '__main__':
    generate(sys.argv[1])
