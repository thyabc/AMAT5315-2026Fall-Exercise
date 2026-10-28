#!/usr/bin/env python3
"""Small independent checks; no trained weights, GPU, network, or training needed."""
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp
from evaluator import CORPUS_SHA256, evaluate, forward, load_model, numeric_npz, shapes

HERE = Path(__file__).parent
loader = importlib.machinery.SourceFileLoader('gate', str(HERE / 'check'))
spec = importlib.util.spec_from_loader(loader.name, loader)
gate = importlib.util.module_from_spec(spec)
loader.exec_module(gate)


class CheckerTest(unittest.TestCase):
    def test_comparison_rejects_different_hardware_or_matrix_precision(self):
        config = json.loads((HERE.parent/'speedrun/config-baseline.json').read_text())
        config.update(device='cuda:0', device_kind='NVIDIA RTX A4500', precision='float32',
                      matmul_precision='highest')
        for field, value in [('device_kind', 'NVIDIA A100'), ('matmul_precision', 'default')]:
            narrow = dict(config, n_embd=64)
            narrow[field] = value
            with self.subTest(field=field), patch.object(gate, 'json_file', side_effect=[config, narrow]):
                with self.assertRaisesRegex(gate.EvidenceFailure, 'comparison-controls'):
                    gate.check_comparison(Path('/unused'), None, None, None, None, None)

    @classmethod
    def setUpClass(cls):
        cls.config = dict(n_embd=8, n_layer=1, n_head=2)
        rng = np.random.default_rng(7)
        cls.params = {k: rng.normal(0, 0.15, size=s) for k, s in shapes(cls.config).items()}
        for k in cls.params:
            if k.endswith('.scale'): cls.params[k][:] = 1

    def test_causal_mask_and_future_perturbation(self):
        a = np.arange(12)[None, :]
        b = a.copy(); b[:, 5:] += 20
        la, attention = forward(self.params, self.config, a)
        lb, _ = forward(self.params, self.config, b)
        np.testing.assert_allclose(la[:, :5], lb[:, :5], atol=1e-14, rtol=0)
        self.assertEqual(np.max(np.triu(attention, 1)), 0)
        np.testing.assert_allclose(attention.sum(-1), 1)
        bad_a, _ = forward(self.params, self.config, a, causal=False)
        bad_b, _ = forward(self.params, self.config, b, causal=False)
        self.assertGreater(np.max(abs(bad_a[:, :5]-bad_b[:, :5])), 100*gate.PROBE_TOL)

    def test_shift_tail_and_token_weighting(self):
        ids = np.random.default_rng(11).integers(0, 65, 134)
        result = evaluate(self.params, self.config, ids)
        self.assertEqual(result['block_tokens'], [128, 5])
        manual = []
        for start in (0, 128):
            x = ids[start:min(start+128, len(ids)-1)][None, :]
            logits, _ = forward(self.params, self.config, x)
            lp = logits[0]-logsumexp(logits[0], axis=-1, keepdims=True)
            manual.extend(-lp[np.arange(x.size), ids[start+1:start+x.size+1]])
        self.assertAlmostEqual(result['loss'], np.mean(manual), places=13)
        self.assertAlmostEqual(result['ppl'], np.exp(np.mean(manual)), places=11)
        wrong = evaluate(self.params, self.config, ids, target_shift=0)
        self.assertGreater(abs(wrong['loss']-result['loss']), 10*gate.SCORE_TOL)
        wrong_weighting = np.mean(np.array(result['block_nll']) / result['block_tokens'])
        self.assertGreater(abs(wrong_weighting-result['loss']), 10*gate.SCORE_TOL)
        self.assertAlmostEqual(np.exp(-np.log([.5, .25]).mean()), np.sqrt(8))
        self.assertNotAlmostEqual(np.exp(-np.log([.5, .25]).mean()), 3)

    def test_unsafe_and_nonfinite_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bad.npz'
            for array in (np.array([np.nan]), np.array([np.inf]), np.array([object()])):
                np.savez(path, value=array)
                with self.assertRaises(ValueError): numeric_npz(path)

    def test_model_schema_checks_shapes_and_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            art = Path(directory)
            vocabulary = [str(i) for i in range(65)]
            architecture = dict(self.config, vocab_size=65, block_size=128,
                                layer_norm_epsilon=1e-5, gelu='exact', dropout=0.)
            meta = dict(schema_version=1, architecture=architecture, vocabulary=vocabulary,
                        corpus_sha256=CORPUS_SHA256, step=0)
            def save(params):
                np.savez(art/'model.npz', **params)
                meta['checkpoint_sha256'] = hashlib.sha256((art/'model.npz').read_bytes()).hexdigest()
                (art/'model.json').write_text(json.dumps(meta))
            save(self.params)
            load_model(art, vocabulary)
            architecture['n_head'] = 3
            save(self.params)
            with self.assertRaises(ValueError): load_model(art, vocabulary)
            architecture['n_head'] = 2
            changed = dict(self.params, token_embedding=np.zeros((64, 8)))
            save(changed)
            with self.assertRaises(ValueError): load_model(art, vocabulary)
            changed = dict(self.params, unknown=np.ones(1))
            save(changed)
            with self.assertRaises(ValueError): load_model(art, vocabulary)

    def test_resume_recomputes_raw_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            art = Path(directory); (art/'resume').mkdir()
            (art/'resume.json').write_text(json.dumps(dict(split_step=10, max_weight_error=0)))
            full = {'p::w': np.array([.4, .6]), 'mu::w': np.array([.1, .2]),
                    'nu::w': np.array([.01, .02]), 'step': np.array(12),
                    'key': np.array([14, 37], dtype=np.uint32), 'elapsed_seconds': np.array(1.)}
            np.savez(art/'resume'/'uninterrupted.npz', **full)
            np.savez(art/'resume'/'resumed.npz', **full)
            np.savez(art/'resume'/'losses.npz', uninterrupted=np.array([3., 2.]), resumed=np.array([3., 2.]))
            gate.check_resume(art)
            for key, delta in [('p::w', .01), ('mu::w', .01), ('nu::w', .01), ('key', 1)]:
                changed = {k: v.copy() for k, v in full.items()}
                changed[key] = (changed[key]+delta).astype(changed[key].dtype)
                np.savez(art/'resume'/'resumed.npz', **changed)
                with self.assertRaises(gate.EvidenceFailure): gate.check_resume(art)


if __name__ == '__main__':
    unittest.main()
