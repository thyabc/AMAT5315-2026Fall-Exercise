"""Unit tests for the Week 8 CPU answer key (fast; no long training).

    python -m unittest test_cpu.py
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'checker'))
import lm  # noqa: E402
import baselines  # noqa: E402
import scaling  # noqa: E402
import evaluator  # noqa: E402  (checker's independent float64 NumPy decoder)


class Baselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result, cls.uni, cls.bi = baselines.baselines()

    def test_uniform_is_ln_vocabulary(self):
        self.assertAlmostEqual(self.result['uniform']['loss'], math.log(65), places=12)
        self.assertAlmostEqual(self.result['uniform']['ppl'], 65.0, places=9)

    def test_two_target_example(self):
        self.assertAlmostEqual(self.result['two_target_example']['ppl'], math.sqrt(8), places=12)

    def test_unigram_contract_value(self):
        self.assertAlmostEqual(self.result['unigram']['loss'], 3.347305, places=6)
        self.assertAlmostEqual(self.result['unigram']['ppl'], 28.426032, places=5)

    def test_distributions_normalized_and_bigram_beats_unigram(self):
        self.assertAlmostEqual(self.uni.sum(), 1.0, places=12)
        np.testing.assert_allclose(self.bi.sum(axis=1), 1.0, atol=1e-12)
        self.assertLess(self.result['bigram']['loss'], self.result['unigram']['loss'])

    def test_every_validation_character_but_first_scored_once(self):
        self.assertEqual(self.result['scored_tokens'], self.result['val_tokens'] - 1)


class Model(unittest.TestCase):
    config = lm.make_config(2, 16, 2)

    def setUp(self):
        self.params = lm.initialize(self.config, lm.jax.random.PRNGKey(3))
        # Random weights large enough that attention is not uniform.
        self.params = {k: v * 20 if k.endswith('weight') else v for k, v in self.params.items()}

    def test_parameter_counts_match_arrays(self):
        for layers, width in [(1, 8), (2, 16), (3, 48)]:
            config = lm.make_config(layers, width, 2)
            params = lm.initialize(config, lm.jax.random.PRNGKey(0))
            counts = lm.parameter_counts(config)
            self.assertEqual(counts['total'], sum(v.size for v in params.values()))
            self.assertEqual(counts['non_embedding'],
                             sum(v.size for k, v in params.items() if k.startswith('blocks.')))

    def test_blocks_cover_targets_once(self):
        ids = np.arange(300) % 65
        x, y, mask = lm.blocks(ids)
        self.assertEqual(mask.sum(), 299)
        np.testing.assert_array_equal(y[mask], ids[1:])
        np.testing.assert_array_equal(x[mask], ids[:-1])

    def test_jax_score_matches_independent_numpy(self):
        ids = lm.corpus()[1][:700]
        mine = lm.score(self.params, self.config, ids)
        params64 = {k: np.asarray(v, np.float64) for k, v in self.params.items()}
        theirs = evaluator.evaluate(params64, self.config, ids)
        self.assertEqual(mine['token_count'], theirs['token_count'])
        self.assertAlmostEqual(mine['loss'], theirs['loss'], delta=1e-4)

    def test_causality_probe(self):
        causal = lm.causality_probe(self.params, self.config, causal=True, pairs=2)
        leaky = lm.causality_probe(self.params, self.config, causal=False, pairs=2)
        self.assertLess(causal['max_prefix_logit_change'], 1e-5)
        self.assertGreater(leaky['max_prefix_logit_change'], 1e-2)


class Fit(unittest.TestCase):
    def test_recovers_synthetic_law(self):
        N = np.array([6e3, 1.4e4, 2.5e4, 5.6e4, 1e5])
        y = scaling.law(N, 0.6, 0.35, 1.4)
        params, _ = scaling.fit_law(N, y, np.full(5, 1e-3))
        np.testing.assert_allclose(params, [0.6, 0.35, 1.4], rtol=1e-4)

    def test_pure_power_law_on_exact_power_law(self):
        N = np.array([1e3, 1e4, 1e5])
        a, alpha = scaling.fit_pure(N, 2.0 * (N / scaling.N0) ** -0.1, np.full(3, 0.01))
        self.assertAlmostEqual(alpha, 0.1, places=10)
        self.assertAlmostEqual(a, 2.0, places=10)

    def test_token_weighting_differs_from_batch_average(self):
        # Batches of unequal length: (2 tokens, NLL ln 8) and (1 token, NLL ln 1).
        nll, tokens = np.array([math.log(8), 0.0]), np.array([2, 1])
        right = math.exp(nll.sum() / tokens.sum())
        wrong = np.mean(np.exp(nll / tokens))
        self.assertAlmostEqual(right, 2.0, places=12)
        self.assertNotAlmostEqual(wrong, right, places=3)


if __name__ == '__main__':
    unittest.main()
