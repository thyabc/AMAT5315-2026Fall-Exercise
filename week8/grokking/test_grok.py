"""Unit tests for the grokking answer key.  .venv/bin/python -m unittest week8/grokking/test_grok.py"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fourier  # noqa: E402
import grok  # noqa: E402
import numpy_check  # noqa: E402


class Split(unittest.TestCase):
    def test_partition_and_labels(self):
        (a, b, c), tr, te = grok.make_split(13, 0.3, seed=4)
        self.assertEqual(np.intersect1d(tr, te).size, 0)
        self.assertEqual(tr.size + te.size, 169)
        self.assertEqual(tr.size, round(0.3 * 169))
        np.testing.assert_array_equal(c, (a + b) % 13)

    def test_random_labels_differ(self):
        (a, b, c), _, _ = grok.make_split(13, 0.3, seed=4, labels="random")
        self.assertLess(np.mean(c == (a + b) % 13), 0.3)


class Fourier(unittest.TestCase):
    p = 17

    def test_basis_orthonormal(self):
        F = fourier.fourier_basis(self.p)
        np.testing.assert_allclose(F.T @ F, np.eye(self.p), atol=1e-12)

    def test_pure_frequency(self):
        x = np.arange(self.p)
        E = np.stack([np.cos(2 * np.pi * 3 * x / self.p + 0.4), 2 * np.sin(2 * np.pi * 5 * x / self.p)], 1)
        pw = fourier.frequency_power(E)
        self.assertEqual(sorted(np.argsort(pw)[::-1][:2].tolist()), [3, 5])
        self.assertAlmostEqual(pw.sum(), (E ** 2).sum())        # Parseval
        self.assertAlmostEqual(fourier.top_k_fraction(E, 2), 1.0)
        self.assertEqual(fourier.key_frequencies(E, 2), [3, 5])

    def test_matches_numpy_fft(self):
        E = np.random.default_rng(0).normal(size=(self.p, 6))
        np.testing.assert_allclose(fourier.frequency_power(E), numpy_check.power(E), rtol=1e-10)

    def test_keep_and_remove_are_complements(self):
        E = np.random.default_rng(1).normal(size=(self.p, 4))
        keep, rem = fourier.keep_only(E, [2, 7]), fourier.remove(E, [2, 7])
        # keep has the constant + 2 + 7; remove has everything but 2 and 7.
        const = fourier.projector(self.p, [], constant=True) @ E
        np.testing.assert_allclose(keep + rem - const, E, atol=1e-5)
        np.testing.assert_allclose(keep, numpy_check.fft_filter(E, [2, 7], True), atol=1e-5)
        np.testing.assert_allclose(rem, numpy_check.fft_filter(E, [2, 7], False), atol=1e-5)

    def test_stacked_embedding(self):
        E = np.random.default_rng(2).normal(size=(2, self.p, 3))
        np.testing.assert_allclose(fourier.frequency_power(E), numpy_check.power(E), rtol=1e-10)


class Models(unittest.TestCase):
    def test_numpy_forward_matches_jax(self):
        import jax
        p = 11
        a = np.repeat(np.arange(p), p); b = np.tile(np.arange(p), p)
        for model, arch in [("transformer", dict(d_model=16, n_heads=2, d_mlp=32)), ("mlp", dict(width=24))]:
            init, logits = grok.MODELS[model]
            P = jax.tree.map(np.asarray, init(jax.random.PRNGKey(0), p, **arch))
            ref = np.asarray(logits(P, a, b))
            ours = (numpy_check.transformer_logits if model == "transformer" else numpy_check.mlp_logits)(
                {k: v.astype(np.float64) for k, v in P.items()}, a, b)
            np.testing.assert_allclose(ours, ref, rtol=1e-4, atol=1e-4)

    def test_mlp_analytic_fourier_solution_is_exact(self):
        """Gromov's analytic quadratic-MLP solution computes (a + b) mod p exactly."""
        p, M = 13, 4                       # phases on an M x M grid cancel every cross term
        x = np.arange(p)
        Wa, Wb, U = [], [], []
        for k in range(1, (p - 1) // 2 + 1):
            for i in range(M * M):
                phi, psi = 2 * np.pi * (i // M) / M, 2 * np.pi * (i % M) / M
                Wa.append(np.cos(2 * np.pi * k * x / p + phi))
                Wb.append(np.cos(2 * np.pi * k * x / p + psi))
                U.append(np.cos(-2 * np.pi * k * x / p - phi - psi))
        P = {"W_a": np.array(Wa).T, "W_b": np.array(Wb).T, "W_U": np.array(U)}
        a = np.repeat(x, p); b = np.tile(x, p)
        pred = np.argmax(numpy_check.mlp_logits(P, a, b), -1)
        np.testing.assert_array_equal(pred, (a + b) % p)

    def test_short_training_memorizes(self):
        r = grok.train(model="mlp", p=11, frac=0.5, steps=300, log_every=50, arch=dict(width=64), lr=3e-3)
        self.assertGreater(r["history"]["train_acc"][-1], 0.99)
        self.assertIn("memorized", r["snapshots"])
        self.assertEqual(len(r["history"]["step"]), 7)


if __name__ == "__main__":
    unittest.main()
