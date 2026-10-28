"""Fourier analysis over the residue index and frequency ablations (NumPy only).

For a matrix E with rows indexed by a residue x = 0..p-1 (p odd prime), the real
Fourier basis is the constant vector and, for each frequency k = 1..(p-1)/2, the pair
cos(2 pi k x / p), sin(2 pi k x / p). The basis is orthogonal; we normalize it, so the
power of frequency k is the squared norm of E projected onto its two-dimensional
subspace, summed over the embedding columns.
"""
from __future__ import annotations

import numpy as np


def fourier_basis(p):
    """(p, p) orthonormal matrix; column 0 constant, columns 2k-1, 2k are cos k, sin k."""
    x = np.arange(p)
    cols = [np.ones(p) / np.sqrt(p)]
    for k in range(1, (p - 1) // 2 + 1):
        c, s = np.cos(2 * np.pi * k * x / p), np.sin(2 * np.pi * k * x / p)
        cols += [c / np.linalg.norm(c), s / np.linalg.norm(s)]
    return np.stack(cols, 1)


def frequency_power(E):
    """Power per frequency 0..(p-1)/2 of E (p, d) or of a stack (m, p, d), summed over d and m."""
    E = np.asarray(E, dtype=np.float64)
    if E.ndim == 3:
        return sum(frequency_power(e) for e in E)
    p = E.shape[0]
    coef = fourier_basis(p).T @ E                 # (p, d) coefficients
    sq = (coef ** 2).sum(1)
    return np.concatenate([[sq[0]], sq[1::2] + sq[2::2]])


def top_k_fraction(E, k=5):
    """Fraction of the non-constant power carried by the k strongest frequencies."""
    pw = frequency_power(E)[1:]
    return float(np.sort(pw)[::-1][:k].sum() / pw.sum())


def key_frequencies(E, k=5):
    pw = frequency_power(E)[1:]
    return sorted((np.argsort(pw)[::-1][:k] + 1).tolist())


def projector(p, freqs, constant=False):
    """Orthogonal projector onto the span of the given frequencies (and the constant)."""
    F = fourier_basis(p)
    cols = ([0] if constant else []) + [c for f in freqs for c in (2 * f - 1, 2 * f)]
    Q = F[:, cols]
    return Q @ Q.T


def keep_only(E, freqs):
    """Project the residue rows onto the constant plus the given frequencies."""
    E = np.asarray(E, dtype=np.float64)
    P = projector(E.shape[-2], freqs, constant=True)
    return (P @ E).astype(np.float32) if E.ndim == 2 else np.stack([P @ e for e in E]).astype(np.float32)


def remove(E, freqs):
    """Remove the given frequencies from the residue rows."""
    E = np.asarray(E, dtype=np.float64)
    P = projector(E.shape[-2], freqs)
    return (E - P @ E).astype(np.float32) if E.ndim == 2 else np.stack([e - P @ e for e in E]).astype(np.float32)
