"""Estimate pi using Monte Carlo sampling."""

import random


def estimate_pi(n, seed):
    """Return a repeatable Monte Carlo estimate of pi using ``n`` points."""
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("n must be a positive integer")

    rng = random.Random(seed)
    inside_points = 0

    for _ in range(n):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1.0:
            inside_points += 1

    return 4.0*inside_points / n
