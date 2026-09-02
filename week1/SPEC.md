\# Pi Estimator Specification



The program estimates pi using Monte Carlo sampling.



The function estimate\_pi(n, seed)

throws n random points into a unit square,

counts points inside the unit circle,

and returns:



4 × inside\_points / total\_points



The result is correct when:



abs(estimate\_pi(1\_000\_000, seed=2026) - pi) < 1e-2

