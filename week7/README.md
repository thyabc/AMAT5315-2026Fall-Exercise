# Week 7: Diffusion and Flow Matching

## Part 1 - Diffusion Models on a Gaussian Mixture

### Implementation
- JAX noise predictor with time conditioning
- Four equally weighted Gaussian components
- 1000-step forward diffusion
- 25000 training updates, batch size 512
- Linear learning-rate decay
- DDPM reverse sampling without clipping
- 10 saved training stages

### Results
- Evaluation samples: 2000
- Final training loss: 0.2081
- Cluster populations: 0.2550, 0.2505, 0.2380, 0.2565
- Maximum centre-coordinate error: 0.0272
- Maximum per-coordinate spread error: 0.0121
- Forward terminal mean: 0.0168, -0.0078
- Forward terminal variance: 1.0985, 1.0224

The cluster population, centre and spread checks passed.
The forward terminal variance also meets the stated tolerance,
although the first coordinate is close to the upper boundary.

### Evidence
- evidence/forward_diffusion.png
- evidence/forward_diffusion.gif
- evidence/training_stages.gif
- evidence/generated_vs_target.png
- evidence/reverse_diffusion.gif

### Reproduce
Run from the week7 directory:
JAX_PLATFORMS=cpu .venv/bin/python src/part1_diffusion.py
