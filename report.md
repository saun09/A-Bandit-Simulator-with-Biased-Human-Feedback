# Bandit Simulator with Biased Human Feedback

## Problem
Simulate how biased or noisy human feedback impacts the learning efficiency and fairness of online decision-making agents using bandit algorithms.

## Setup
- 10 items with latent quality
- User models: honest, noisy, random, recency-biased
- Algorithms: Epsilon-Greedy, Thompson Sampling
- Evaluated on cumulative regret and fairness (group selection ratio)

## Results
- Thompson Sampling consistently outperformed ε-Greedy under high noise.
- Recency-biased feedback led to over-selection of lower-quality items.
- Group A received ~70% of selections under random user model, indicating fairness issues.

## Future Work
- Add strategic agents
- Test with real feedback datasets
- Deploy Streamlit demo for interaction
