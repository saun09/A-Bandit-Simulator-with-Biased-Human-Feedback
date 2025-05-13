# A-Bandit-Simulator-with-Biased-Human-Feedback
https://saun09-a-bandit-simulator-with-biased-human-feedback.streamlit.app/
<p align="center">
  <img src="https://github.com/user-attachments/assets/6c9cb6b4-88fb-488c-bba6-780cd4c02f33" width="400"/>
 <img src="https://github.com/user-attachments/assets/c302d155-8b23-4c22-b26a-4f555f9ac38d" width="393"/>


</p>


## Problem
Simulate using python, how biased or noisy human feedback impacts the learning efficiency and fairness of online decision-making agents using bandit algorithms.

## Introduction
Online systems frequently learn from human preferences rather than explicit rewards. In real-world settings, human feedback is often noisy, delayed, or strategically biased (ex-favoring recent or popular items). We build a Python simulator to quantify how these biases impact:  
- **Learning efficiency** (cumulative regret)  
- **Fairness** (selection disparity between two synthetic item groups)
  
## Setup
- 10 items with latent quality
- User models: honest, noisy, random, recency-biased
- Algorithms: Epsilon-Greedy, Thompson Sampling
- Evaluated on cumulative regret and fairness (group selection ratio)

## Simulator Design  
- **Items:** 10 synthetic items, each assigned a latent quality ∈ [0,1].  
- **Groups:** Each item is randomly labeled group A or B for fairness tracking.  

## User Models  
1. **Honest:** Always prefers the higher-quality item.  
2. **Noisy:** Chooses correctly with probability sigmoid(Δ/0.2).  
3. **Random:** 50/50 choice.  
4. **Recency-biased:** Always favors the most recently shown item.

## Algorithms  
- **ε-Greedy:** ε=0.1  
- **Thompson Sampling** (Beta(1,1) priors)  

## Experimental Protocol  
- **Seeds:** [0,1,2,3,4]  
- **Timesteps per run:** 500  
- **Runs:** 5 per (algorithm × user model)  
- **Data:**  
  - Regret time series saved in `results/{agent}_{bias}_regret.csv`  
  - Final group-A ratios in `results/{agent}_{bias}_groups.csv`

## Results
- Thompson Sampling consistently outperformed ε-Greedy under high noise.
- Recency-biased feedback led to over-selection of lower-quality items.
- Group A received ~70% of selections under random user model, indicating fairness issues.
 ## Final Cumulative Regret

| Algorithm          | Bias         | Mean Regret | Std Dev Regret |
|--------------------|--------------|-------------|----------------|
| ε-Greedy           | honest       | 0.000000    | 0.000000       |
| ε-Greedy           | noisy        | 0.001387    | 0.001902       |
| ε-Greedy           | random       | 0.414210    | 0.072947       |
| ε-Greedy           | recent_bias  | 0.586225    | 0.211482       |
| Thompson Sampling  | honest       | 0.000000    | 0.000000       |
| Thompson Sampling  | noisy        | 0.014591    | 0.030790       |
| Thompson Sampling  | random       | 0.228210    | 0.239611       |
| Thompson Sampling  | recent_bias  | 0.366670    | 0.286378       |


- This table shows how much regret (missed reward) the learning agent accumulates when interacting with different types of user feedback. Lower regret means the algorithm is learning user preferences more efficiently.
![combined_regret](https://github.com/user-attachments/assets/89b58f2f-bfd1-4485-9c8a-39c90c589345)
For example, the ε-Greedy algorithm under **random feedback** shows high regret (0.414), meaning it's unable to reliably learn what users like—similar to a recommender system that asks users to rate videos they’ve barely watched. In contrast, **honest feedback** leads to zero regret, indicating ideal learning conditions.

Thompson Sampling generally performs better across biases, especially under noise, highlighting its robustness in environments like YouTube’s recommender engine, where user likes/dislikes are delayed, inconsistent, or strategically manipulated.
## Fairness: Group-A Selection Ratio  

| Algorithm          | Bias         | Mean Group-A Ratio | Std Dev Ratio |
|--------------------|--------------|---------------------|----------------|
| ε-Greedy           | honest       | 0.7564              | 0.190585       |
| ε-Greedy           | noisy        | 0.7212              | 0.217984       |
| ε-Greedy           | random       | 0.4668              | 0.456665       |
| ε-Greedy           | recent_bias  | 0.5456              | 0.394138       |
| Thompson Sampling  | honest       | 0.4024              | 0.540077       |
| Thompson Sampling  | noisy        | 0.4512              | 0.418601       |
| Thompson Sampling  | random       | 0.5328              | 0.156586       |
| Thompson Sampling  | recent_bias  | 0.6120              | 0.127851       |

- This table measures fairness by tracking how often items from **Group A** were selected. Ideally, if both groups have equal quality, the ratio should be close to **0.5**.

![combined_group_ratio](https://github.com/user-attachments/assets/1d652120-7a4b-4dde-95a9-2d74e870ed3b)
Significant deviation (ex- ε-Greedy with honest users selecting Group A 75% of the time) suggests **algorithmic favoritism**—which may stem from biased feedback loops. This mimics real-world settings like TikTok, where viral trends can lead to echo chambers, disproportionately promoting one content genre or demographic group over others.

Thompson Sampling maintains near-0.5 group balance on average, indicating that probabilistic exploration can help mitigate unintended group-level disparities in attention or exposure.

## Average Regret & Fairness Disparity
| Algorithm          | Avg. Regret (All Biases) | Avg. Std Regret | Avg. Group-A Ratio | Avg. Disparity from 0.5 |
|--------------------|--------------------------|------------------|---------------------|--------------------------|
| ε-Greedy           | 0.250455                 | 0.071583         | 0.62255             | 0.12255                  |
| Thompson Sampling  | 0.152934                 | 0.139695         | 0.49960             | 0.00040                  |




## Interpretation
-Thompson Sampling has lower regret overall and is closer to fair selection (group parity).
-ε-Greedy learns slower and shows larger bias toward certain groups, especially under “honest” and “noisy” feedback.
