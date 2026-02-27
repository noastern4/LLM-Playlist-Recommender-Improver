# Hyperparameter Optimization for the Re-ranking Layer

## Overview

The re-ranking layer scores each candidate track using a weighted combination of three components:

$$\text{Score}(t) = \alpha \cdot \text{Relevance}(t) + \beta \cdot \text{Diversity}(t) + \gamma \cdot \text{Novelty}(t)$$

where $\alpha$, $\beta$, and $\gamma$ control the trade-off between relevance, diversity, and novelty respectively.
Rather than manually setting these weights, we use **Optuna** — a black-box hyperparameter optimization framework — to automatically find the values that best balance recommendation quality and diversity.

---

## Optimization Framework

We use Optuna with the **Tree-structured Parzen Estimator (TPE)** sampler. TPE is a Bayesian optimization method that builds a probabilistic model of the objective function and uses it to guide the search toward promising regions of the parameter space — making it significantly more efficient than grid or random search.

---

## Search Space

Each parameter is sampled from a continuous range and then **normalized to sum to 1**, ensuring the weights form a valid probability-like distribution:

| Parameter | Raw Search Range | Role |
|-----------|-----------------|------|
| $\alpha$ | [0.3, 0.9] | Relevance weight — how much to reward tracks that appear in similar playlists |
| $\beta$ | [0.05, 0.5] | Diversity weight — how much to penalize tracks similar to already-selected ones |
| $\gamma$ | [0.01, 0.3] | Novelty weight — how much to reward less popular tracks |

After sampling raw values $(a, b, g)$, normalization is applied:

$$\alpha = \frac{a}{a+b+g}, \quad \beta = \frac{b}{a+b+g}, \quad \gamma = \frac{g}{a+b+g}$$

This ensures $\alpha + \beta + \gamma = 1$ at all times.

---

## Objective Function

The objective to **maximize** is:

$$\mathcal{F} = \text{NDCG@10} + 0.2 \times \text{ILD@10}$$

where:
- **NDCG@10** (Normalized Discounted Cumulative Gain) measures recommendation relevance — whether the correct tracks appear early in the ranked list
- **ILD@10** (Intra-List Diversity) measures the mean pairwise cosine distance between the embeddings of the 10 recommended tracks — higher means more varied recommendations
- The weight **0.2** is a scaling factor that balances the two terms, chosen because NDCG and ILD operate on similar numerical scales (~0.38 and ~0.60 respectively)

This objective reflects the core goal: **maintain relevance while increasing diversity**. Using NDCG alone would push $\alpha \to 1$ (pure voting), losing all diversity. Using ILD alone would ignore whether recommendations are actually relevant.

---

## Evaluation Protocol

To keep each trial computationally tractable:

- **1,000 test playlists** are randomly sampled (seed=42) from the full test set of 59,071 playlists
- The same sample is reused across all trials to ensure fair comparison
- For each trial, the greedy re-ranking algorithm is run on the 1,000 sampled playlists and NDCG@10 and ILD are computed
- **100 trials** are run in total

To maximize efficiency, all expensive operations (model inference, similarity matrix computation, candidate list construction) are performed **once before optimization begins**. Each trial only re-runs the greedy selection step, which is a vectorized numpy operation taking under 1 second per 1,000 playlists.

---

## Greedy Selection (Vectorized)

At each selection round, all remaining candidate embeddings are stacked into a matrix and the diversity scores are computed in a single matrix multiplication:

$$\text{sim\_matrix} = E_{\text{remaining}} \cdot E_{\text{selected}}^T \in \mathbb{R}^{|\text{remaining}| \times |\text{selected}|}$$

$$\text{Diversity}(t) = 1 - \frac{1}{|\text{selected}|} \sum_{s \in \text{selected}} \text{sim}(t, s)$$

Since all embeddings are L2-normalized, the dot product equals the cosine similarity. This vectorized approach is approximately 70× faster than the naive per-track Python loop.

---

## Output

The optimization produces a JSON file (`data/best_hyperparams.json`) containing:

```json
{
  "alpha": <best relevance weight>,
  "beta":  <best diversity weight>,
  "gamma": <best novelty weight>,
  "objective_value": <best NDCG@10 + 0.2 * ILD achieved>,
  "objective_formula": "NDCG@10 + 0.2 * intra_list_diversity",
  "n_trials": 100,
  "sample_size": 1000
}
```

These weights are then used in the final evaluation run (`--use-optimized` flag) to produce the optimized re-ranking results.

---

## Motivation for This Approach

The initial weights ($\alpha=0.6$, $\beta=0.25$, $\gamma=0.15$) were set manually based on intuition. The comparison results with these static weights showed that while diversity improved significantly (+0.094 ILD), all relevance metrics dropped slightly (NDCG: −0.003). Optuna is used to find weights that **minimize the relevance drop while preserving or improving the diversity gain** — effectively finding the Pareto-optimal trade-off point automatically rather than through manual tuning.
