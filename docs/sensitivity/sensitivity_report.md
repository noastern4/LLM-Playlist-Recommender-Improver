# Sensitivity Analysis — λ in Objective Formula

**Generated:** 2026-02-27 15:06:06

**Objective:** `NDCG@10 + λ × intra_list_diversity`

## Results by λ

| λ | alpha | beta | gamma | NDCG@10 | Δ NDCG | ILD | Δ ILD | Genre Div | Avg Pop |
|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 1.000 | 0.000 | 0.000 | 0.3851 | +0.0000 | 0.6051 | +0.0000 | 15.57 | 55.33 |
| 0.1 | 0.549 | 0.386 | 0.065 | 0.3784 | -0.0067 | 0.7503 | +0.1452 | 16.52 | 54.78 |
| 0.2 | 0.360 | 0.489 | 0.151 | 0.3701 | -0.0150 | 0.8221 | +0.2170 | 17.56 | 51.78 |
| 0.3 | 0.360 | 0.489 | 0.151 | 0.3701 | -0.0150 | 0.8221 | +0.2170 | 17.56 | 51.78 |

**Baseline (pure voting):**
- NDCG@10 = 0.3851
- ILD = 0.6051
- Genre diversity = 15.57

## Charts

![Trade-off Curve](tradeoff_curve.png)

![Metrics by Lambda](metrics_by_lambda.png)

## Interpretation

- As λ increases, the optimizer gives more weight to diversity, increasing ILD but decreasing NDCG@10.
- λ=0 is equivalent to the pure voting baseline (no diversity term).
- The trade-off curve shows the Pareto frontier between relevance and diversity.