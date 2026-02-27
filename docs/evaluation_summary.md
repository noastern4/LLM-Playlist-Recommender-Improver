# Evaluation Summary

## Overview

This document describes the full evaluation pipeline for the LLM-Based Playlist Recommender system.
There are two evaluation phases:

1. **Baseline evaluation** — pure voting recommender using the fine-tuned model
2. **Re-ranking evaluation** — voting + greedy re-ranking for diversity and novelty

---

## Phase 1 — Baseline Evaluation

### How It Works

For each test playlist:
1. Encode the playlist title using the fine-tuned cross-entropy model (384-dim embedding)
2. Find the **top-50 most similar playlists** by cosine similarity against all playlist embeddings
3. Aggregate tracks from those 50 playlists using a **voting mechanism** — count how many playlists contain each track
4. Recommend the **top-10 tracks** by vote count
5. Compare against the test playlist's actual tracks and compute metrics

### Scripts

| Script | Purpose |
|--------|---------|
| `embeddings/playlists_embeddings_final.py` | Encode all playlist titles with fine-tuned model |
| `similarity/testset_test_model_fast.py` | Batch evaluation on full test set |

### How to Run

```bash
# Step 1: Generate playlist embeddings (once, ~minutes on GPU)
python3 embeddings/playlists_embeddings_final.py

# Step 2: Evaluate on test set (~5 minutes on RTX 4090)
python3 similarity/testset_test_model_fast.py
```

### Test Set

- **59,071 test playlists** across **157 clusters**
- Source: `data/clusters/clusters_test.csv`

### Results

Evaluated at cutoffs @10, @66, and @500:

| Metric | @10 | @66 | @500 |
|--------|-----|-----|------|
| Precision | 0.1898 | 0.1185 | 0.0487 |
| Recall | 0.0413 | 0.1456 | 0.3971 |
| MRR | 0.3436 | 0.3537 | 0.3547 |
| R-Precision | 0.1573 | 0.1573 | 0.1573 |
| NDCG | 0.3851 | 0.4221 | 0.4527 |
| HIT | 0.1906 | 0.1745 | 0.3971 |

Results saved to: `evaluation_results_cross_entropy.csv`

### Comparison with Paper

| Metric | Our Model | Paper (FT-C) |
|--------|-----------|-------------|
| Precision@66 | 0.1185 | 0.1228 |
| Recall@66 | 0.1456 | 0.1383 |
| NDCG@66 | 0.4221 | 0.4311 |
| MRR@66 | 0.3537 | 0.3542 |
| R-Precision@66 | 0.1573 | 0.1332 |

Results closely reproduce the paper. Small differences are expected due to random seed and early stopping at epoch 12 (paper ran 100 epochs).

---

## Phase 2 — Re-ranking Evaluation

### Motivation

The baseline recommends the most popular/voted tracks, which leads to:
- **Popularity bias** — same mainstream songs recommended repeatedly
- **Low diversity** — similar tracks clustered together
- **No novelty** — less-known but relevant tracks never appear

The re-ranking layer addresses this by optimizing for three goals simultaneously.

### Re-ranking Score Formula

For each candidate track:

```
Score(t) = alpha * Relevance(t)
         + beta  * Diversity(t)
         + gamma * Novelty(t)
```

Default weights: `alpha=0.6, beta=0.25, gamma=0.15`

| Component | Formula | Source |
|-----------|---------|--------|
| **Relevance** | `vote_count / max_votes` | MPD voting |
| **Diversity** | mean cosine distance to already-selected tracks | Track embeddings |
| **Novelty** | `1 / log2(popularity + 2)` | Spotify CSV metadata |

### Selection Process (Greedy)

1. Take **top-100 candidates** by vote count
2. Iteratively select tracks one by one:
   - Score all remaining candidates
   - Pick the one with the highest score
   - Add it to the selected list (updates diversity for next round)
3. Return the top-10 selected tracks

Diversity is **dynamic** — it penalizes candidates that are too similar to tracks already selected.

### Data Sources

| Data | File | Used For |
|------|------|---------|
| Track embeddings | `data/embeddings/track_embeddings.pkl` | Diversity score |
| Track/artist metadata | `spotify_replace/featured_Spotify_track_info.csv` | Popularity |
| Artist genres | `spotify_replace/CLEANED_featured_Spotify_artist_info.csv` | Genre evaluation metric |
| Enriched metadata | `data/spotify_track_metadata.csv` | Runtime lookup |

**Coverage note:** The Spotify CSVs cover ~15k tracks and ~7.6k artists.
Against the full test set (527k unique tracks):
- Popularity matched: ~0.2% (tracks not matched use fallback popularity = 50)
- Genres matched: ~22.5% of unique tracks (59% of playlist rows)

### Metrics

The re-ranking evaluation computes all original metrics **plus** new diversity/novelty metrics:

| Metric | Type | Goal |
|--------|------|------|
| Precision@10 | Relevance | Higher is better |
| Recall@10 | Relevance | Higher is better |
| NDCG@10 | Relevance | Higher is better |
| MRR@10 | Relevance | Higher is better |
| R-Precision | Relevance | Higher is better |
| HIT@10 | Relevance | Higher is better |
| intra_list_diversity | Diversity | Higher = more diverse tracks |
| avg_popularity | Novelty | Lower = less popularity bias |
| genre_diversity | Diversity | Higher = more genres covered |
| catalog_coverage | Coverage | Higher = broader catalog explored |

### Scripts

| Script | Purpose |
|--------|---------|
| `reranking/spotify_enrichment.py` | Build enriched metadata from CSVs |
| `reranking/track_embeddings.py` | Compute track embeddings with fine-tuned model |
| `reranking/reranker.py` | Core greedy re-ranking algorithm (importable module) |
| `reranking/evaluate_improved.py` | Full evaluation — baseline or improved mode |
| `reranking/compare_results.py` | Side-by-side comparison table |
| `reranking/run_reranking_pipeline.py` | Orchestrator — runs all steps in order |

### How to Run

```bash
# Run the full re-ranking pipeline (all 5 steps)
python reranking/run_reranking_pipeline.py

# Check current status of each step
python reranking/run_reranking_pipeline.py --list

# Run from a specific step
python reranking/run_reranking_pipeline.py --from 3

# Run a single step
python reranking/run_reranking_pipeline.py --step 5
```

### Pipeline Steps

| Step | Script | Output | Notes |
|------|--------|--------|-------|
| 1 | `spotify_enrichment.py` | `data/spotify_track_metadata.csv` | CSV join, no GPU needed |
| 2 | `track_embeddings.py` | `data/embeddings/track_embeddings.pkl` | GPU required, ~minutes |
| 3 | `evaluate_improved.py --mode baseline` | `evaluation_results_reranking_baseline.csv` | GPU required |
| 4 | `evaluate_improved.py --mode improved` | `evaluation_results_reranking_improved.csv` | GPU required |
| 5 | `compare_results.py` | `docs/comparison/comparison_TIMESTAMP.txt` | No GPU needed |

### Re-run From Scratch

To force a full re-run, delete the output files:

```bash
rm evaluation_results_reranking_baseline.csv
rm evaluation_results_reranking_improved.csv
```

Then run:
```bash
python reranking/run_reranking_pipeline.py
```

### Expected Trade-offs

The re-ranker intentionally trades a small amount of relevance for gains in diversity and novelty:

| Metric | Expected Change |
|--------|----------------|
| Precision / NDCG / MRR | Slight decrease |
| intra_list_diversity | Increase |
| genre_diversity | Increase |
| avg_popularity | Decrease (less popular bias) |
| catalog_coverage | Increase |

Comparison results are saved automatically to `docs/comparison/` after each run.

---

## File Reference

### Input Data

| File | Description |
|------|-------------|
| `data/clusters/clusters_test.csv` | 59,071 test playlists with cluster labels |
| `data/output/items.csv` | Playlist-track associations (~66M rows) |
| `data/output/tracks.csv` | Track metadata: name, artist, album (~2.2M tracks) |
| `data/embeddings/playlists_embeddings.pkl` | Playlist title embeddings (fine-tuned model) |
| `data/embeddings/track_embeddings.pkl` | Track embeddings for diversity scoring |
| `data/spotify_track_metadata.csv` | Enriched track metadata: popularity + genres |

### Output Files

| File | Description |
|------|-------------|
| `evaluation_results_cross_entropy.csv` | Baseline evaluation results (59,071 rows) |
| `evaluation_results_reranking_baseline.csv` | Re-ranking pipeline baseline results |
| `evaluation_results_reranking_improved.csv` | Re-ranking pipeline improved results |
| `docs/comparison/comparison_*.txt` | Timestamped comparison reports |
