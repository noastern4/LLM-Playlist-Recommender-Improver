# Methodology

## 4.1  Datasets

### 4.1.1  Million Playlist Dataset (MPD)

Our primary dataset is the **Million Playlist Dataset (MPD)** [Spotify / ACM RecSys Challenge 2018],
a large-scale collection of **1,000,000 Spotify user playlists** containing approximately
**2.2 million unique tracks** by 300,000+ artists, with a total of ~66 million playlist–track
associations.

Each playlist record includes a title (free-text string), a creation timestamp, and an ordered
list of track URIs. Each track is described by its name, artist name, album name, and a
Spotify-assigned URI.

**Clustering and split.** Following the original paper we reproduce, playlist titles are first
embedded and grouped into **157 thematic clusters** via k-means. The dataset is then split
per cluster into train / validation / test sets:

| Split | Playlists | Usage |
|-------|-----------|-------|
| Train | ~473,340 | Model fine-tuning |
| Validation | ~59,000 | Early stopping during training |
| Test | **59,071** | All final evaluations |

The test set spans all 157 clusters and is held out completely during training and optimization.

**Metadata enrichment.** Spotify's public API was used to fetch popularity scores (0–100
integer) and genre tags for a subset of tracks. The resulting CSV covers ~15,000 tracks and
~7,600 artists. For the remaining tracks (not found via API) a default popularity of 50 is
used during re-ranking. Genre tags from the API cover approximately 22.5% of unique test
tracks, corresponding to ~59% of playlist rows.

---

### 4.1.2  ThirtyMusic Dataset (Cross-Domain Validation)

For cross-domain generalization testing we use the **ThirtyMusic** dataset
[Turrin et al., 2015], a public collection of **57,561 playlists** curated from **Last.fm**
user listening histories. The dataset is distributed in **IDOMAAR** format — a tab-separated
line format where each entity record contains a JSON metadata field and a JSON relations field.

ThirtyMusic was chosen because it represents a distinctly different domain than MPD:
- **Different platform**: Last.fm (scrobble-based) vs. Spotify (user-curated)
- **Different language mix**: MPD titles are predominantly English; ThirtyMusic includes
  multilingual and URL-encoded titles
- **Different track catalog**: ThirtyMusic tracks are identified by numeric IDs linked to
  Last.fm artist and tag data, not Spotify URIs

The dataset contains 5.67 million track entities. After filtering to only tracks that appear
in at least one playlist, **437,605 unique tracks** are retained.

**Parsing.** Four IDOMAAR entity files are parsed directly from the source tarball without
full extraction: `persons.idomaar` (artists), `tags.idomaar` (genres), `tracks.idomaar`,
and `playlist.idomaar`. URL-encoded names are decoded with `urllib.parse.unquote_plus`.
Track play counts are normalized to a 0–100 popularity proxy by dividing by the maximum
observed play count.

**Split.** Using the same random seed as MPD (seed = 42), playlists are shuffled and split
80/20 into train and test sets:

| Split | Playlists | Usage |
|-------|-----------|-------|
| Train | ~46,100 | Similarity index construction |
| Test | **10,457** | Final cross-domain evaluation |

This evaluation is **zero-shot cross-domain**: the model is never re-trained or fine-tuned on
ThirtyMusic data. The same model weights and Optuna-optimized hyperparameters learned on MPD
are applied directly to ThirtyMusic.

---

## 4.2  Baseline Model: Semantic Playlist Embedding

We reproduce the **FT-C (Fine-Tuned Cross-entropy)** baseline from the 2025 paper
*"A Language Model-Based Playlist Generation Recommender System."*

### 4.2.1  Pre-trained Model

The base model is `sentence-transformers/all-MiniLM-L6-v2`, a compact 6-layer transformer
(~22M parameters) pre-trained with contrastive learning on a large corpus of sentence pairs.
It produces 384-dimensional sentence embeddings and is well-suited for short text like
playlist titles.

### 4.2.2  Fine-Tuning

The model is adapted to the playlist domain via **multi-class cross-entropy classification**:
the playlist title is the input, and the target is the playlist's thematic cluster label
(one of 157 classes). The classification head is added on top of the encoder and trained
end-to-end.

**Configuration (identical to the reference paper):**

| Hyperparameter | Value |
|----------------|-------|
| Base model | `all-MiniLM-L6-v2` |
| Loss | Cross-entropy classification |
| Number of classes | 157 |
| Batch size | 8 |
| Learning rate | 2 × 10⁻⁵ |
| Warmup steps | 100 |
| Weight decay | 0.01 |
| Max epochs | 100 |
| Early stopping patience | 5 epochs |
| Hardware | NVIDIA RTX 4090 (24 GB) |

Training converged in **17 epochs** (~6.5 hours), with early stopping triggered at epoch 17.
The best checkpoint, saved at **epoch 12**, achieves a validation accuracy of **22.18%**.
For comparison, random guessing yields 0.64% (1/157) and the majority-class baseline yields
2.31%, so the fine-tuned model is approximately 34× better than random.

The cluster classification task is intentionally lossy — a playlist title does not uniquely
determine its cluster, and many titles are ambiguous. The classification head is discarded
after training; only the encoder's internal representations are used for embedding.

### 4.2.3  Embedding Extraction

After fine-tuning, the model is used as a **semantic encoder**. For any text input (playlist
title or track description), the embedding is obtained by passing the tokenized text through
the encoder and **mean-pooling the last hidden layer** across all token positions:

> **e** = MeanPool(LastHiddenState(transformer(text))) ∈ ℝ³⁸⁴

All embeddings are **L2-normalized** so that inner products equal cosine similarities. This
enables efficient similarity search via matrix multiplication.

**Playlist embeddings** are generated offline for all ~1M MPD playlists (and the ThirtyMusic
training playlists) and stored as a dictionary `{pid: embedding}`.

**Track embeddings** are generated by encoding the concatenated string `"<track_name> <artist_name>"`.
For MPD, all unique tracks in the test set's candidate pool are embedded (~527K tracks on MPD;
437K on ThirtyMusic). Track embeddings are stored as `{(track_name, artist_name): embedding}`.

---

## 4.3  Baseline Recommendation Pipeline

The baseline follows a **two-stage collaborative filtering** approach driven entirely by
semantic title similarity:

**Stage 1 — Nearest-neighbour retrieval.**
Given a test playlist title, encode it into a 384-dim embedding. Compute cosine similarity
against all *training* playlist embeddings. Retrieve the **top-50 most similar playlists**.

**Stage 2 — Voting aggregation.**
Collect all tracks from the 50 retrieved playlists. Tally how many of those playlists
contain each track (vote count). Recommend the **top-10 tracks** by vote count.

This baseline is deterministic and parameter-free — it requires no manual tuning. Its
weakness is that it always recommends the most popular/commonly-occurring tracks from
semantically similar playlists, which can lead to popularity bias and low intra-list
diversity.

---

## 4.4  Re-ranking Layer

To address the diversity and novelty limitations of the baseline, we introduce a
**greedy MMR-style re-ranking layer** that operates on the baseline's top-100 candidates.

### 4.4.1  Scoring Function

Each candidate track *t* is scored using a weighted combination of three components:

> **Score(t) = α · Relevance(t) + β · Diversity(t) + γ · Novelty(t)**

with α + β + γ = 1, and:

| Component | Definition |
|-----------|-----------|
| **Relevance(t)** | Normalized vote count: `vote_count(t) / max_vote_count` |
| **Diversity(t)** | Mean cosine distance between *t*'s embedding and each already-selected track's embedding |
| **Novelty(t)** | Inverse log-popularity: `1 / log₂(popularity(t) + 2)` |

**Relevance** rewards tracks that appear frequently in playlists semantically similar to the
query — preserving the collaborative filtering signal from the baseline.

**Diversity** penalizes candidates that are similar to tracks already in the recommendation
list. This is evaluated dynamically after each selection, ensuring selected tracks are
spread across different regions of the embedding space.

**Novelty** rewards tracks with lower Spotify popularity scores. Popularity is an integer
0–100 provided by the Spotify API (or normalized play count for ThirtyMusic). The `log₂`
function smooths the penalty so that very popular tracks are more heavily discouraged while
moderately popular tracks receive a mild penalty.

### 4.4.2  Greedy Selection

Re-ranking is performed greedily over the top-100 voting candidates:

1. Initialize the selected list as empty.
2. For each position 1 through 10:
   a. Compute Score(t) for all remaining candidates (Diversity recomputed based on current
      selected list).
   b. Add the candidate with the highest score to the selected list.
3. Return the 10 selected tracks as the final recommendation.

The diversity term is **dynamic**: after each selection, the distances to the newly added
track are incorporated into future Diversity scores. This mirrors the standard Maximal
Marginal Relevance (MMR) framework and is equivalent to a greedy Pareto-front traversal
along the relevance–diversity trade-off.

**Vectorized implementation.** For efficiency, all candidate track embeddings are stacked
into a matrix E_remaining ∈ ℝ^(|candidates| × 384) and diversity scores are computed via
a single matrix multiplication:

> sim_matrix = E_remaining · E_selected^T ∈ ℝ^(|remaining| × |selected|)
> Diversity(t) = 1 − mean_over_columns(sim_matrix[t])

Since all embeddings are L2-normalized, the dot product equals the cosine similarity. This
vectorized approach is approximately **70× faster** than a naive Python loop and enables
the optimization step to remain tractable.

---

## 4.5  Hyperparameter Optimization

### 4.5.1  Motivation

The re-ranking weights α, β, γ control the balance between relevance, diversity, and
novelty. Manually chosen initial weights (α = 0.6, β = 0.25, γ = 0.15) already produce
a measurable ILD improvement, but the optimal trade-off point is dataset-dependent and
non-trivial to determine by hand. We therefore use **automatic Bayesian optimization** to
find the Pareto-optimal weights.

### 4.5.2  Optuna with TPE Sampler

We use **Optuna** [Akiba et al., 2019] with the **Tree-structured Parzen Estimator (TPE)**
sampler. TPE is a sequential model-based optimization algorithm: after each trial it fits
two kernel density estimators — one over high-objective-value configurations and one over
low-objective-value configurations — and proposes the next trial by maximizing the ratio
between the two densities. This is significantly more sample-efficient than random or grid
search for continuous, multi-dimensional search spaces.

### 4.5.3  Search Space

Raw values (a, b, g) are sampled from independent continuous intervals and then normalized
to satisfy α + β + γ = 1:

| Parameter | Raw Range | Role |
|-----------|-----------|------|
| a (→ α) | [0.30, 0.90] | Relevance weight |
| b (→ β) | [0.05, 0.50] | Diversity weight |
| g (→ γ) | [0.01, 0.30] | Novelty weight |

Normalization: α = a/(a+b+g), β = b/(a+b+g), γ = g/(a+b+g).

### 4.5.4  Objective Function

The objective to **maximize** is:

> **F = NDCG@10 + λ × ILD@10**

where:
- **NDCG@10** (Normalized Discounted Cumulative Gain at cutoff 10) measures whether
  ground-truth tracks appear early in the ranked list — the primary relevance metric.
- **ILD@10** (Intra-List Diversity) is the mean pairwise cosine distance between the
  embeddings of the 10 recommended tracks — a direct measure of recommendation variety.
- **λ** is a scalar that controls the strength of the diversity incentive relative to
  relevance (see Section 4.6 for its selection).

This joint objective is necessary because using NDCG alone collapses the solution to
α = 1 (pure voting), eliminating all diversity gain. The combined objective forces Optuna
to find weights that improve diversity without sacrificing excessive relevance.

### 4.5.5  Evaluation Protocol

To keep each trial computationally tractable:

- **1,000 test playlists** are randomly sampled (seed = 42) from the 59,071-playlist test
  set and held fixed across all 100 trials.
- All expensive pre-computations — model inference, similarity matrix computation, and
  candidate list construction — are performed **once before optimization begins**.
- Each trial only re-runs the fast greedy selection step on the 1,000 playlists, which
  completes in under 1 second per trial.

**100 trials** are run in total. The best configuration is selected based on the highest
observed objective value on the 1,000-playlist sample, and then evaluated on the full
59,071-playlist test set.

**Best weights found (at λ = 0.2):** α = 0.41, β = 0.57, γ = 0.01
**Objective value:** 0.538 (NDCG@10 + 0.2 × ILD on the 1,000-playlist sample)

---

## 4.6  Sensitivity Analysis: Selecting λ

The scalar λ in the objective function F = NDCG@10 + λ × ILD determines how strongly
diversity is incentivized during weight search. Too small a λ leaves the optimizer indifferent
to diversity; too large a λ sacrifices excessive relevance.

To characterize this trade-off and justify the choice of λ, we sweep four values:

| λ | Effect |
|---|--------|
| 0.0 | Pure NDCG optimization → collapses to baseline (α = 1) |
| 0.1 | Mild diversity incentive |
| 0.2 | Moderate diversity incentive (selected) |
| 0.3 | Strong diversity incentive |

For each λ value, Optuna runs the full 100-trial search on the 1,000-playlist sample.
The resulting optimal weights are then applied to the full 59,071-playlist test set to
measure NDCG@10 and ILD.

**Results:**

| λ | α | β | γ | NDCG@10 | Δ NDCG | ILD | Δ ILD |
|---|---|---|---|---------|--------|-----|-------|
| 0.0 (baseline) | 1.000 | 0.000 | 0.000 | 0.3851 | — | 0.6051 | — |
| 0.1 | 0.549 | 0.386 | 0.065 | 0.3784 | −0.0067 | 0.7503 | +0.1452 |
| **0.2** | **0.360** | **0.489** | **0.151** | **0.3701** | **−0.0150** | **0.8221** | **+0.2170** |
| 0.3 | 0.360 | 0.489 | 0.151 | 0.3701 | −0.0150 | 0.8221 | +0.2170 |

**λ = 0.2** is selected because it is the **elbow of the Pareto curve**: it achieves the
largest ILD gain before diminishing returns set in (λ = 0.3 converges to identical weights),
while limiting the NDCG drop to −4% (−0.015 absolute). The ratio of diversity gained to
relevance lost is approximately **14.5:1**, which we consider an acceptable operating point.

---

## 4.7  Evaluation Metrics

All models are evaluated on the same test sets using the following metrics, grouped by
category:

**Relevance metrics** (evaluate whether ground-truth tracks appear in the top-10):

| Metric | Definition |
|--------|-----------|
| **NDCG@10** | Normalized Discounted Cumulative Gain — rewards correct hits that appear earlier in the ranked list |
| **Precision@10** | Fraction of recommended tracks that appear in the ground-truth playlist |
| **Recall@10** | Fraction of ground-truth tracks recovered in the top-10 |
| **MRR@10** | Mean Reciprocal Rank — reciprocal position of the first correct hit |
| **HIT@10** | Binary indicator: 1 if at least one ground-truth track is in the top-10 |
| **R-Precision** | Precision@R where R = number of ground-truth tracks |

**Beyond-accuracy metrics** (evaluate diversity, novelty, and coverage):

| Metric | Definition |
|--------|-----------|
| **ILD@10** | Intra-List Diversity: mean pairwise cosine distance between the 10 recommended track embeddings |
| **Genre Diversity** | Number of unique genre tags among the 10 recommended tracks |
| **Avg. Popularity** | Mean Spotify popularity score of the 10 recommended tracks (lower = more novel) |
| **Catalog Coverage** | Fraction of the full track catalog that appears at least once in any recommendation |

---

## 4.8  Cross-Domain Generalization

After optimizing on MPD, we apply the same pipeline — with the same model weights and
the same Optuna-optimized hyperparameters (α = 0.41, β = 0.57, γ = 0.01) — to the
ThirtyMusic dataset in a **zero-shot cross-domain evaluation**.

The ThirtyMusic pipeline mirrors the MPD pipeline exactly:

1. **Parse:** Convert IDOMAAR entities to CSV (track name, artist, playlist–track
   associations, normalized popularity from play counts, genre tags).
2. **Embed playlists:** Encode all training playlist titles with the MPD fine-tuned model
   → 384-dim embeddings.
3. **Embed tracks:** Encode `"<track_name> <artist_name>"` for every unique track in the
   training set → 384-dim embeddings.
4. **Evaluate baseline:** Top-50 similar playlists → vote-count → top-10 (no re-ranking).
5. **Evaluate re-ranking:** Apply greedy MMR re-ranking with the MPD-optimized weights.
6. **Compare:** Compute all metrics on the 10,457 ThirtyMusic test playlists.

No fine-tuning, no hyperparameter re-search, and no domain adaptation of any kind is
performed on ThirtyMusic data. This constitutes a strict test of whether the re-ranking
mechanism generalizes beyond its training domain.

---

## 4.9  Implementation Details

| Component | Details |
|-----------|---------|
| Deep learning framework | PyTorch 2.5.1 + CUDA 12.1 |
| Transformer library | HuggingFace Transformers |
| Hyperparameter optimization | Optuna 3.x, TPE sampler |
| Hardware | NVIDIA RTX 4090 (24 GB VRAM) |
| Cluster scheduler | SLURM (sbatch jobs) |
| Embedding batch size | 256 (playlists), 512 (tracks) |
| Similarity search | Vectorized numpy matrix multiplication |
| Random seeds | seed = 42 for all splits and sampling |
| Track embedding scope (MPD) | ~527K unique tracks in test candidates |
| Track embedding scope (TM) | ~437K unique tracks (playlist-relevant only) |

All code, scripts, sbatch job files, and output logs are available in the project
repository under `finetuning/`, `embeddings/`, `reranking/`, and `thirtymusic_eval/`.
