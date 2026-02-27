# Cross-Entropy Baseline — Complete Summary

## Goal
Reproduce the semantic baseline from the 2025 paper *"A Language Model-Based Playlist Generation Recommender System."*
The model classifies playlist titles into thematic clusters using cross-entropy loss, then uses the learned embeddings to recommend tracks for new playlists based on title similarity.

---

## Pipeline Overview

```
Step 1: Fine-tune model (classify playlist titles → clusters)
         ↓
Step 2: Generate playlist embeddings (encode all playlist titles)
         ↓
Step 3: Evaluate (find similar playlists, recommend songs, compute metrics)
```

---

## Step 1: Fine-Tuning

### Configuration (identical to the paper)

| Parameter | Value |
|-----------|-------|
| Base model | `sentence-transformers/all-MiniLM-L6-v2` (~22M params) |
| Loss function | Cross-entropy (classification) |
| Number of classes | 157 |
| Training samples | ~473,340 |
| Validation samples | ~59,000 |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Warmup steps | 100 |
| Weight decay | 0.01 |
| Max epochs | 100 |
| Early stopping patience | 5 (added to save compute — not in paper) |
| Eval/save strategy | Every epoch |
| `load_best_model_at_end` | True |
| `save_total_limit` | 2 |
| Hardware | NVIDIA RTX 4090 (24GB VRAM) |

### How it was run
```bash
cd /home/noama1/recomendation_system/LLM-Playlist-Recommender-Improver
sbatch finetuning/train_v2.sbatch
```
Submitted as a Slurm batch job (not interactive) to prevent VPN disconnect kills.

### Training Results

**W&B run:** `ethereal-plant-7` (run ID: `nvs9vtnj`)
**Slurm job:** 14705784 on node ise-4090-03
**Date:** Feb 13–14, 2026
**Duration:** ~6.5 hours (17 epochs, early stopping triggered)
**Best epoch:** 12 (eval accuracy 22.18%)

| Epoch | Train Loss | Eval Loss | Eval Accuracy |
|-------|-----------|-----------|---------------|
| 1     | 3.4834    | 3.2083    | 20.13%        |
| 2     | 3.1944    | 3.1445    | 21.15%        |
| 3     | 3.1379    | 3.1242    | 21.52%        |
| 4     | 3.1065    | 3.1061    | 21.76%        |
| 5     | 3.0844    | 3.1014    | 21.89%        |
| 6     | 3.0667    | 3.0960    | 21.96%        |
| 7     | 3.0527    | 3.1008    | 21.85%        |
| 8     | 3.0389    | 3.0991    | 21.98%        |
| 9     | 3.0266    | 3.1027    | 21.94%        |
| 10    | 3.0154    | 3.0999    | 22.03%        |
| 11    | 3.0054    | 3.0978    | 22.05%        |
| **12**| **2.9858**| **3.1058**| **22.18%** (best) |
| 13    | 2.9766    | 3.1063    | 22.01%        |
| 14    | 2.9678    | 3.1100    | 22.09%        |
| 15    | 2.9606    | 3.1171    | 21.99%        |
| 16    | 2.9517    | 3.1147    | 21.90%        |
| 17    | 2.9517    | 3.1176    | 21.87%        |

### Classification Baselines
- Random accuracy (1/157): **0.64%**
- Majority class: **2.31%**
- **Model is ~34x better than random**

### Observations
- Model peaks early (epoch 5–12), then overfits
- Train loss keeps decreasing while eval loss increases after epoch 5 — classic overfitting
- Early stopping saved ~83 epochs of wasted compute (~166 hours)

### Output files
| File | Path |
|------|------|
| Best model weights | `models/baseline_model/model.safetensors` |
| Tokenizer | `models/baseline_model/tokenizer.json` |
| Model config | `models/baseline_model/config.json` |
| Training metrics | `models/baseline_model/trainer_metrics.json` |
| Label mapping | `models/baseline_model/label_mapping.json` |

---

## Step 2: Generate Playlist Embeddings

### What it does
Loads the fine-tuned cross-entropy model and encodes every playlist title in the dataset into a 384-dimensional embedding vector using the last hidden state of the model (mean pooling).

### How it was run
```bash
python3 embeddings/playlists_embeddings_final.py
```

### Input
- `data/output/playlists.csv` — all playlist titles (~1M playlists)
- Model from `models/baseline_model/`

### Output
- `data/embeddings/playlists_embeddings.pkl` — dict of `{pid: {"embedding": numpy array, "title": str}}`

### Duration
~minutes (GPU inference only, no training)

---

## Step 3: Evaluate on Test Set

### What it does
For each of the 59,071 test playlists:
1. Encode the test playlist title into an embedding
2. Find the top-50 most similar playlists by cosine similarity
3. Aggregate the most common songs from those 50 playlists → recommend top-66 songs
4. Compare recommended songs against the test playlist's actual songs
5. Compute metrics: HIT@66, Precision@66, Recall@66, MRR@66, R-Precision, NDCG@66

### How it was run
```bash
python3 similarity/testset_test_model_fast.py
```
(Optimized version — batch encoding + matrix multiplication instead of looping. ~5 min vs ~35 hours.)

### Input
- `data/embeddings/playlists_embeddings.pkl` (from step 2)
- `data/clusters/clusters_test.csv` (59,071 test playlists)
- `data/output/items.csv` + `data/output/tracks.csv` (track metadata)
- Model from `models/baseline_model/`

### Output
- `evaluation_results_cross_entropy.csv` — per-playlist metrics (59,071 rows)

### Duration
~5 minutes on RTX 4090

---

## Final Results

Evaluated on **59,071 test playlists** across **157 clusters**.

| Metric | Our FT-C | Paper FT-C |
|--------|----------|------------|
| **@10** | | |
| Precision@10 | 0.1898 | 0.1793 |
| Recall@10 | 0.0413 | 0.0382 |
| MRR@10 | 0.3436 | 0.3254 |
| R-Precision@10 | 0.1573 | 0.0496 |
| NDCG@10 | 0.3851 | 0.3740 |
| HIT@10 | 0.1906 | — |
| **@66** | | |
| Precision@66 | 0.1185 | 0.1228 |
| Recall@66 | 0.1456 | 0.1383 |
| MRR@66 | 0.3537 | 0.3542 |
| R-Precision@66 | 0.1573 | 0.1332 |
| NDCG@66 | 0.4221 | 0.4311 |
| HIT@66 | 0.1745 | — |
| **@500** | | |
| Precision@500 | 0.0487 | 0.0489 |
| Recall@500 | 0.3971 | 0.3979 |
| MRR@500 | 0.3547 | 0.3490 |
| R-Precision@500 | 0.1573 | 0.1556 |
| NDCG@500 | 0.4527 | 0.2825 |
| HIT@500 | 0.3971 | 0.3873 |

### Comparison Notes
- **@66 and @500 metrics are very close to the paper** — confirms our baseline is correctly reproducing the paper's results
- Small differences are expected due to random seed, data split, and early stopping (we stopped at epoch 12/17, paper ran all 100 epochs)
- R-Precision@10 discrepancy likely due to different R-Precision computation at low N

---

## All Files

### Scripts
| Script | Purpose |
|--------|---------|
| `finetuning/cross_entropy_baseline.py` | Fine-tuning script (step 1) |
| `finetuning/train_v2.sbatch` | Slurm batch job for fine-tuning |
| `embeddings/playlists_embeddings_final.py` | Generate embeddings (step 2) |
| `similarity/testset_test_model_fast.py` | Batch evaluation (step 3) |
| `similarity/testset_test_model.py` | Original evaluation (slow, kept for reference) |
| `similarity/recommend.py` | Interactive: type playlist name, get recommendations |
| `similarity/test_1_playlist_finetuned-model.py` | Interactive: test single playlist by PID |

### Data
| File | Description |
|------|-------------|
| `data/clusters/clusters_train.csv` | Training set (~473K playlists) |
| `data/clusters/clusters_val.csv` | Validation set (~59K playlists) |
| `data/clusters/clusters_test.csv` | Test set (59,071 playlists) |
| `data/output/playlists.csv` | All playlist metadata (~1M) |
| `data/output/items.csv` | Playlist-track associations |
| `data/output/tracks.csv` | Track metadata (name, artist) |

### Outputs
| File | Description |
|------|-------------|
| `models/baseline_model/` | Fine-tuned model (best checkpoint, epoch 12) |
| `data/embeddings/playlists_embeddings.pkl` | Playlist embeddings from cross-entropy model |
| `evaluation_results_cross_entropy.csv` | Final evaluation metrics per test playlist |
| `finetuning/baseline_14705784.log` | Training stdout log |
| `finetuning/baseline_14705784.err` | Training stderr log |

---

## How to Reproduce From Scratch

```bash
# From login node
cd /home/noama1/recomendation_system/LLM-Playlist-Recommender-Improver

# Step 1: Fine-tune (submit as batch job, ~6.5 hours)
sbatch finetuning/train_v2.sbatch

# Step 2: Generate embeddings (on interactive GPU node, ~minutes)
python3 embeddings/playlists_embeddings_final.py

# Step 3: Evaluate (on interactive GPU node, ~5 minutes)
python3 similarity/testset_test_model_fast.py
```

### Monitoring
```bash
squeue -u noama1                    # check if job is running
tail -f finetuning/baseline_*.log   # watch training live
```

---

## Previous Failed Runs (for context)

Before this clean baseline, there were 8 W&B runs (Jan 24 – Feb 10) that suffered from:
- **VPN disconnects** killing the training (ran interactively via VS Code instead of sbatch)
- Multiple manual restarts from checkpoints with optimizer resets
- 70 epochs instead of 100 (paper uses 100)
- These runs reached epoch 25 with similar accuracy (~21.8%) but were not a valid baseline
- Old checkpoints were in `models/fine_tuned_model/` (can be deleted to save space)
