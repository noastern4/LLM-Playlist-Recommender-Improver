# Baseline Pipeline — Full Process Guide

## Overview

Two independent baselines, each following the same pipeline:

```
Fine-tune model → Generate embeddings → Evaluate on test set
```

---

## BASELINE 1: Cross-Entropy (DONE)

### Step 1: Fine-tune (COMPLETED)
- **Script:** `finetuning/cross_entropy_baseline.py`
- **Run via:** `sbatch finetuning/train_v2.sbatch`
- **Output:**
  - `models/baseline_model/model.safetensors` — best model weights
  - `models/baseline_model/tokenizer.json` — tokenizer
  - `models/baseline_model/config.json` — model config
  - `models/baseline_model/trainer_metrics.json` — per-epoch metrics
  - `models/baseline_model/label_mapping.json` — cluster ID → label index
- **Status:** DONE (best accuracy 22.18% at epoch 12)

### Step 2: Generate Playlist Embeddings (TODO)
- **Script:** `embeddings/playlists_embeddings_final.py`
- **Run:** `python3 embeddings/playlists_embeddings_final.py`
- **What it does:** Loads the cross-entropy model, encodes every playlist title into an embedding using the last hidden state
- **Input:** `data/output/playlists.csv` + model from `models/baseline_model/`
- **Output:** `data/embeddings/playlists_embeddings.pkl` — dict of {pid: {embedding, title}}
- **Estimated time:** minutes (inference only)

### Step 3: Evaluate on Test Set (TODO)
- **Script:** `similarity/testset_test_model.py`
- **Run:** `python3 similarity/testset_test_model.py`
- **What it does:** For each test playlist, finds top-50 similar playlists by cosine similarity, recommends top-66 songs, computes metrics
- **Input:**
  - `data/embeddings/playlists_embeddings.pkl` (from step 2)
  - `data/clusters/clusters_test.csv` (test playlists)
  - `data/output/items.csv` + `data/output/tracks.csv` (track metadata)
  - Model from `models/baseline_model/`
- **Output:** `evaluation_results_scheduler.csv` (in project root) — per-playlist metrics:
  - HIT@66, Precision@66, Recall@66, MRR@66, R-Precision, NDCG@66

### Step 3b: Interactive Test (OPTIONAL)
- **Script:** `similarity/recommend.py`
- **Run:** `python3 similarity/recommend.py`
- **What it does:** Lets you type a playlist name, shows top similar playlists and recommended songs
- **Input:** Same as step 3 + user input (playlist name)
- **Output:** Printed to terminal (no file saved)

### Step 3c: Single Playlist Test (OPTIONAL)
- **Script:** `similarity/test_1_playlist_finetuned-model.py`
- **Run:** `python3 similarity/test_1_playlist_finetuned-model.py`
- **What it does:** Same as recommend.py but takes a playlist PID and shows metrics
- **Input:** Same as step 3 + user input (PID)
- **Output:** Printed to terminal (no file saved)

### Cross-Entropy Run Order:
```bash
# Step 2
python3 embeddings/playlists_embeddings_final.py

# Step 3 (requires step 2 output)
python3 similarity/testset_test_model.py
```

---

## BASELINE 2: Triplet Loss (TODO)

### Step 1: Fine-tune (TODO)
- **Script:** `finetuning/triplet_loss_baseline.py`
- **Run via:** `sbatch finetuning/train_triplet.sbatch`
- **Output:**
  - `models/triplet_model/` — SentenceTransformer model files
- **Estimated time:** ~30 hours (early stopping with patience=5)

### Step 2: Generate Playlist Embeddings (TODO — NEEDS NEW SCRIPT)
- **Problem:** `embeddings/playlists_embeddings_final.py` uses `AutoModelForSequenceClassification` which only works for the cross-entropy model. The triplet model is a `SentenceTransformer` — different loading method, different embedding extraction.
- **What's needed:** A new embedding script (or modify existing) that loads the model with `SentenceTransformer('models/triplet_model/')` and uses `model.encode()` for embeddings
- **Output:** `data/embeddings/playlists_embeddings_triplet.pkl`

### Step 3: Evaluate on Test Set (TODO — NEEDS UPDATED PATHS)
- **Same scripts as cross-entropy** but pointing to:
  - Triplet model: `models/triplet_model/`
  - Triplet embeddings: `data/embeddings/playlists_embeddings_triplet.pkl`
- **Output:** Separate results CSV for comparison

---

## Summary of All Files Created

### Models
| File | Source | Description |
|------|--------|-------------|
| `models/baseline_model/*` | cross-entropy fine-tune | Classification model (DONE) |
| `models/triplet_model/*` | triplet fine-tune | SentenceTransformer model (TODO) |

### Embeddings
| File | Source | Description |
|------|--------|-------------|
| `data/embeddings/playlists_embeddings.pkl` | step 2 (cross-entropy) | Playlist embeddings from cross-entropy model (TODO) |
| `data/embeddings/playlists_embeddings_triplet.pkl` | step 2 (triplet) | Playlist embeddings from triplet model (TODO) |

### Results
| File | Source | Description |
|------|--------|-------------|
| `evaluation_results_scheduler.csv` | step 3 (cross-entropy) | Per-playlist metrics for cross-entropy baseline (TODO) |
| TBD | step 3 (triplet) | Per-playlist metrics for triplet baseline (TODO) |

### Training Logs
| File | Source | Description |
|------|--------|-------------|
| `finetuning/baseline_14705784.log` | cross-entropy sbatch | Training stdout (DONE) |
| `finetuning/baseline_14705784.err` | cross-entropy sbatch | Training stderr (DONE) |
| `finetuning/triplet_*.log` | triplet sbatch | Training stdout (TODO) |
| `finetuning/triplet_*.err` | triplet sbatch | Training stderr (TODO) |

---

## Execution Order

```
CROSS-ENTROPY PATH:                    TRIPLET PATH:

1. Fine-tune (DONE)                    1. Fine-tune
   sbatch train_v2.sbatch                sbatch train_triplet.sbatch
         |                                      |
         v                                      v
2. Generate embeddings                 2. Generate embeddings (NEW SCRIPT NEEDED)
   python3 embeddings/                    python3 embeddings/
   playlists_embeddings_final.py          playlists_embeddings_triplet.py
         |                                      |
         v                                      v
3. Evaluate                            3. Evaluate
   python3 similarity/                    python3 similarity/
   testset_test_model.py                  testset_test_model_triplet.py
         |                                      |
         v                                      v
   evaluation_results_                    evaluation_results_
   scheduler.csv                          triplet.csv
         |                                      |
         +------ COMPARE RESULTS ------+
```

**Important:** The two paths are completely independent. You can run them in parallel on different nodes, or sequentially. The cross-entropy path is ready to go right now.
