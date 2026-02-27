import os
import csv
import math
import pickle
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_fine_tuned_model(model_dir):
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected.")
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, output_hidden_states=True)
    model.eval()
    model.to(device)

    return tokenizer, model, device

def get_embeddings_batch(texts, tokenizer, model, device, batch_size=256):
    """Encode a list of texts into embeddings in batches."""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding test playlists", unit="batch"):
        batch_texts = texts[i:i+batch_size]
        with torch.no_grad():
            inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True).to(device)
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            last_hidden = outputs.hidden_states[-1]
            embeddings = last_hidden.mean(dim=1)
        all_embeddings.append(embeddings)
    return torch.cat(all_embeddings, dim=0)

def load_playlist_embeddings(embeddings_file):
    with open(embeddings_file, 'rb') as f:
        playlist_embeddings = pickle.load(f)
    return playlist_embeddings

def load_playlist_tracks_with_artists(items_csv, tracks_csv):
    track_metadata = {}
    with open(tracks_csv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading track metadata", unit="track"):
            track_metadata[row["track_uri"]] = {
                "track_name": row["track_name"],
                "artist_name": row["artist_name"],
            }

    playlist_tracks = {}
    with open(items_csv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading playlist tracks", unit="playlist"):
            pid_str = row["pid"].strip()
            track_uri = row["track_uri"]
            if pid_str not in playlist_tracks:
                playlist_tracks[pid_str] = []
            if track_uri in track_metadata:
                playlist_tracks[pid_str].append(track_metadata[track_uri])

    return playlist_tracks

def get_top_songs_with_artists(similar_playlists, playlist_tracks, top_k=500):
    song_counter = Counter()
    for pid, _ in similar_playlists:
        pid_str = str(pid)
        if pid_str in playlist_tracks:
            for track_info in playlist_tracks[pid_str]:
                pair = (track_info["track_name"], track_info["artist_name"])
                song_counter[pair] += 1
    return song_counter.most_common(top_k)

def compute_metrics(recommended_songs, relevant_songs, top_n=10):
    G_T = set(relevant_songs)
    G_A = set(artist for _, artist in relevant_songs)

    R = len(G_T)
    top_r = recommended_songs[:R]
    S_T = set(top_r)
    S_A = set(artist for _, artist in top_r)

    exact_matches = S_T & G_T
    matched_artists = S_A & G_A
    track_score = len(exact_matches)
    artist_score = len(matched_artists) * 0.25
    r_precision = (track_score + artist_score) / R if R > 0 else 0.0

    hits = sum(1 for song in recommended_songs[:top_n] if song in G_T)
    hit_score = hits / min(top_n, len(G_T)) if len(G_T) > 0 else 0.0
    precision = hits / len(recommended_songs[:top_n]) if len(recommended_songs[:top_n]) > 0 else 0.0
    recall = hits / len(G_T) if len(G_T) > 0 else 0.0

    mrr = 0.0
    for i, song in enumerate(recommended_songs[:top_n]):
        if song in G_T:
            mrr = 1 / (i + 1)
            break

    relevance_list = [1 if song in G_T else 0 for song in recommended_songs[:top_n]]

    def dcg(rel):
        return sum(rel_i / math.log2(idx + 2) for idx, rel_i in enumerate(rel))

    dcg_val = dcg(relevance_list)
    ideal_rel = sorted(relevance_list, reverse=True)
    idcg_val = dcg(ideal_rel)
    ndcg = dcg_val / idcg_val if idcg_val > 0 else 0.0

    return {
        "HIT": hit_score,
        "Precision": precision,
        "Recall": recall,
        "MRR": mrr,
        "R-Precision": r_precision,
        "NDCG": ndcg
    }

def main():
    model_dir = "/home/noama1/recomendation_system/LLM-Playlist-Recommender-Improver/models/baseline_model"
    playlist_embeddings_file = "/home/noama1/recomendation_system/LLM-Playlist-Recommender-Improver/data/embeddings/playlists_embeddings.pkl"
    items_csv = "/home/noama1/recomendation_system/LLM-Playlist-Recommender-Improver/data/output/items.csv"
    tracks_csv = "/home/noama1/recomendation_system/LLM-Playlist-Recommender-Improver/data/output/tracks.csv"
    clusters_test_csv = "/home/noama1/recomendation_system/LLM-Playlist-Recommender-Improver/data/clusters/clusters_test.csv"

    results_csv = "/home/noama1/recomendation_system/LLM-Playlist-Recommender-Improver/evaluation_results_cross_entropy.csv"

    # Cutoffs matching the paper
    CUTOFFS = [10, 66, 500]

    # Load model
    tokenizer, model, device = load_fine_tuned_model(model_dir)
    print("Loaded the model.")

    # Load precomputed playlist embeddings and build matrix ONCE
    playlist_embeddings = load_playlist_embeddings(playlist_embeddings_file)
    print("Loaded the playlists.")
    pids = list(playlist_embeddings.keys())
    all_embs_np = np.stack([playlist_embeddings[pid]["embedding"] for pid in pids], axis=0)
    all_embs_torch = torch.from_numpy(all_embs_np).to(device)
    all_embs_norm = F.normalize(all_embs_torch, dim=1)
    print(f"Built embedding matrix: {all_embs_norm.shape}")

    # Load tracks
    playlist_tracks = load_playlist_tracks_with_artists(items_csv, tracks_csv)
    print("Loaded the tracks.")

    # Read all test playlists
    test_rows = []
    with open(clusters_test_csv, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_rows.append(row)
    print(f"Test playlists: {len(test_rows)}")

    # Batch encode all test playlist titles
    test_titles = [row["Playlist Title"].strip() for row in test_rows]
    test_embs = get_embeddings_batch(test_titles, tokenizer, model, device, batch_size=256)
    test_embs_norm = F.normalize(test_embs, dim=1)

    # Process in chunks
    print("Computing similarity matrix...")
    chunk_size = 1000
    all_results = []

    for start in tqdm(range(0, len(test_rows), chunk_size), desc="Evaluating", unit="chunk"):
        end = min(start + chunk_size, len(test_rows))
        chunk_embs = test_embs_norm[start:end]

        sim_matrix = torch.mm(chunk_embs, all_embs_norm.t())

        top_values, top_indices = torch.topk(sim_matrix, k=50, dim=1)
        top_values = top_values.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        for i in range(end - start):
            row = test_rows[start + i]
            cluster_id = row["Cluster ID"]
            test_pid = row["Playlist ID"].strip()

            similar_playlists = [(pids[idx], top_values[i][j]) for j, idx in enumerate(top_indices[i])]

            # Get top 500 songs (max cutoff)
            top_songs = get_top_songs_with_artists(similar_playlists, playlist_tracks, top_k=500)

            relevant_songs_info = playlist_tracks.get(test_pid, [])
            relevant_songs = list({
                (trk["track_name"], trk["artist_name"]) for trk in relevant_songs_info
            })

            recommended_songs = [song_artist for song_artist, _ in top_songs]

            # Compute metrics at each cutoff
            result_row = {
                "Cluster ID": cluster_id,
                "Playlist ID": test_pid,
                "Playlist Title": test_titles[start + i],
            }
            for n in CUTOFFS:
                metrics = compute_metrics(recommended_songs, relevant_songs, top_n=n)
                result_row[f"HIT@{n}"] = metrics["HIT"]
                result_row[f"Precision@{n}"] = metrics["Precision"]
                result_row[f"Recall@{n}"] = metrics["Recall"]
                result_row[f"MRR@{n}"] = metrics["MRR"]
                result_row[f"R-Precision@{n}"] = metrics["R-Precision"]
                result_row[f"NDCG@{n}"] = metrics["NDCG"]

            all_results.append(result_row)

    # Build fieldnames
    fieldnames = ["Cluster ID", "Playlist ID", "Playlist Title"]
    for n in CUTOFFS:
        for metric in ["HIT", "Precision", "Recall", "MRR", "R-Precision", "NDCG"]:
            fieldnames.append(f"{metric}@{n}")

    with open(results_csv, 'w', encoding='utf8', newline='') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    # Print summary matching paper format
    import pandas as pd
    df = pd.DataFrame(all_results)
    print(f"\n{'='*50}")
    print(f"CROSS-ENTROPY BASELINE RESULTS (FT-C)")
    print(f"Test playlists: {len(df)}")
    print(f"{'='*50}")
    print(f"{'Metric':<20} {'Value':>10}")
    print(f"{'-'*30}")
    for n in CUTOFFS:
        for metric in ["Precision", "Recall", "MRR", "R-Precision", "NDCG", "HIT"]:
            col = f"{metric}@{n}"
            print(f"{col:<20} {df[col].mean():>10.4f}")
        print(f"{'-'*30}")

    print(f"\nResults saved in '{results_csv}'.")

if __name__ == "__main__":
    main()
