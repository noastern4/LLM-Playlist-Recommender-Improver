"""
Playlist Recommendation Pipeline
================================
This script runs all steps of the Language Model-Based Playlist Generation Recommender.

Usage:
    python run_pipeline.py              # Run all steps
    python run_pipeline.py --step 1     # Run specific step
    python run_pipeline.py --from 3     # Run from step 3 onwards
"""

import os
import subprocess
import sys

# === CONFIGURATION ===
PROJECT_ROOT = '/home/noama1/recomendation_system/LLM-Playlist-Recommender-Improver'
DATASET_PATH = '/home/noama1/.cache/kagglehub/datasets/himanshuwagh/spotify-million/versions/1/data'
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'output')


def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"▶ {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")

    result = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)

    if result.returncode == 0:
        print(f"✓ {description} - COMPLETED")
    else:
        print(f"✗ {description} - FAILED (exit code: {result.returncode})")
        return False
    return True


def step_0_setup():
    """Setup and verify paths."""
    print("\n" + "="*60)
    print("STEP 0: Setup and Configuration")
    print("="*60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")

    if os.path.exists(DATASET_PATH):
        files = [f for f in os.listdir(DATASET_PATH) if f.endswith('.json')]
        print(f"✓ Dataset found! {len(files)} JSON files available.")
        return True
    else:
        print("✗ Dataset not found!")
        print("  Run: python p.py")
        return False


def step_1_transform():
    """Transform JSON to CSV."""
    return run_command(
        "python3 transform-dataset/json2csv.py",
        "Step 1: Transform Dataset (JSON to CSV)"
    )


def step_2_embeddings():
    """Generate track embeddings."""
    return run_command(
        "python3 clustering-no-split/embeddings/track_embeddings_no-split.py",
        "Step 2: Generate Track Embeddings"
    )


def step_3_clustering():
    """Apply K-means clustering."""
    success = run_command(
        "python3 clustering-no-split/clusters/clustering-no-split.py",
        "Step 3a: Apply K-means Clustering"
    )
    if success:
        success = run_command(
            "python3 clustering-no-split/clusters/percent-no-split.py",
            "Step 3b: Calculate Percentage Matches"
        )
    return success


def step_4_clean():
    """Clean clusters."""
    return run_command(
        "python3 clustering-no-split/clean/clean-clusters-no-split.py",
        "Step 4: Clean Clusters"
    )


def step_5_split():
    """Split into train/test/validation."""
    return run_command(
        "python3 clustering-no-split/split/split_represented.py",
        "Step 5: Split Data (Train/Test/Validation)"
    )


def step_6_finetune():
    """Fine-tune the model."""
    success = run_command(
        "python3 finetuning/cross_entropy_model_finetuning.py",
        "Step 6a: Fine-tune with Cross-Entropy Loss"
    )
    if success:
        success = run_command(
            "python3 finetuning/finetuning_triplet_loss.py",
            "Step 6b: Fine-tune with Triplet Loss"
        )
    return success


def step_7_playlist_embeddings():
    """Generate playlist embeddings."""
    return run_command(
        "python3 embeddings/playlists_embeddings_final.py",
        "Step 7: Generate Playlist Embeddings"
    )


def step_8_recommend():
    """Generate and evaluate recommendations."""
    run_command(
        "python3 similarity/test_1_playlist_finetuned-model.py",
        "Step 8a: Test on Single Playlist"
    )
    run_command(
        "python3 similarity/recommend.py",
        "Step 8b: Generate Recommendations"
    )
    run_command(
        "python3 similarity/testset_test_model.py",
        "Step 8c: Evaluate on Test Set"
    )
    return True


# All steps in order
STEPS = {
    0: ("Setup", step_0_setup),
    1: ("Transform Dataset", step_1_transform),
    2: ("Generate Embeddings", step_2_embeddings),
    3: ("Clustering", step_3_clustering),
    4: ("Clean Clusters", step_4_clean),
    5: ("Split Data", step_5_split),
    6: ("Fine-tune Model", step_6_finetune),
    7: ("Playlist Embeddings", step_7_playlist_embeddings),
    8: ("Recommendations", step_8_recommend),
}


def print_menu():
    """Print available steps."""
    print("\n" + "="*60)
    print("PLAYLIST RECOMMENDATION PIPELINE")
    print("="*60)
    print("\nAvailable steps:")
    for num, (name, _) in STEPS.items():
        print(f"  {num}. {name}")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the playlist recommendation pipeline")
    parser.add_argument("--step", type=int, help="Run a specific step (0-8)")
    parser.add_argument("--from", dest="from_step", type=int, help="Run from this step onwards")
    parser.add_argument("--list", action="store_true", help="List all steps")
    args = parser.parse_args()

    if args.list:
        print_menu()
        return

    if args.step is not None:
        # Run single step
        if args.step in STEPS:
            name, func = STEPS[args.step]
            func()
        else:
            print(f"Invalid step: {args.step}. Use --list to see available steps.")
        return

    if args.from_step is not None:
        # Run from specific step
        start = args.from_step
    else:
        # Run all steps
        start = 0

    print_menu()
    print(f"Running steps {start} to 8...\n")

    for step_num in range(start, 9):
        if step_num in STEPS:
            name, func = STEPS[step_num]
            success = func()
            if not success and step_num > 0:
                print(f"\n⚠ Pipeline stopped at step {step_num}")
                print("Fix the issue and run: python run_pipeline.py --from", step_num)
                break

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()








# FROM TERMINAL:
# # Step 2: Generate embeddings
# python3 clustering-no-split/embeddings/track_embeddings_no-split.py

# # Step 3: Clustering
# python3 clustering-no-split/clusters/clustering-no-split.py
# python3 clustering-no-split/clusters/percent-no-split.py

# # Step 4: Clean
# python3 clustering-no-split/clean/clean-clusters-no-split.py

# # Step 5: Split
# python3 clustering-no-split/split/split_represented.py

# # Step 6: Fine-tune
# python3 finetuning/cross_entropy_model_finetuning.py
# python3 finetuning/finetuning_triplet_loss.py


# # Step 7: Playlist embeddings
# python3 embeddings/playlists_embeddings_final.py

# # Step 8: Recommendations
# python3 similarity/recommend.py



#################### 14/02/2026  UPDATED?  ####################

#  Step 7 — Generate playlist embeddings:
#   python3 embeddings/playlists_embeddings_final.py

#   Step 8 — Evaluate:
#   python3 similarity/test_1_playlist_finetuned-model.py
#   python3 similarity/testset_test_model.py
#   python3 similarity/recommend.py


