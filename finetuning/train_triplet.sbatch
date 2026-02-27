#!/bin/bash
#SBATCH --job-name=triplet_bl
#SBATCH --partition=rtx4090
#SBATCH --qos=normal
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/noama1/recomendation_system/LLM-Playlist-Recommender-Improver/finetuning/triplet_%j.log
#SBATCH --error=/home/noama1/recomendation_system/LLM-Playlist-Recommender-Improver/finetuning/triplet_%j.err

echo "============================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "Started at:   $(date)"
echo "============================================"

# Activate conda environment
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate env_deep

# Run triplet loss training from scratch
cd /home/noama1/recomendation_system/LLM-Playlist-Recommender-Improver
python3 finetuning/triplet_loss_baseline.py

echo "============================================"
echo "Finished at:  $(date)"
echo "============================================"
