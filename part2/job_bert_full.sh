#!/bin/bash
# BERT + LSTM on the full SNLI pool. Run `python run_bert_full.py --prepare` on the login node first.
# Embeddings are saved in shards, so if the job hits the time limit just resubmit it.
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --job-name=bert_lstm_full
#SBATCH --output=slurm_bert_full_%A.out

set -uo pipefail

PY="$HOME/.conda/envs/nlp2/bin/python"
REPO="$HOME/nlp2proj/part2"
export HF_HOME="/scratch-shared/$USER/hf_cache"

cd "$REPO"
"$PY" -c "import torch; print('cuda', torch.cuda.is_available())"

"$PY" -u run_bert_full.py \
    --embeddings_dir "/scratch-shared/$USER/bert_full_embeddings" \
    --results results/bert_lstm_full \
    --num_epochs 8
