#!/bin/bash
# ---- Step 5 recurrent screen on a FULL A100 (gpu partition, 128 SBU/h) ----
# Same command/results/checkpoint as job_rnn.sh, so folds already computed on a
# MIG slice are REUSED here (the checkpoint is partition-independent) — switching
# wastes nothing. An LSTM saturates a full A100 far better than a 1/7 MIG slice,
# so this usually finishes in a fraction of the time and, despite the 2x rate,
# often costs FEWER SBU in total. Use when the MIG per-fold time (from the live
# log) says the screen won't fit comfortably on a slice.
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1                   # one full A100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --job-name=nlp2_rnn_a100
#SBATCH --output=slurm_output_%A.out

set -uo pipefail

PY="$HOME/.conda/envs/nlp2/bin/python"
REPO="$HOME/SNLI-stage-2-nlp"
GLOVE="/scratch-shared/$USER/glove"
RESULTS="$HOME/nlp2/results_p2"
export HF_HOME="/scratch-shared/$USER/hf_cache"

cd "$REPO"
"$PY" -c "import torch; print('cuda', torch.cuda.is_available())"

"$PY" -u run_sweep.py \
    --glove_dir "$GLOVE" \
    --results "$RESULTS" \
    --seeds 0 \
    --axes rnn_cell rnn_direction rnn_readout rnn_hidden rnn_embtrain rnn_reg rnn_pair rnn_lr
