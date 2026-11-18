#!/bin/bash
# ---- Step 5: recurrent-encoder screen (seed 0), SBU-minimal (one MIG slice) ----
# 8 axes / 13 unique configs x 5 folds. Several configs fine-tune the full
# embedding table (slow), so wall is generous; the run is CHECKPOINTED to the
# same results folder, so if it hits the wall just resubmit and it resumes.
# NOTE: finalise partition + wall AFTER the timing probe (see notebook). If the
# probe says a fold is minutes not seconds, split the axes across 2-3 jobs or
# move to a full A100 (gpu partition, 128 SBU/h) which an RNN saturates better.
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9          # the MIG slice's share — do NOT raise
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --job-name=nlp2_rnn
#SBATCH --output=slurm_output_%A.out

set -uo pipefail

# Call the env's python DIRECTLY (no module load / conda activate — see job_sweep.sh).
PY="$HOME/.conda/envs/nlp2/bin/python"
REPO="$HOME/SNLI-stage-2-nlp"
GLOVE="/scratch-shared/$USER/glove"
RESULTS="$HOME/nlp2/results_p2"          # same folder + checkpoint as Step 4
export HF_HOME="/scratch-shared/$USER/hf_cache"

cd "$REPO"
"$PY" -c "import torch; print('cuda', torch.cuda.is_available())"

"$PY" -u run_sweep.py \
    --glove_dir "$GLOVE" \
    --results "$RESULTS" \
    --seeds 0 \
    --axes rnn_cell rnn_direction rnn_readout rnn_hidden rnn_embtrain rnn_reg rnn_pair rnn_lr
