#!/bin/bash
# ---- Step 5 confirmation: combined recurrent winner on seeds 0/3/5 ----
# Full A100 (the screen ran here in ~1h). The rnn_final axis is 6 configs; the
# seed-0 folds for lstm[anchor]/gru/maxpool are reused from the screen's
# checkpoint, so only the stacked configs + seeds 3/5 are new. Checkpointed:
# resubmit to resume if it ever hits the wall.
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --job-name=nlp2_rnn_confirm
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
    --seeds 0 3 5 \
    --axes rnn_final
