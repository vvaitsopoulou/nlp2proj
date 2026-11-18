#!/bin/bash
# ---- Step 4 confirmation: combined winner on seeds 0/3/5 (SBU-minimal) ----
# One MIG slice (64 SBU/h). The 'final' axis runs 4 configs x 3 seeds x 5 folds;
# seed-0 folds for mean/meanmax/fine-tuned are reused from the checkpoint, so
# only COMBINED is new on seed 0. Two of the four configs fine-tune the full
# embedding table, so this is slower than the frozen screen -> 3h wall.
# It is checkpointed: if it hits the wall, just resubmit and it resumes.
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9          # the MIG slice's share — do NOT raise
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --job-name=nlp2_confirm
#SBATCH --output=slurm_output_%A.out

set -uo pipefail

# Call the env's python DIRECTLY (no module load / conda activate — see job_sweep.sh).
PY="$HOME/.conda/envs/nlp2/bin/python"
REPO="$HOME/SNLI-stage-2-nlp"
GLOVE="/scratch-shared/$USER/glove"      # holds glove.6B.{50,100,300}d.txt
RESULTS="$HOME/nlp2/results_p2"          # same folder + checkpoint as the screen
export HF_HOME="/scratch-shared/$USER/hf_cache"

cd "$REPO"
"$PY" -c "import torch; print('cuda', torch.cuda.is_available())"

"$PY" -u run_sweep.py \
    --glove_dir "$GLOVE" \
    --results "$RESULTS" \
    --seeds 0 3 5 \
    --axes final
