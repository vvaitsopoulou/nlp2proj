#!/bin/bash
# ---- SBU-minimal GPU job: one MIG slice (64 SBU/h, half of a full A100) ----
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9          # the MIG slice's share — do NOT raise (over a slice doubles cost)
#SBATCH --mem=60G                  # included in the slice
#SBATCH --time=02:00:00            # + ~20% margin; resubmit to resume from the checkpoint
#SBATCH --job-name=nlp2_sweep
#SBATCH --output=slurm_output_%A.out

set -uo pipefail

# Call the env's python DIRECTLY. No `module load` / `conda activate`: on this
# cluster they fight over PATH and leave `python` pointing at the base Anaconda.
# The nlp2 torch wheel bundles its own CUDA, so it needs no CUDA module.
PY="$HOME/.conda/envs/nlp2/bin/python"
REPO="$HOME/SNLI-stage-2-nlp"
GLOVE="/scratch-shared/$USER/glove"      # holds glove.6B.{50,100,300}d.txt
RESULTS="$HOME/nlp2/results_p2"          # keep results in home (scratch is wiped)
export HF_HOME="/scratch-shared/$USER/hf_cache"

cd "$REPO"
"$PY" -c "import torch; print('cuda', torch.cuda.is_available())"

"$PY" -u run_sweep.py \
    --glove_dir "$GLOVE" \
    --results "$RESULTS" \
    --seeds 0 \
    --axes composition pooling embedding_size stopwords tokenizer_hyphen head_width head_dropout learning_rate batch_size embedding_training
