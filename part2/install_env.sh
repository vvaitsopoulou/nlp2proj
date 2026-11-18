#!/bin/bash
# One-time environment build, run as a short MIG job so the CUDA check is real.
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --job-name=nlp2_env
#SBATCH --output=slurm_env_%A.out

set -euo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1

# accept channel terms of service (first time only; harmless if already accepted)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

conda create -n nlp2 python=3.11 -y
source activate nlp2

pip install --upgrade pip
pip install torch numpy scikit-learn scipy nltk datasets pandas tabulate

# pre-stage NLTK data into home so compute nodes read it offline later
python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','stopwords','wordnet','omw-1.4']]"

# verify the CUDA build (a CPU-only torch installs fine and fails only in training)
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.version.cuda)"
