#!/bin/bash
#SBATCH --partition=gpu-shared
#SBATCH --account=<YOUR_ACCOUNT_NAME>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --job-name=qrc48
#SBATCH --output=logs/qrc48_%j.log
#SBATCH --error=logs/qrc48_%j.err

echo "Loading environment..."
source ~/.bashrc
conda activate llama_env

# ================================
# ENSURE USER SETS TOKEN
# ================================
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: Please export HF_TOKEN before submitting the job."
    echo "Example: export HF_TOKEN=hf_xxx..."
    exit 1
fi

echo "Using HF_TOKEN (hidden)."

# ================================
# PRINT CACHE LOCATIONS
# ================================
echo "HuggingFace cache directory: $HOME/.cache/huggingface"
mkdir -p $HOME/.cache/huggingface

# ================================
# START THE JOB
# ================================
echo "Starting QRC sweep..."
python eval_qrc_sweep.py

echo "Job complete."
