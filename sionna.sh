#!/bin/sh
#SBATCH --job-name=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=50:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "===================================================="
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "===================================================="

# ── Paramètres ────────────────────────────────────────────────────────
# Usage: sbatch run.sh mon_script.py [gpu_id]
# Exemples:
#   sbatch run.sh main_w.py        → GPU 0 par défaut
#   sbatch run.sh main_w.py 1      → GPU 1
#   sbatch run.sh main_w.py 2      → GPU 2

if [ -z "$1" ]; then
    echo "❌ ERREUR: Aucun script Python spécifié."
    echo "Usage: sbatch run.sh nom_du_script.py [gpu_id]"
    exit 1
fi

SCRIPT_PYTHON=$1
GPU_ID=${2:-0}        # GPU 0 par défaut si non spécifié

echo "🚀 Script   : $SCRIPT_PYTHON"
echo "🖥️  GPU ID   : $GPU_ID"

mkdir -p logs

source /export/tmp/sala/miniconda/etc/profile.d/conda.sh
conda activate tf-gpu

# Force TensorFlow à utiliser uniquement le GPU spécifié
export CUDA_VISIBLE_DEVICES=$GPU_ID

SCRIPT_REALPATH=$(realpath $SCRIPT_PYTHON)
cd $(dirname $SCRIPT_REALPATH)

python -u $(basename $SCRIPT_REALPATH)

echo "===================================================="
echo "Job finished on $(date)"
echo "===================================================="