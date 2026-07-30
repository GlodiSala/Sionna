#!/bin/sh
#SBATCH --job-name=gpu_test             # Nom par défaut (sera écrasé si tu utilises -J)
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=100:00:00
#SBATCH --output=logs/%x_%j.out         # %x = nom du job, %j = ID du job
#SBATCH --error=logs/%x_%j.err

echo "===================================================="
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "===================================================="

# 1. Vérifier si un paramètre a été fourni
if [ -z "$1" ]; then
    echo "❌ ERREUR: Aucun script Python spécifié."
    echo "Usage: sbatch nom_du_script_sbatch.sh nom_du_script_python.py"
    exit 1
fi

SCRIPT_PYTHON=$1

echo "🚀 Lancement du script : $SCRIPT_PYTHON"

# Créer le dossier logs si nécessaire
mkdir -p logss

# Charger ton Conda local
source /export/tmp/sala/anaconda3/bin/activate
conda activate tf_torch

# 2. Exécuter le script passé en paramètre
python $SCRIPT_PYTHON

echo "===================================================="
echo "Job finished on $(date)"
echo "====================================================" 