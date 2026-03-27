#!/bin/bash
#SBATCH --chdir=/work/jose.lara/OII-450-1-2024
#SBATCH --job-name=OII_joblib
#SBATCH --partition=CPU
#SBATCH --array=1-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=27
#SBATCH --mem=10G
#SBATCH --time=24:00:00
#SBATCH --output=/work/jose.lara/OII-450-1-2024/Logs/SSH/Log_joblib_%A_%a.out

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mi_tesis

cd /work/jose.lara/OII-450-1-2024
mkdir -p /work/jose.lara/OII-450-1-2024/Logs/SSH

# Etiquetar este lote para poder resumir tiempos desde la BD
export OII_BATCH_ID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

# Evitar pelear por el mismo SQLite en NFS: usar shard por array-task y correrlo en disco local.
SHARDS_DIR="/work/jose.lara/OII-450-1-2024/BD/shards"
SHARD_SRC_DB="$SHARDS_DIR/resultados_${SLURM_ARRAY_TASK_ID}.db"
if [[ ! -f "$SHARD_SRC_DB" ]]; then
  echo "[ERROR] No existe shard DB: $SHARD_SRC_DB" >&2
  echo "       Ejecuta primero: python Scripts/shard_resultados_db.py --shards 3" >&2
  exit 2
fi

RUN_DB_DIR="${SLURM_TMPDIR:-/tmp}"
RUN_DB_PATH="$RUN_DB_DIR/resultados_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.db"

cp -f "$SHARD_SRC_DB" "$RUN_DB_PATH"
export OII_DB_PATH="$RUN_DB_PATH"

# Evitar oversubscription dentro de cada proceso
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -u /work/jose.lara/OII-450-1-2024/main_joblib.py --n-jobs "${SLURM_CPUS_PER_TASK:-27}"

# Copiar BD actualizada de vuelta al /work
cp -f "$RUN_DB_PATH" "$SHARD_SRC_DB"

python -u /work/jose.lara/OII-450-1-2024/Scripts/resumen_tiempos.py \
  --db "$SHARD_SRC_DB" \
  --batch-id "$OII_BATCH_ID" \
  --out "/work/jose.lara/OII-450-1-2024/Logs/resumen_${OII_BATCH_ID}.log"
