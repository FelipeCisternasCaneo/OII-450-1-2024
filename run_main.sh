#!/bin/bash
#SBATCH --chdir=/work/jose.lara/OII-450-1-2024
#SBATCH --job-name=OII_multicore
#SBATCH --partition=CPU
#SBATCH --array=1-3                  # Usas tus 3 slots permitidos [cite: 89]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=27           # Pides 27 núcleos por cada Job
#SBATCH --mem=10G                     # Subimos un poco la RAM para los 27 procesos
#SBATCH --time=24:00:00              # Límite normal [cite: 76]
#SBATCH --output=/work/jose.lara/OII-450-1-2024/Logs/SSH/Log_%A_%a.out

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mi_tesis

cd /work/jose.lara/OII-450-1-2024

# Nota: Slurm NO crea carpetas para --output. Este mkdir ayuda para el resto de logs,
# pero la carpeta debe existir ANTES de ejecutar `sbatch run_main.sh`.
mkdir -p /work/jose.lara/OII-450-1-2024/Logs/SSH

# Etiquetar este lote para poder resumir tiempos desde la BD sin polling
export OII_BATCH_ID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

# Usamos xargs para lanzar 27 instancias de python en paralelo dentro de este mismo Job
seq 27 | xargs -I{} -P 27 python -u /work/jose.lara/OII-450-1-2024/main.py

# Resumen final (sin monitor): makespan = max(ts_fin) - min(ts_inicio)
python -u /work/jose.lara/OII-450-1-2024/Scripts/resumen_tiempos.py \
	--batch-id "$OII_BATCH_ID" \
	--out "/work/jose.lara/OII-450-1-2024/Logs/resumen_${OII_BATCH_ID}.log"