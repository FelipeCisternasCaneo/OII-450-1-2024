import argparse
import os
import shutil
import time

from joblib import Parallel, delayed

from BD.sqlite import BD
from Util.log import log_fecha_hora, log_final
from Util.util import verificar_y_crear_carpetas

# Reutilizamos la lógica actual sin duplicar código
from main import procesar_experimento  # noqa: E402


def _default_n_jobs() -> int:
    env = os.environ.get("SLURM_CPUS_PER_TASK")
    if env:
        with_safety = int(env)
        return max(1, with_safety)
    return max(1, os.cpu_count() or 1)


def _worker_loop(worker_id: int) -> int:
    """Toma trabajos desde SQLite hasta que no queden pendientes."""
    processed = 0
    bd = BD()
    while True:
        data = bd.obtenerExperimento()
        if data is None:
            break
        procesar_experimento(data, bd)
        processed += 1
    return processed


def main() -> int:
    p = argparse.ArgumentParser(description="Ejecuta experimentos en paralelo usando joblib (sin xargs).")
    p.add_argument("--n-jobs", type=int, default=_default_n_jobs(), help="Número de workers (default: SLURM_CPUS_PER_TASK)")
    args = p.parse_args()

    n_jobs = max(1, int(args.n_jobs))

    # Evitar oversubscription de BLAS/OpenMP dentro de cada proceso.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    verificar_y_crear_carpetas()

    start_time = time.time()
    log_fecha_hora("Inicio de la ejecución (joblib)")

    print(f"[JOBLIB] Workers: {n_jobs}")

    # Cada worker compite por tomar el siguiente experimento (bloqueo en SQLite).
    processed_counts = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
        delayed(_worker_loop)(wid) for wid in range(n_jobs)
    )

    total_processed = int(sum(processed_counts))

    end_time = time.time()
    total_time = end_time - start_time

    log_fecha_hora("Fin de la ejecución (joblib)")
    log_final(total_time)

    shutil.rmtree(os.path.join(os.path.dirname(__file__), "Resultados", "transitorio"), ignore_errors=True)

    print(f"[JOBLIB] Experimentos procesados: {total_processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
