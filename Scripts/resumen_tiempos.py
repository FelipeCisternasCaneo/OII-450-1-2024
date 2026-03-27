#!/usr/bin/env python3

import argparse
import os
import sqlite3
from datetime import datetime, timezone

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # Python < 3.9
    ZoneInfo = None  # type: ignore[assignment]
    ZoneInfoNotFoundError = Exception  # type: ignore[assignment]


def _resolve_report_tz():
    """Resuelve la TZ del reporte.

    Nota Windows: normalmente necesitas `pip install tzdata` para usar IANA TZ.
    Si no está disponible, caemos a UTC para no romper el resumen.
    """
    tz_name = os.environ.get("OII_TZ", "America/Santiago")
    if ZoneInfo is None:
        return timezone.utc, "UTC (Python<3.9: sin zoneinfo)"
    try:
        return ZoneInfo(tz_name), tz_name
    except ZoneInfoNotFoundError:
        return timezone.utc, f"UTC (sin tzdata: {tz_name})"


_REPORT_TZ, _REPORT_TZ_LABEL = _resolve_report_tz()


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def _project_root() -> str:
    """Raíz del proyecto asumiendo que este script vive en ./Scripts/."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_db_path(db_arg: str) -> str:
    """Resuelve la ruta de la BD aunque el script se ejecute desde otro CWD."""
    if not db_arg:
        raise ValueError("--db está vacío")

    # Absoluta: usar tal cual
    if os.path.isabs(db_arg):
        return db_arg

    # 1) Relativa al CWD
    if os.path.exists(db_arg):
        return db_arg

    # 2) Relativa al root del proyecto
    candidate = os.path.join(_project_root(), db_arg.lstrip("./"))
    return candidate


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _parse_sqlite_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    # SQLite CURRENT_TIMESTAMP => 'YYYY-MM-DD HH:MM:SS'
    # A veces puede venir con fracciones: 'YYYY-MM-DD HH:MM:SS.SSS'
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def _env_batch_id() -> str | None:
    """Obtiene batch_id en el mismo formato que Slurm muestra en squeue (p.ej. 10805_1)."""
    batch = os.environ.get("OII_BATCH_ID")
    if batch:
        return batch

    slurm_job = os.environ.get("SLURM_JOB_ID")
    slurm_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_job and slurm_task:
        return f"{slurm_job}_{slurm_task}"
    if slurm_job:
        return slurm_job
    return None


def _as_chile_from_sqlite_utc(dt_naive: datetime | None) -> datetime | None:
    """Interpreta datetime naive (SQLite CURRENT_TIMESTAMP) como UTC y convierte a Chile."""
    if dt_naive is None:
        return None
    return dt_naive.replace(tzinfo=timezone.utc).astimezone(_REPORT_TZ)


def _fmt_dt(dt: datetime | None) -> str:
    if dt is None:
        return "N/A"
    # Ej: 2026-01-12 22:17:16 -0300
    return dt.strftime("%Y-%m-%d %H:%M:%S %z")


def main() -> int:
    p = argparse.ArgumentParser(description="Genera resumen de tiempos del lote desde la BD (sin monitor/polling).")
    p.add_argument(
        "--db",
        default="./BD/resultados.db",
        help="Ruta a resultados.db (si es relativa, se intenta resolver desde el CWD y desde la raíz del proyecto)",
    )
    p.add_argument("--batch-id", default=None, help="Filtrar por batch_id")
    p.add_argument("--out", default=None, help="Archivo de salida .log (si se omite, imprime por stdout)")
    args = p.parse_args()

    # Si no pasan --batch-id, intentar detectarlo desde el entorno (Slurm).
    if not args.batch_id:
        args.batch_id = _env_batch_id()

    where = ""
    params: tuple = ()
    if args.batch_id:
        where = "WHERE e.batch_id = ?"
        params = (args.batch_id,)

    db_path = _resolve_db_path(args.db)
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"No se encontró la base de datos en '{args.db}'.\n"
            f"Probado: '{db_path}'.\n"
            "Sugerencia: ejecuta desde la raíz del repo o pasa --db con ruta absoluta."
        )

    with _connect(db_path) as conn:
        cur = conn.cursor()

        # Conteos por estado
        cur.execute(
            f"""
            SELECT e.estado, COUNT(*)
            FROM experimentos e
            {where}
            GROUP BY e.estado
            """,
            params,
        )
        counts = {estado: int(cnt) for estado, cnt in cur.fetchall()}

        # Ventana de tiempo (makespan) según timestamps de experimento.
        # - min_start: primer inicio observado
        # - max_end: último fin observado (solo terminados/error con ts_fin)
        # - max_end_or_now: último fin o "ahora" si hay experimentos sin ts_fin
        cur.execute(
            f"""
            SELECT
                MIN(e.ts_inicio),
                MAX(e.ts_fin),
                MAX(COALESCE(e.ts_fin, CURRENT_TIMESTAMP))
            FROM experimentos e
            {where}
            """,
            params,
        )
        min_start, max_end, max_end_or_now = cur.fetchone() or (None, None, None)
        dt_start = _as_chile_from_sqlite_utc(_parse_sqlite_ts(min_start))
        dt_end = _as_chile_from_sqlite_utc(_parse_sqlite_ts(max_end))
        dt_end_or_now = _as_chile_from_sqlite_utc(_parse_sqlite_ts(max_end_or_now))

        makespan_finished_s = (dt_end - dt_start).total_seconds() if (dt_start and dt_end) else None
        makespan_so_far_s = (dt_end_or_now - dt_start).total_seconds() if (dt_start and dt_end_or_now) else None

        # Diagnóstico de timestamps faltantes
        cur.execute(
            f"""
            SELECT
                SUM(CASE WHEN e.ts_inicio IS NULL THEN 1 ELSE 0 END) AS sin_inicio,
                SUM(CASE WHEN e.ts_inicio IS NOT NULL AND e.ts_fin IS NULL THEN 1 ELSE 0 END) AS sin_fin
            FROM experimentos e
            {where}
            """,
            params,
        )
        sin_inicio, sin_fin = cur.fetchone() or (0, 0)

        # Suma de tiempoEjecucion (CPU time agregado) y max (tiempo del experimento más largo)
        cur.execute(
            f"""
            SELECT
                COUNT(r.id_resultado),
                COALESCE(SUM(r.tiempoEjecucion), 0),
                COALESCE(MAX(r.tiempoEjecucion), 0),
                COALESCE(MIN(r.tiempoEjecucion), 0)
            FROM resultados r
            JOIN experimentos e ON e.id_experimento = r.fk_id_experimento
            {where}
            """,
            params,
        )
        n_res, sum_exec, max_exec, min_exec = cur.fetchone() or (0, 0.0, 0.0, 0.0)

    lines: list[str] = []
    now = datetime.now(tz=_REPORT_TZ).strftime("%Y-%m-%d %H:%M:%S %z")

    lines.append("=" * 70)
    lines.append("RESUMEN DE EJECUCIÓN (desde BD)")
    lines.append("=" * 70)
    lines.append(f"Generado: {now}")
    lines.append(f"Zona horaria reporte: {_REPORT_TZ_LABEL}")
    lines.append(f"BD: {db_path}")
    if args.batch_id:
        lines.append(f"Lote (batch_id): {args.batch_id}")
    lines.append("")

    # Estados
    pendiente = counts.get("pendiente", 0)
    ejecutando = counts.get("ejecutando", 0)
    terminado = counts.get("terminado", 0)
    error = counts.get("error", 0)
    total = sum(counts.values())

    lines.append("Estado de experimentos")
    lines.append(f"- Total: {total}")
    lines.append(f"- Terminado: {terminado}")
    lines.append(f"- Error: {error}")
    lines.append(f"- Pendiente: {pendiente}")
    lines.append(f"- Ejecutando: {ejecutando}")
    lines.append("")

    # Tiempos
    lines.append("Tiempos")
    if dt_start and makespan_finished_s is not None and pendiente == 0 and ejecutando == 0 and sin_fin == 0:
        # Caso "lote finalizado"
        lines.append(f"- Tiempo real (makespan): {_fmt_seconds(makespan_finished_s)}")
        lines.append(f"- Ventana (inicio..fin): {_fmt_dt(dt_start)} .. {_fmt_dt(dt_end)}")
    elif dt_start and makespan_so_far_s is not None:
        # Caso "hay actividad o faltan ts_fin": reportar tiempo real transcurrido desde el primer inicio
        lines.append(f"- Tiempo real (hasta ahora): {_fmt_seconds(makespan_so_far_s)}")
        if dt_end:
            lines.append(f"- Último fin: {_fmt_dt(dt_end)}")
        lines.append(f"- Primer inicio: {_fmt_dt(dt_start)}")
    else:
        lines.append("- Tiempo real: no disponible")

    if total > 0 and (sin_inicio or sin_fin):
        lines.append(f"- Timestamps faltantes: sin ts_inicio={int(sin_inicio)} sin ts_fin={int(sin_fin)}")

    lines.append(f"- Suma tiempos individuales (CPU): {_fmt_seconds(float(sum_exec))}")
    lines.append(f"- Experimento más corto: {_fmt_seconds(float(min_exec))}")
    lines.append(f"- Experimento más largo: {_fmt_seconds(float(max_exec))}")
    lines.append(f"- Resultados registrados en tabla 'resultados': {int(n_res)}")

    if total > 0 and int(n_res) == 0:
        lines.append("")
        lines.append("Nota: hay experimentos pero no hay resultados; puede que hayan fallado antes de insertarResultados().")

    lines.append("=" * 70)

    out_text = "\n".join(lines) + "\n"

    # Guardado: siempre asegurar Logs/ en la raíz del proyecto (no depende del CWD).
    logs_dir = os.path.join(_project_root(), "Logs")
    os.makedirs(logs_dir, exist_ok=True)

    out_path = args.out
    if not out_path:
        # Default: Logs/resumen_<batch>.log o Logs/resumen_global.log
        if args.batch_id:
            out_path = os.path.join(logs_dir, f"resumen_{args.batch_id}.log")
        else:
            out_path = os.path.join(logs_dir, "resumen_global.log")
    else:
        # Si el usuario pasa un nombre sin carpeta (ej. --out resumen.log),
        # guardarlo dentro de Logs/ para mantener todo ordenado.
        if not os.path.isabs(out_path) and os.path.dirname(out_path) == "":
            out_path = os.path.join(logs_dir, out_path)
        elif not os.path.isabs(out_path):
            # Rutas relativas con carpeta: resolverlas contra la raíz del proyecto
            out_path = os.path.join(_project_root(), out_path)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as fh:
        fh.write(out_text)

    # Mantener comportamiento original: si no pidieron --out, también imprimir.
    if not args.out:
        print(out_text, end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
