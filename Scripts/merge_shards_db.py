#!/usr/bin/env python3

import argparse
import glob
import os
import shutil
import sqlite3
import time


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=60)
    conn.execute("PRAGMA busy_timeout = 60000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _backup_file(path: str, backup_dir: str) -> str:
    _ensure_dir(backup_dir)
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = os.path.basename(path)
    dst = os.path.join(backup_dir, f"{base}.bak.{ts}")
    shutil.copy2(path, dst)
    return dst


def _table_exists(cur: sqlite3.Cursor, table: str) -> bool:
    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table,)
    )
    return cur.fetchone() is not None


def _count(cur: sqlite3.Cursor, sql: str, params: tuple = ()) -> int:
    cur.execute(sql, params)
    row = cur.fetchone()
    return int(row[0]) if row else 0


def _merge_attached_shard(base_conn: sqlite3.Connection, shard_path: str, dry_run: bool) -> dict:
    """Mergea datos desde una BD ya ATTACH-eada como `shard`.

    Importante: en SQLite, DETACH no se puede ejecutar dentro de una transacción.
    Por eso el flujo recomendado es:
      ATTACH -> BEGIN -> merge -> COMMIT -> DETACH
    (o sin BEGIN/COMMIT en dry-run).
    """
    stats = {
        "shard": shard_path,
        "iteraciones_insertadas": 0,
        "resultados_insertados": 0,
        "experimentos_actualizados": 0,
    }

    cur = base_conn.cursor()

    # 1) Actualizar estado/ts/batch_id en experimentos (no pisar con NULL)
    update_sql = """
        UPDATE experimentos
           SET estado   = COALESCE((SELECT s.estado   FROM shard.experimentos s WHERE s.id_experimento = experimentos.id_experimento), estado),
               batch_id = COALESCE((SELECT s.batch_id FROM shard.experimentos s WHERE s.id_experimento = experimentos.id_experimento), batch_id),
               ts_inicio= COALESCE((SELECT s.ts_inicio FROM shard.experimentos s WHERE s.id_experimento = experimentos.id_experimento), ts_inicio),
               ts_fin   = COALESCE((SELECT s.ts_fin    FROM shard.experimentos s WHERE s.id_experimento = experimentos.id_experimento), ts_fin)
         WHERE id_experimento IN (SELECT id_experimento FROM shard.experimentos)
    """
    if not dry_run:
        cur.execute(update_sql)
    stats["experimentos_actualizados"] = cur.rowcount if cur.rowcount != -1 else 0

    # 2) Insertar resultados evitando duplicados por fk_id_experimento
    resultados_to_add = _count(
        cur,
        """
        SELECT COUNT(*)
          FROM shard.resultados r
         WHERE NOT EXISTS (
               SELECT 1 FROM resultados br WHERE br.fk_id_experimento = r.fk_id_experimento
         )
        """,
    )
    if not dry_run and resultados_to_add:
        cur.execute(
            """
            INSERT INTO resultados (fitness, tiempoEjecucion, solucion, fk_id_experimento)
            SELECT r.fitness, r.tiempoEjecucion, r.solucion, r.fk_id_experimento
              FROM shard.resultados r
             WHERE NOT EXISTS (
                   SELECT 1 FROM resultados br WHERE br.fk_id_experimento = r.fk_id_experimento
             )
            """
        )
    stats["resultados_insertados"] = resultados_to_add

    # 3) Insertar iteraciones evitando duplicados por (fk_id_experimento, nombre)
    iteraciones_to_add = _count(
        cur,
        """
        SELECT COUNT(*)
          FROM shard.iteraciones i
         WHERE NOT EXISTS (
               SELECT 1
                 FROM iteraciones bi
                WHERE bi.fk_id_experimento = i.fk_id_experimento
                  AND bi.nombre = i.nombre
         )
        """,
    )
    if not dry_run and iteraciones_to_add:
        cur.execute(
            """
            INSERT INTO iteraciones (nombre, archivo, fk_id_experimento)
            SELECT i.nombre, i.archivo, i.fk_id_experimento
              FROM shard.iteraciones i
             WHERE NOT EXISTS (
                   SELECT 1
                     FROM iteraciones bi
                    WHERE bi.fk_id_experimento = i.fk_id_experimento
                      AND bi.nombre = i.nombre
             )
            """
        )
    stats["iteraciones_insertadas"] = iteraciones_to_add

    return stats


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Mergea shards SQLite (BD/shards/resultados_*.db) de vuelta en BD/resultados.db.\n"
            "Copia resultados + iteraciones y sincroniza estado/ts/batch_id de experimentos.\n"
            "Hace backup antes de modificar."
        )
    )
    p.add_argument("--base-db", default="./BD/resultados.db", help="Ruta a resultados.db")
    p.add_argument("--shards-dir", default="./BD/shards", help="Carpeta con resultados_*.db")
    p.add_argument("--pattern", default="resultados_*.db", help="Patrón de shards")
    p.add_argument("--backup-dir", default="./BD/backups", help="Dónde guardar backup de la BD base")
    p.add_argument("--dry-run", action="store_true", help="No escribe, solo reporta conteos")
    args = p.parse_args()

    root = _project_root()

    base_db = args.base_db
    if not os.path.isabs(base_db):
        base_db = os.path.join(root, base_db.lstrip("./"))

    shards_dir = args.shards_dir
    if not os.path.isabs(shards_dir):
        shards_dir = os.path.join(root, shards_dir.lstrip("./"))

    backup_dir = args.backup_dir
    if not os.path.isabs(backup_dir):
        backup_dir = os.path.join(root, backup_dir.lstrip("./"))

    if not os.path.exists(base_db):
        raise FileNotFoundError(f"No existe base db: {base_db}")

    shard_paths = sorted(glob.glob(os.path.join(shards_dir, args.pattern)))
    shard_paths = [p for p in shard_paths if os.path.isfile(p)]
    if not shard_paths:
        raise FileNotFoundError(f"No se encontraron shards en {shards_dir} con patrón {args.pattern}")

    # Backup
    if not args.dry_run:
        backup_path = _backup_file(base_db, backup_dir)
        print(f"[OK] Backup creado: {backup_path}")

    with _connect(base_db) as base_conn:
        cur = base_conn.cursor()
        for t in ("experimentos", "resultados", "iteraciones"):
            if not _table_exists(cur, t):
                raise RuntimeError(f"Tabla requerida no existe en base db: {t}")

        all_stats = []
        for shard_path in shard_paths:
            try:
                # Importante: en SQLite no se puede DETACH dentro de una transacción.
                # Por eso, en modo escritura hacemos BEGIN/COMMIT por shard.
                cur = base_conn.cursor()
                cur.execute("ATTACH DATABASE ? AS shard", (shard_path,))

                if not args.dry_run:
                    base_conn.execute("BEGIN")

                st = _merge_attached_shard(base_conn, shard_path, dry_run=args.dry_run)
                all_stats.append(st)

                if not args.dry_run:
                    base_conn.commit()

                # DETACH solo después del COMMIT (o sin transacción en dry-run)
                cur.execute("DETACH DATABASE shard")

                print(
                    f"[MERGE] {os.path.basename(shard_path)} | "
                    f"iteraciones +{st['iteraciones_insertadas']} | "
                    f"resultados +{st['resultados_insertados']} | "
                    f"experimentos upd ~{st['experimentos_actualizados']}"
                )
            except Exception:
                try:
                    if not args.dry_run:
                        base_conn.rollback()
                    try:
                        cur = base_conn.cursor()
                        cur.execute("DETACH DATABASE shard")
                    except Exception:
                        pass
                finally:
                    raise

        if args.dry_run:
            print("[DRY-RUN] No se escribió nada.")
        else:
            print("[OK] Merge completado.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
