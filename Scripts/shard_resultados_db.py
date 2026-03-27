#!/usr/bin/env python3

import argparse
import os
import sqlite3
from typing import Iterable


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def _read_schema(src: sqlite3.Connection) -> list[str]:
    cur = src.cursor()
    cur.execute(
        """
        SELECT sql
        FROM sqlite_master
        WHERE type IN ('table','index','trigger','view')
          AND name NOT LIKE 'sqlite_%'
          AND sql IS NOT NULL
        ORDER BY type='table' DESC, name
        """
    )
    return [row[0] for row in cur.fetchall()]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _iter_experimentos(src: sqlite3.Connection, where: str, params: tuple) -> Iterable[tuple]:
    cur = src.cursor()
    cur.execute(f"SELECT * FROM experimentos {where} ORDER BY id_experimento", params)
    yield from cur.fetchall()


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Parte una BD SQLite (resultados.db) en N shards (para correr en paralelo sin pelear por el mismo archivo).\n"
            "Copia la tabla instancias completa, y reparte la tabla experimentos (sin resultados)."
        )
    )
    p.add_argument("--db", default="./BD/resultados.db", help="BD fuente")
    p.add_argument("--out-dir", default="./BD/shards", help="Carpeta destino para shards")
    p.add_argument("--shards", type=int, default=3, help="Cantidad de shards")
    p.add_argument(
        "--only-pendiente",
        action="store_true",
        help="Solo shardear experimentos con estado='pendiente' (útil si ya hay resultados en la BD fuente)",
    )
    args = p.parse_args()

    root = _project_root()
    src_db = args.db
    if not os.path.isabs(src_db):
        src_db = os.path.join(root, src_db.lstrip("./"))

    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(root, out_dir.lstrip("./"))

    shards = max(1, int(args.shards))

    if not os.path.exists(src_db):
        raise FileNotFoundError(f"No existe BD fuente: {src_db}")

    _ensure_dir(out_dir)

    with _connect(src_db) as src:
        schema_sql = _read_schema(src)

        where = ""
        params: tuple = ()
        if args.only_pendiente:
            where = "WHERE estado = 'pendiente'"

        experimentos = list(_iter_experimentos(src, where, params))

        # Leer instancias completas (se mantienen ids)
        cur = src.cursor()
        cur.execute("SELECT * FROM instancias ORDER BY id_instancia")
        instancias = cur.fetchall()

    # Crear shards
    for shard_idx in range(1, shards + 1):
        shard_path = os.path.join(out_dir, f"resultados_{shard_idx}.db")
        if os.path.exists(shard_path):
            os.remove(shard_path)

        with _connect(shard_path) as dst:
            cur = dst.cursor()
            for stmt in schema_sql:
                cur.execute(stmt)

            # Copiar instancias
            if instancias:
                placeholders = ",".join(["?"] * len(instancias[0]))
                cur.executemany(f"INSERT INTO instancias VALUES ({placeholders})", instancias)

            dst.commit()

    # Repartir experimentos por id_experimento % shards
    for row in experimentos:
        exp_id = int(row[0])
        shard_idx = (exp_id % shards) + 1  # 1..shards
        shard_path = os.path.join(out_dir, f"resultados_{shard_idx}.db")
        with _connect(shard_path) as dst:
            cur = dst.cursor()
            placeholders = ",".join(["?"] * len(row))
            cur.execute(f"INSERT INTO experimentos VALUES ({placeholders})", row)
            dst.commit()

    print(f"OK: shards creados en {out_dir}")
    for shard_idx in range(1, shards + 1):
        print(f"- resultados_{shard_idx}.db")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
