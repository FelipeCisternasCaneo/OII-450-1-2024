import sqlite3
import os
import glob

def _current_batch_id() -> str | None:
    """Obtiene un batch_id desde variables de entorno (Slurm/usuario)."""
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


class SQLiteConnectionManager:
    """
    Gestor de conexión y transacciones para SQLite, incluyendo soporte para
    shards de resultados y connection pooling.
    """

    def __init__(self):
        # Ruta por defecto robusta (no depende del CWD)
        # Permite override vía variable de entorno OII_DB_PATH
        file_dir = os.path.dirname(os.path.abspath(__file__))
        parent = os.path.dirname(file_dir)
        if os.path.basename(parent).lower() == 'src':
            self._project_root = os.path.dirname(parent)
        else:
            self._project_root = parent

        env_db = os.environ.get("OII_DB_PATH")
        if env_db:
            self._dataBase = env_db
        else:
            self._dataBase = os.path.join(self._project_root, "data", "database", "resultados.db")
        self._conexion = None
        self._cursor = None
        self._pooling_active = False  # Flag para connection pooling
        self._pooling_depth = 0  # Contador de contextos anidados
        self._experimentos_cols_aseguradas = False
        self._shards_cache = None
        self._base_iteraciones_vacias = None

    def _listar_shards(self):
        """Lista shards disponibles en data/database/shards/resultados_*.db (ordenados)."""
        if self._shards_cache is not None:
            return self._shards_cache

        shards_dir = self._abs_repo_path("data", "database", "shards")
        pattern = os.path.join(shards_dir, "resultados_*.db")
        paths = sorted([p for p in glob.glob(pattern) if os.path.isfile(p)])
        self._shards_cache = paths
        return paths

    def _iteraciones_vacias_en_base(self):
        """Devuelve True si la BD actual (resultados.db) no tiene iteraciones."""
        if self._base_iteraciones_vacias is not None:
            return self._base_iteraciones_vacias

        try:
            conn = sqlite3.connect(self.getDataBase(), timeout=10)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM iteraciones")
            n = int(cur.fetchone()[0])
            conn.close()
            self._base_iteraciones_vacias = n == 0
        except Exception:
            # Si no se puede chequear, asumir que NO está vacía para no cambiar comportamiento.
            self._base_iteraciones_vacias = False

        return self._base_iteraciones_vacias

    def _deberia_consultar_shards(self):
        """Heurística: si estamos usando resultados.db sin iteraciones, leer desde shards."""
        try:
            is_base = os.path.basename(self.getDataBase()) == "resultados.db"
        except Exception:
            is_base = False
        if not is_base:
            return False
        if not self._iteraciones_vacias_en_base():
            return False
        return len(self._listar_shards()) > 0

    @staticmethod
    def _configurar_sqlite(conn: sqlite3.Connection) -> None:
        """Configura pragmas de SQLite con enfoque en robustez bajo concurrencia.

        Nota: Cambiar `journal_mode` puede requerir locks; por eso se hace best-effort.
        Se puede forzar vía env `OII_SQLITE_JOURNAL_MODE` (por ejemplo: WAL, DELETE).
        """
        # Espera si la BD está ocupada (también se puede setear por connect(timeout=...))
        try:
            conn.execute("PRAGMA busy_timeout = 30000;")
        except Exception:
            pass

        # Pragmas seguros / de rendimiento moderado
        try:
            conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
        try:
            conn.execute("PRAGMA cache_size = -8000;")
        except Exception:
            pass

        # journal_mode: por defecto DELETE (más compatible en FS de red).
        journal_mode = (
            (os.environ.get("OII_SQLITE_JOURNAL_MODE") or "DELETE").strip().upper()
        )
        if journal_mode:
            try:
                conn.execute(f"PRAGMA journal_mode={journal_mode};")
            except sqlite3.OperationalError:
                # Si hay lock/busy, no reventar: se puede seguir con el modo actual.
                pass

    def _abs_repo_path(self, *parts: str) -> str:
        return os.path.join(self._project_root, *parts)

    def getDataBase(self):
        return self._dataBase

    def setDataBase(self, dataBase):
        self._dataBase = dataBase

    def getConexion(self):
        return self._conexion

    def setConexion(self, conexion):
        self._conexion = conexion

    def getCursor(self):
        return self._cursor

    def setCursor(self, cursor):
        self._cursor = cursor

    def conectar(self):
        # Si pooling está activo y ya hay conexión, reutilizarla
        if self._pooling_active and self._conexion is not None:
            return

        # Asegurar que exista el directorio de la BD
        db_path = self.getDataBase()
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        conn = sqlite3.connect(db_path, timeout=30)
        self._configurar_sqlite(conn)

        cursor = conn.cursor()

        self.setConexion(conn)
        self.setCursor(cursor)

        # Asegurar columnas nuevas en BD existentes (retrocompatible)
        try:
            self._asegurar_columnas_experimentos(cursor)
            conn.commit()
        except Exception:
            # No bloquear el flujo si la tabla aún no existe u ocurre un error menor.
            pass

    @staticmethod
    def _tabla_info_columnas(cursor, table_name: str):
        cursor.execute(f"PRAGMA table_info({table_name})")
        return {row[1] for row in cursor.fetchall()}  # row[1] = nombre columna

    @classmethod
    def _asegurar_columnas_experimentos(cls, cursor):
        """Agrega columnas de monitoreo si faltan (idempotente)."""
        cols = cls._tabla_info_columnas(cursor, "experimentos")

        if "batch_id" not in cols:
            cursor.execute("ALTER TABLE experimentos ADD COLUMN batch_id TEXT")
            cols.add("batch_id")

        if "ts_inicio" not in cols:
            cursor.execute("ALTER TABLE experimentos ADD COLUMN ts_inicio TEXT")
            cols.add("ts_inicio")

        if "ts_fin" not in cols:
            cursor.execute("ALTER TABLE experimentos ADD COLUMN ts_fin TEXT")
            cols.add("ts_fin")

    def desconectar(self):
        # Si pooling está activo, no cerrar la conexión aún
        if self._pooling_active:
            return

        if self._conexion is not None:
            self._conexion.close()
            self._conexion = None
            self._cursor = None

    def __enter__(self):
        """Context manager para connection pooling: with bd:"""
        self._pooling_depth += 1
        if self._pooling_depth == 1:
            self._pooling_active = True
            self.conectar()  # Abrir conexión al entrar al contexto
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cerrar conexión al salir del contexto"""
        self._pooling_depth -= 1
        if self._pooling_depth == 0:
            self._pooling_active = False
            if self._conexion is not None:
                self._conexion.close()
                self._conexion = None
                self._cursor = None
        return False  # No suprimir excepciones

    def commit(self):
        self.getConexion().commit()
