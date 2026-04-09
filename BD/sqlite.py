import sqlite3
import os
import time
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


from Problem.SCP.problem import SetCoveringProblem


class BD:
    def __init__(self):
        # Ruta por defecto robusta (no depende del CWD)
        # Permite override vía variable de entorno OII_DB_PATH
        self.__project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        env_db = os.environ.get("OII_DB_PATH")
        if env_db:
            self.__dataBase = env_db
        else:
            self.__dataBase = os.path.join(self.__project_root, "BD", "resultados.db")
        self.__conexion = None
        self.__cursor = None
        self.__pooling_active = False  # Flag para connection pooling
        self.__pooling_depth = 0  # Contador de contextos anidados
        self.__experimentos_cols_aseguradas = False
        self.__shards_cache = None
        self.__base_iteraciones_vacias = None

    def _listar_shards(self):
        """Lista shards disponibles en BD/shards/resultados_*.db (ordenados)."""
        if self.__shards_cache is not None:
            return self.__shards_cache

        shards_dir = self._abs_repo_path("BD", "shards")
        pattern = os.path.join(shards_dir, "resultados_*.db")
        paths = sorted([p for p in glob.glob(pattern) if os.path.isfile(p)])
        self.__shards_cache = paths
        return paths

    def _iteraciones_vacias_en_base(self):
        """Devuelve True si la BD actual (resultados.db) no tiene iteraciones."""
        if self.__base_iteraciones_vacias is not None:
            return self.__base_iteraciones_vacias

        try:
            conn = sqlite3.connect(self.getDataBase(), timeout=10)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM iteraciones")
            n = int(cur.fetchone()[0])
            conn.close()
            self.__base_iteraciones_vacias = n == 0
        except Exception:
            # Si no se puede chequear, asumir que NO está vacía para no cambiar comportamiento.
            self.__base_iteraciones_vacias = False

        return self.__base_iteraciones_vacias

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
        return os.path.join(self.__project_root, *parts)

    def getDataBase(self):
        return self.__dataBase

    def setDataBase(self, dataBase):
        self.__dataBase = dataBase

    def getConexion(self):
        return self.__conexion

    def setConexion(self, conexion):
        self.__conexion = conexion

    def getCursor(self):
        return self.__cursor

    def setCursor(self, cursor):
        self.__cursor = cursor

    def conectar(self):
        # Si pooling está activo y ya hay conexión, reutilizarla
        if self.__pooling_active and self.__conexion is not None:
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
        if self.__pooling_active:
            return

        if self.__conexion is not None:
            self.__conexion.close()
            self.__conexion = None
            self.__cursor = None

    def __enter__(self):
        """Context manager para connection pooling: with bd:"""
        self.__pooling_depth += 1
        if self.__pooling_depth == 1:
            self.__pooling_active = True
            self.conectar()  # Abrir conexión al entrar al contexto
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cerrar conexión al salir del contexto"""
        self.__pooling_depth -= 1
        if self.__pooling_depth == 0:
            self.__pooling_active = False
            if self.__conexion is not None:
                self.__conexion.close()
                self.__conexion = None
                self.__cursor = None
        return False  # No suprimir excepciones

    def commit(self):
        self.getConexion().commit()

    def construirTablas(self):
        self.conectar()

        self.getCursor().execute(
            """ CREATE TABLE IF NOT EXISTS instancias(
                id_instancia INTEGER PRIMARY KEY AUTOINCREMENT,
                tipo_problema TEXT,
                nombre TEXT,
                optimo REAL,
                param TEXT
            )"""
        )

        self.getCursor().execute(
            """ CREATE TABLE IF NOT EXISTS experimentos(
                id_experimento INTEGER PRIMARY KEY AUTOINCREMENT,
                experimento TEXT,
                MH TEXT,
                binarizacion TEXT,
                paramMH TEXT,
                ML TEXT,
                paramML TEXT,
                ML_FS TEXT,
                paramML_FS TEXT,
                estado TEXT,
                fk_id_instancia INTEGER,
                batch_id TEXT,
                ts_inicio TEXT,
                ts_fin TEXT,
                FOREIGN KEY (fk_id_instancia) REFERENCES instancias (id_instancia)
            )"""
        )

        self.getCursor().execute(
            """ CREATE TABLE IF NOT EXISTS resultados(
                id_resultado INTEGER PRIMARY KEY AUTOINCREMENT,
                fitness REAL,
                tiempoEjecucion REAL,
                solucion TEXT,
                fk_id_experimento INTEGER,
                FOREIGN KEY (fk_id_experimento) REFERENCES experimentos (id_experimento)
            )"""
        )

        self.getCursor().execute(
            """ CREATE TABLE IF NOT EXISTS iteraciones(
                id_archivo INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT,
                archivo BLOB,
                fk_id_experimento INTEGER,
                FOREIGN KEY (fk_id_experimento) REFERENCES experimentos (id_experimento)
            )"""
        )

        # Crear índices para optimizar consultas frecuentes
        self.getCursor().execute(
            """CREATE INDEX IF NOT EXISTS idx_estado ON experimentos(estado)"""
        )
        self.getCursor().execute(
            """CREATE INDEX IF NOT EXISTS idx_instancia_nombre ON instancias(nombre)"""
        )
        self.getCursor().execute(
            """CREATE INDEX IF NOT EXISTS idx_iteraciones_fk ON iteraciones(fk_id_experimento)"""
        )
        self.getCursor().execute(
            """CREATE INDEX IF NOT EXISTS idx_resultados_fk ON resultados(fk_id_experimento)"""
        )

        self.commit()

        # Asegurar columnas nuevas si la tabla existía previamente sin ellas
        self._asegurar_columnas_experimentos(self.getCursor())
        self.commit()

        # Insertar instancias de todos los dominios registrados
        from Solver.domain_managers import ensure_registered
        from Solver.domain_managers.registry import get_all as get_all_domains

        ensure_registered()

        for _dtype, entry in get_all_domains().items():
            if entry.insert_instances is not None:
                entry.insert_instances(self)

        self.desconectar()

    def insertarExperimentos(self, data, corridas, id):
        self.conectar()

        # Asegurar columnas nuevas en BD existentes
        self._asegurar_columnas_experimentos(self.getCursor())

        # Bulk insert usando executemany
        batch_id = data.get("batch_id") if isinstance(data, dict) else None

        valores = [
            (
                str(data["experimento"]),
                str(data["MH"]),
                str(data["binarizacion"]),
                str(data["paramMH"]),
                str(data["ML"]),
                str(data["paramML"]),
                str(data["ML_FS"]),
                str(data["paramML_FS"]),
                str(data["estado"]),
                id,
                batch_id,
                None,
                None,
            )
            for _ in range(corridas)
        ]

        self.getCursor().executemany(
            """INSERT INTO experimentos (
                   experimento, MH, binarizacion, paramMH, ML, paramML, ML_FS, paramML_FS,
                   estado, fk_id_instancia, batch_id, ts_inicio, ts_fin
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            valores,
        )

        self.commit()
        self.desconectar()

    data = [
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "F7",
        "F8",
        "F9",
        "F10",
        "F11",
        "F12",
        "F13",
        "F14",
        "F15",
        "F16",
        "F17",
        "F18",
        "F19",
        "F20",
        "F21",
        "F22",
        "F23",
        "F1CEC2017",
        "F2CEC2017",
        "F3CEC2017",
        "F4CEC2017",
        "F5CEC2017",
        "F6CEC2017",
        "F7CEC2017",
        "F8CEC2017",
        "F9CEC2017",
        "F10CEC2017",
        "F11CEC2017",
        "F12CEC2017",
        "F13CEC2017",
        "F14CEC2017",
        "F15CEC2017",
        "F16CEC2017",
        "F17CEC2017",
        "F18CEC2017",
        "F19CEC2017",
        "F20CEC2017",
        "F21CEC2017",
        "F22CEC2017",
        "F23CEC2017",
        "F24CEC2017",
        "F25CEC2017",
        "F26CEC2017",
        "F27CEC2017",
        "F28CEC2017",
        "F29CEC2017",
        "F30CEC2017",
    ]

    def insertarInstanciasBEN(self):
        self.conectar()

        tipoProblema = "BEN"

        # Filtrar solo funciones clásicas (F1-F23), excluir CEC2017
        funciones_clasicas = [f for f in self.data if not f.endswith("CEC2017")]

        config_ben = {
            "F1": ("-100", "100", 0),
            "F2": ("-10", "10", 0),
            "F3": ("-100", "100", 0),
            "F4": ("-100", "100", 0),
            "F5": ("-30", "30", 0),
            "F6": ("-100", "100", 0),
            "F7": ("-1.28", "1.28", 0),
            "F8": ("-500", "500", -418.9829),
            "F9": ("-5.12", "5.12", 0),
            "F10": ("-32", "32", 0),
            "F11": ("-600", "600", 0),
            "F12": ("-50", "50", 0),
            "F13": ("-50", "50", 0),
            "F14": ("-65.536", "65.536", 1),
            "F15": ("-5", "5", 0.00030),
            "F16": ("-5", "5", -1.0316),
            "F17": ("-5", "5", 0.398),
            "F18": ("-2", "2", 3),
            "F19": ("0", "1", -3.86),
            "F20": ("0", "1", -3.32),
            "F21": ("0", "10", -10.1532),
            "F22": ("0", "10", -10.4028),
            "F23": ("0", "10", -10.5363),
        }

        for instancia in funciones_clasicas:
            if instancia not in config_ben:
                raise ValueError(
                    f"Advertencia: La función '{instancia}' no está definida en la configuración de BEN."
                )

            lb, ub, optimo = config_ben[instancia]
            param = f"lb:{lb},ub:{ub}"

            self.getCursor().execute(
                """INSERT INTO instancias (tipo_problema, nombre, optimo, param) VALUES(?, ?, ?, ?)""",
                (tipoProblema, instancia, optimo, param),
            )

        self.commit()
        self.desconectar()

    def insertarInstanciasCEC2017(self):
        """
        Inserta las instancias de funciones CEC2017 en la base de datos.
        Similar al manejo de funciones clásicas pero con tipo 'CEC2017'.
        """
        from Problem.Benchmark.CEC.cec2017.functions import all_functions

        self.conectar()

        # Óptimos globales de CEC2017: f* = i * 100
        optimos_cec2017 = {
            "f1": 100,
            "f2": 200,
            "f3": 300,
            "f4": 400,
            "f5": 500,
            "f6": 600,
            "f7": 700,
            "f8": 800,
            "f9": 900,
            "f10": 1000,
            "f11": 1100,
            "f12": 1200,
            "f13": 1300,
            "f14": 1400,
            "f15": 1500,
            "f16": 1600,
            "f17": 1700,
            "f18": 1800,
            "f19": 1900,
            "f20": 2000,
            "f21": 2100,
            "f22": 2200,
            "f23": 2300,
            "f24": 2400,
            "f25": 2500,
            "f26": 2600,
            "f27": 2700,
            "f28": 2800,
            "f29": 2900,
            "f30": 3000,
        }

        tipoProblema = "BEN"

        # Parámetros de CEC2017 (rango [-100, 100] para todas)
        lb = -100
        ub = 100
        param = f"lb:{lb},ub:{ub}"

        # Insertar cada función
        for func in all_functions:
            nombre_base = func.__name__  # f1, f2, ..., f30
            nombre_funcion = (
                f"{nombre_base.upper()}CEC2017"  # F1CEC2017, F2CEC2017, ...
            )
            optimo = optimos_cec2017.get(nombre_base, 0)

            # Insertar en base de datos (tipo 'CEC2017')
            self.getCursor().execute(
                """INSERT INTO instancias (tipo_problema, nombre, optimo, param) VALUES(?, ?, ?, ?)""",
                (tipoProblema, nombre_funcion, optimo, param),
            )

        self.commit()
        self.desconectar()

    def insertarInstanciasSCP(self):
        self.conectar()

        instances_dir = self._abs_repo_path("Problem", "SCP", "Instances")
        data = os.listdir(instances_dir)

        for d in data:
            tipoProblema = "SCP"
            nombre = d.split(".")[0]
            optimo = SetCoveringProblem.get_known_optimum(nombre, unicost=False)
            nombre = f"{nombre[3:]}"
            param = ""

            self.getCursor().execute(
                f"""  INSERT INTO instancias (tipo_problema, nombre, optimo, param) VALUES(?, ?, ?, ?) """,
                (tipoProblema, nombre, optimo, param),
            )

        self.commit()
        self.desconectar()

    def insertarInstanciasUSCP(self):
        self.conectar()

        instances_dir = self._abs_repo_path("Problem", "USCP", "Instances")
        data = os.listdir(instances_dir)
        for d in data:
            tipoProblema = "USCP"
            nombre = d.split(".")[0]
            optimo = SetCoveringProblem.get_known_optimum(nombre, unicost=True)

            nombre = f"u{nombre[4:]}"

            param = ""

            self.getCursor().execute(
                f"""  INSERT INTO instancias (tipo_problema, nombre, optimo, param) VALUES(?, ?, ?, ?) """,
                (tipoProblema, nombre, optimo, param),
            )

        self.commit()
        self.desconectar()

    def obtenerExperimento(self):
        """Obtiene y marca 1 experimento como 'ejecutando' de forma tolerante a locks."""
        db_path = self.getDataBase()
        max_reintentos = int(os.environ.get("OII_SQLITE_MAX_RETRIES", "30"))
        sleep_base = float(os.environ.get("OII_SQLITE_RETRY_SLEEP", "0.05"))

        conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)
        try:
            self._configurar_sqlite(conn)

            while True:
                for intento in range(max_reintentos):
                    try:
                        # BEGIN IMMEDIATE: reserva lock de escritura sin ser tan agresivo como EXCLUSIVE.
                        conn.execute("BEGIN IMMEDIATE;")
                        cursor = conn.cursor()

                        if not self.__experimentos_cols_aseguradas:
                            try:
                                self._asegurar_columnas_experimentos(cursor)
                                self.__experimentos_cols_aseguradas = True
                            except sqlite3.OperationalError as e:
                                # Si está locked/busy, seguimos con reintentos.
                                msg = str(e).lower()
                                conn.execute("ROLLBACK;")
                                if "locked" in msg or "busy" in msg:
                                    time.sleep(min(1.0, sleep_base * (1 + intento)))
                                    continue
                                raise
                            except Exception:
                                # No bloquear el flujo si la tabla aún no existe u ocurre un error menor.
                                self.__experimentos_cols_aseguradas = True

                        row = cursor.execute(
                            "SELECT * FROM experimentos WHERE estado = 'pendiente' LIMIT 1"
                        ).fetchone()

                        if row is None:
                            conn.execute("COMMIT;")
                            return None

                        experimento_id = row[0]
                        batch_id = _current_batch_id()

                        cursor.execute(
                            """UPDATE experimentos
                               SET estado = 'ejecutando',
                                   ts_inicio = COALESCE(ts_inicio, CURRENT_TIMESTAMP),
                                   batch_id = COALESCE(batch_id, ?)
                               WHERE id_experimento = ? AND estado = 'pendiente'""",
                            (batch_id, experimento_id),
                        )

                        if cursor.rowcount != 1:
                            # Carrera: otro proceso tomó el experimento entre SELECT y UPDATE.
                            conn.execute("ROLLBACK;")
                            time.sleep(min(1.0, sleep_base * (1 + intento)))
                            continue

                        conn.execute("COMMIT;")
                        return [row]

                    except sqlite3.OperationalError as e:
                        msg = str(e).lower()
                        try:
                            conn.execute("ROLLBACK;")
                        except Exception:
                            pass
                        if "locked" in msg or "busy" in msg:
                            time.sleep(min(1.0, sleep_base * (1 + intento)))
                            continue
                        print(f"Error en BD al obtener experimento: {e}")
                        return None
                    except Exception as e:
                        try:
                            conn.execute("ROLLBACK;")
                        except Exception:
                            pass
                        print(f"Error en BD al obtener experimento: {e}")
                        return None

                # Si llegamos aquí, fue demasiada contención. Confirmar si aún hay pendientes.
                try:
                    pendiente = conn.execute(
                        "SELECT 1 FROM experimentos WHERE estado = 'pendiente' LIMIT 1"
                    ).fetchone()
                except sqlite3.OperationalError:
                    pendiente = (1,)

                if not pendiente:
                    return None

                time.sleep(0.5)
        finally:
            conn.close()

    def obtenerExperimentos(self):
        self.conectar()

        cursor = self.getCursor()

        cursor.execute(""" SELECT * FROM experimentos WHERE estado = 'pendiente' """)
        data = cursor.fetchall()

        self.desconectar()

        return data

    def obtenerInstancia(self, id):
        self.conectar()

        cursor = self.getCursor()

        cursor.execute("""SELECT * FROM instancias WHERE id_instancia = ?""", (id,))
        data = cursor.fetchall()

        self.desconectar()

        return data

    def actualizarExperimento(self, id, estado):
        self.conectar()

        # Asegurar columnas nuevas en BD existentes
        self._asegurar_columnas_experimentos(self.getCursor())

        cursor = self.getCursor()
        batch_id = _current_batch_id()
        if estado == "ejecutando":
            if batch_id:
                cursor.execute(
                    """UPDATE experimentos
                       SET estado = ?,
                           ts_inicio = COALESCE(ts_inicio, CURRENT_TIMESTAMP),
                           batch_id = COALESCE(batch_id, ?)
                       WHERE id_experimento = ?""",
                    (estado, batch_id, id),
                )
            else:
                cursor.execute(
                    """UPDATE experimentos
                       SET estado = ?,
                           ts_inicio = COALESCE(ts_inicio, CURRENT_TIMESTAMP)
                       WHERE id_experimento = ?""",
                    (estado, id),
                )
        elif estado in ("terminado", "error"):
            if batch_id:
                cursor.execute(
                    """UPDATE experimentos
                       SET estado = ?,
                           ts_fin = COALESCE(ts_fin, CURRENT_TIMESTAMP),
                           batch_id = COALESCE(batch_id, ?)
                       WHERE id_experimento = ?""",
                    (estado, batch_id, id),
                )
            else:
                cursor.execute(
                    """UPDATE experimentos
                       SET estado = ?,
                           ts_fin = COALESCE(ts_fin, CURRENT_TIMESTAMP)
                       WHERE id_experimento = ?""",
                    (estado, id),
                )
        else:
            cursor.execute(
                """UPDATE experimentos SET estado = ? WHERE id_experimento = ?""",
                (estado, id),
            )

        self.commit()
        self.desconectar()

    def insertarIteraciones(self, nombre_archivo, binary, id):
        self.conectar()

        cursor = self.getCursor()
        cursor.execute(
            f"""  INSERT INTO iteraciones (nombre, archivo, fk_id_experimento) VALUES(?, ?, ?) """,
            (nombre_archivo, binary, id),
        )

        self.commit()
        self.desconectar()

    def insertarResultados(self, BestFitness, tiempoEjecucion, Best, id):
        self.conectar()

        cursor = self.getCursor()

        cursor.execute(
            """INSERT INTO resultados VALUES (NULL, ?, ?, ?, ?)""",
            (BestFitness, round(tiempoEjecucion, 3), str(Best.tolist()), id),
        )

        self.commit()
        self.desconectar()

    def obtenerArchivos(self, instancia, incluir_binarizacion=True):
        # Si la BD base no tiene iteraciones pero existen shards, consultar shards.
        if self._deberia_consultar_shards():
            data_total = []
            if incluir_binarizacion:
                query = """
                    SELECT i.nombre, i.archivo, e.binarizacion
                    FROM experimentos e
                    INNER JOIN iteraciones i ON e.id_experimento = i.fk_id_experimento
                    INNER JOIN instancias i2 ON e.fk_id_instancia = i2.id_instancia
                    WHERE i2.nombre = ?
                    ORDER BY i2.nombre DESC, e.MH DESC
                """
            else:
                query = """
                    SELECT i.nombre, i.archivo
                    FROM experimentos e
                    INNER JOIN iteraciones i ON e.id_experimento = i.fk_id_experimento
                    INNER JOIN instancias i2 ON e.fk_id_instancia = i2.id_instancia
                    WHERE i2.nombre = ?
                    ORDER BY i2.nombre DESC, e.MH DESC
                """

            for shard_path in self._listar_shards():
                try:
                    conn = sqlite3.connect(shard_path, timeout=30)
                    cur = conn.cursor()
                    cur.execute(query, (instancia,))
                    rows = cur.fetchall() or []
                    data_total.extend(rows)
                    conn.close()
                except Exception:
                    try:
                        conn.close()
                    except Exception:
                        pass
            return data_total

        # Comportamiento normal: consultar BD configurada.
        self.conectar()
        cursor = self.getCursor()

        if incluir_binarizacion:
            query = """
                SELECT i.nombre, i.archivo, e.binarizacion 
                FROM experimentos e 
                INNER JOIN iteraciones i ON e.id_experimento = i.fk_id_experimento 
                INNER JOIN instancias i2 ON e.fk_id_instancia = i2.id_instancia 
                WHERE i2.nombre = ? 
                ORDER BY i2.nombre DESC, e.MH DESC
            """
        else:
            query = """
                SELECT i.nombre, i.archivo 
                FROM experimentos e 
                INNER JOIN iteraciones i ON e.id_experimento = i.fk_id_experimento 
                INNER JOIN instancias i2 ON e.fk_id_instancia = i2.id_instancia 
                WHERE i2.nombre = ? 
                ORDER BY i2.nombre DESC, e.MH DESC
            """

        cursor.execute(query, (instancia,))
        data = cursor.fetchall()

        self.desconectar()

        return data

    def obtenerBinarizaciones(self, instancia):
        """Obtiene las binarizaciones distintas registradas para una instancia.

        Nota: en este proyecto, cuando se usan mapas caóticos, el nombre suele venir
        codificado como "<base>_<mapa>" en la columna `experimentos.binarizacion`.
        """
        self.conectar()
        cursor = self.getCursor()

        query = """
            SELECT DISTINCT e.binarizacion
            FROM experimentos e
            INNER JOIN instancias i2 ON e.fk_id_instancia = i2.id_instancia
            WHERE i2.nombre = ?
            ORDER BY e.binarizacion ASC
        """

        cursor.execute(query, (instancia,))
        rows = cursor.fetchall() or []
        data = [r[0] for r in rows if r and r[0] is not None]

        self.desconectar()
        return data

    def obtenerInstancias(self, nombres):
        """Obtiene instancias por nombre(s).

        Args:
            nombres: lista de strings con nombres de instancias,
                     o un solo string con un nombre.
        """
        self.conectar()

        cursor = self.getCursor()

        if isinstance(nombres, str):
            nombres = [nombres]

        placeholders = ",".join("?" for _ in nombres)
        cursor.execute(
            f"SELECT DISTINCT id_instancia, nombre FROM instancias WHERE nombre IN ({placeholders})",
            nombres,
        )

        data = cursor.fetchall()

        self.desconectar()

        return data

    def obtenerOptimoInstancia(self, instancia):
        self.conectar()

        cursor = self.getCursor()
        cursor.execute("SELECT optimo FROM instancias WHERE nombre = ?", (instancia,))
        data = cursor.fetchall()

        self.desconectar()

        return data

    def reiniciarDB(self):
        self.conectar()

        self.getCursor().execute(""" DROP TABLE instancias """)
        self.getCursor().execute(""" DROP TABLE experimentos """)
        self.getCursor().execute(""" DROP TABLE resultados """)
        self.getCursor().execute(""" DROP TABLE iteraciones """)

        self.construirTablas()

        self.desconectar()
