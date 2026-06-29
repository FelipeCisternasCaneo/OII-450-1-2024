import sqlite3
import os
import time
from bd.connection import SQLiteConnectionManager, _current_batch_id
from bd import schema


class BD(SQLiteConnectionManager):
    """
    Repositorio de acceso a datos para experimentos, instancias y resultados.
    Hereda la gestión de conexiones y transacciones de SQLiteConnectionManager.
    """

    def __init__(self):
        super().__init__()
        self._experimentos_cols_aseguradas = False

    # Datos estáticos de funciones para compatibilidad / inicialización
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

    def construirTablas(self):
        self.conectar()
        schema.crear_tablas(self.getCursor())
        self.commit()

        # Asegurar columnas nuevas si la tabla existía previamente sin ellas
        self._asegurar_columnas_experimentos(self.getCursor())
        self.commit()

        # Insertar instancias de todos los dominios registrados
        from solver.domain_managers import ensure_registered
        from solver.domain_managers.registry import get_all as get_all_domains

        ensure_registered()

        for _dtype, entry in get_all_domains().items():
            if entry.insert_instances is not None:
                entry.insert_instances(self)

        self.desconectar()

    # Métodos de inserción delegados a schema.py para compatibilidad polimórfica
    def insertarInstanciasBEN(self):
        self.conectar()
        schema.insertar_instancias_ben(self)
        self.commit()
        self.desconectar()

    def insertarInstanciasCEC2017(self):
        self.conectar()
        schema.insertar_instancias_cec2017(self)
        self.commit()
        self.desconectar()

    def insertarInstanciasSCP(self):
        self.conectar()
        schema.insertar_instancias_scp(self)
        self.commit()
        self.desconectar()

    def insertarInstanciasUSCP(self):
        self.conectar()
        schema.insertar_instancias_uscp(self)
        self.commit()
        self.desconectar()

    # ── Operaciones de experimentos y resultados ──────────────────────────────────

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

                        if not self._experimentos_cols_aseguradas:
                            try:
                                self._asegurar_columnas_experimentos(cursor)
                                self._experimentos_cols_aseguradas = True
                            except sqlite3.OperationalError as e:
                                conn.execute("ROLLBACK;")
                                msg = str(e).lower()
                                if "locked" in msg or "busy" in msg:
                                    time.sleep(min(1.0, sleep_base * (1 + intento)))
                                    continue
                                raise
                            except Exception:
                                self._experimentos_cols_aseguradas = True

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
            """INSERT INTO iteraciones (nombre, archivo, fk_id_experimento) VALUES(?, ?, ?)""",
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
        self.getCursor().execute(""" DROP TABLE IF EXISTS instancias """)
        self.getCursor().execute(""" DROP TABLE IF EXISTS experimentos """)
        self.getCursor().execute(""" DROP TABLE IF EXISTS resultados """)
        self.getCursor().execute(""" DROP TABLE IF EXISTS iteraciones """)
        self.construirTablas()
        self.desconectar()
