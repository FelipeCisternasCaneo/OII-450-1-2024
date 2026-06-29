import os
from problem.optima import OPTIMA_CLASSICAL, OPTIMA_CEC2017
from problem.SCP.problem import SetCoveringProblem

def crear_tablas(cursor):
    """Crea la estructura de tablas e índices en la base de datos si no existen."""
    cursor.execute(
        """ CREATE TABLE IF NOT EXISTS instancias(
            id_instancia INTEGER PRIMARY KEY AUTOINCREMENT,
            tipo_problema TEXT,
            nombre TEXT,
            optimo REAL,
            param TEXT
        )"""
    )

    cursor.execute(
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

    cursor.execute(
        """ CREATE TABLE IF NOT EXISTS resultados(
            id_resultado INTEGER PRIMARY KEY AUTOINCREMENT,
            fitness REAL,
            tiempoEjecucion REAL,
            solucion TEXT,
            fk_id_experimento INTEGER,
            FOREIGN KEY (fk_id_experimento) REFERENCES experimentos (id_experimento)
        )"""
    )

    cursor.execute(
        """ CREATE TABLE IF NOT EXISTS iteraciones(
            id_archivo INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT,
            archivo BLOB,
            fk_id_experimento INTEGER,
            FOREIGN KEY (fk_id_experimento) REFERENCES experimentos (id_experimento)
        )"""
    )

    # Crear índices para optimizar consultas frecuentes
    cursor.execute(
        """CREATE INDEX IF NOT EXISTS idx_estado ON experimentos(estado)"""
    )
    cursor.execute(
        """CREATE INDEX IF NOT EXISTS idx_instancia_nombre ON instancias(nombre)"""
    )
    cursor.execute(
        """CREATE INDEX IF NOT EXISTS idx_iteraciones_fk ON iteraciones(fk_id_experimento)"""
    )
    cursor.execute(
        """CREATE INDEX IF NOT EXISTS idx_resultados_fk ON resultados(fk_id_experimento)"""
    )


def insertar_instancias_ben(bd):
    """Pobla las instancias de BEN clásicas F1-F23 en la base de datos."""
    tipoProblema = "BEN"

    # Filtrar solo funciones clásicas (F1-F23), excluir CEC2017
    funciones_clasicas = [f for f in bd.data if not f.endswith("CEC2017")]

    config_ben = {
        "F1": ("-100", "100", OPTIMA_CLASSICAL["F1"]),
        "F2": ("-10", "10", OPTIMA_CLASSICAL["F2"]),
        "F3": ("-100", "100", OPTIMA_CLASSICAL["F3"]),
        "F4": ("-100", "100", OPTIMA_CLASSICAL["F4"]),
        "F5": ("-30", "30", OPTIMA_CLASSICAL["F5"]),
        "F6": ("-100", "100", OPTIMA_CLASSICAL["F6"]),
        "F7": ("-1.28", "1.28", OPTIMA_CLASSICAL["F7"]),
        "F8": ("-500", "500", OPTIMA_CLASSICAL["F8"]),
        "F9": ("-5.12", "5.12", OPTIMA_CLASSICAL["F9"]),
        "F10": ("-32", "32", OPTIMA_CLASSICAL["F10"]),
        "F11": ("-600", "600", OPTIMA_CLASSICAL["F11"]),
        "F12": ("-50", "50", OPTIMA_CLASSICAL["F12"]),
        "F13": ("-50", "50", OPTIMA_CLASSICAL["F13"]),
        "F14": ("-65.536", "65.536", OPTIMA_CLASSICAL["F14"]),
        "F15": ("-5", "5", OPTIMA_CLASSICAL["F15"]),
        "F16": ("-5", "5", OPTIMA_CLASSICAL["F16"]),
        "F17": ("-5", "5", OPTIMA_CLASSICAL["F17"]),
        "F18": ("-2", "2", OPTIMA_CLASSICAL["F18"]),
        "F19": ("0", "1", OPTIMA_CLASSICAL["F19"]),
        "F20": ("0", "1", OPTIMA_CLASSICAL["F20"]),
        "F21": ("0", "10", OPTIMA_CLASSICAL["F21"]),
        "F22": ("0", "10", OPTIMA_CLASSICAL["F22"]),
        "F23": ("0", "10", OPTIMA_CLASSICAL["F23"]),
    }

    for instancia in funciones_clasicas:
        if instancia not in config_ben:
            raise ValueError(
                f"Advertencia: La función '{instancia}' no está definida en la configuración de BEN."
            )

        lb, ub, optimo = config_ben[instancia]
        param = f"lb:{lb},ub:{ub}"

        bd.getCursor().execute(
            """INSERT INTO instancias (tipo_problema, nombre, optimo, param) VALUES(?, ?, ?, ?)""",
            (tipoProblema, instancia, optimo, param),
        )


def insertar_instancias_cec2017(bd):
    """Pobla las instancias de CEC2017 en la base de datos."""
    from problem.Benchmark.CEC.cec2017.functions import all_functions

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
        optimo = OPTIMA_CEC2017.get(nombre_base, 0)

        # Insertar en base de datos (tipo 'CEC2017')
        bd.getCursor().execute(
            """INSERT INTO instancias (tipo_problema, nombre, optimo, param) VALUES(?, ?, ?, ?)""",
            (tipoProblema, nombre_funcion, optimo, param),
        )


def insertar_instancias_scp(bd):
    """Pobla las instancias de SCP en la base de datos leyendo de src/problem/SCP/Instances/."""
    instances_dir = bd._abs_repo_path("src", "problem", "SCP", "Instances")
    data = os.listdir(instances_dir)

    for d in data:
        tipoProblema = "SCP"
        nombre = d.split(".")[0]
        optimo = SetCoveringProblem.get_known_optimum(nombre, unicost=False)
        nombre = f"{nombre[3:]}"
        param = ""

        bd.getCursor().execute(
            f"""  INSERT INTO instancias (tipo_problema, nombre, optimo, param) VALUES(?, ?, ?, ?) """,
            (tipoProblema, nombre, optimo, param),
        )


def insertar_instancias_uscp(bd):
    """Pobla las instancias de USCP en la base de datos leyendo de src/problem/USCP/Instances/."""
    instances_dir = bd._abs_repo_path("src", "problem", "USCP", "Instances")
    data = os.listdir(instances_dir)
    for d in data:
        tipoProblema = "USCP"
        nombre = d.split(".")[0]
        optimo = SetCoveringProblem.get_known_optimum(nombre, unicost=True)

        nombre = f"u{nombre[4:]}"
        param = ""

        bd.getCursor().execute(
            f"""  INSERT INTO instancias (tipo_problema, nombre, optimo, param) VALUES(?, ?, ?, ?) """,
            (tipoProblema, nombre, optimo, param),
        )
