import os
import json

from Util.log import log_error


def _workspace_root() -> str:
    """Devuelve la raíz del proyecto asumiendo que este archivo vive en /Util/."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolver_ruta_existente(ruta: str) -> str:
    """Resuelve una ruta a un archivo existente.

    Acepta rutas relativas (respecto al CWD o a la raíz del proyecto) y maneja
    el caso común de usar 'util/' cuando el directorio real es 'Util/'.
    """
    if not ruta:
        raise ValueError("La ruta de configuración está vacía.")

    candidatos: list[str] = []

    # 1) Tal cual (relativa al CWD o absoluta)
    candidatos.append(ruta)

    # 2) Relativa a la raíz del workspace
    root = _workspace_root()
    candidatos.append(os.path.join(root, ruta.lstrip("./")))

    # 3) Normalización por caso (util -> Util) para Linux
    normalizada = ruta.replace("\\", "/")
    prefijos = ("util/", "./util/")
    for prefijo in prefijos:
        if normalizada.startswith(prefijo):
            resto = normalizada[len(prefijo):]
            candidatos.append(os.path.join(root, "Util", resto))

    # 4) Variantes directas del CWD
    if normalizada.startswith("./util/"):
        candidatos.append(normalizada.replace("./util/", "./Util/", 1))
    if normalizada.startswith("util/"):
        candidatos.append(normalizada.replace("util/", "Util/", 1))

    # Devolver el primero que exista
    for candidato in candidatos:
        if os.path.exists(candidato):
            return candidato

    # Mensaje útil con pistas
    candidatos_str = "\n".join(f"- {c}" for c in candidatos)
    raise FileNotFoundError(
        f"El archivo de configuración '{ruta}' no existe.\n"
        f"Rutas intentadas:\n{candidatos_str}"
    )

# Create a function that converts a digital file into binary
def convert_into_binary(file_path):
    with open(file_path, 'rb') as file:
        binary = file.read()

    return binary

def writeTofile(data, filename):
    # Convert binary data to proper format and write it on Hard Disk
    with open(filename, 'wb') as file:
        file.write(data)
        
def cargar_configuracion(ruta_config):
    ruta_resuelta = _resolver_ruta_existente(ruta_config)
    with open(ruta_resuelta, 'r', encoding='utf-8') as archivo:
        return json.load(archivo)
    
def cargar_directorios():
    return cargar_configuracion('./util/json/dir.json')
    
def cargar_configuracion_exp(CONFIG_FILE, EXPERIMENTS_FILE):
    """Carga la configuración y los experimentos desde archivos JSON."""
    config_path = _resolver_ruta_existente(CONFIG_FILE)
    experiments_path = _resolver_ruta_existente(EXPERIMENTS_FILE)

    with open(config_path, 'r', encoding='utf-8') as config_file:
        config = json.load(config_file)
    with open(experiments_path, 'r', encoding='utf-8') as experiments_file:
        experiments = json.load(experiments_file)
    return config, experiments
    
def parse_parametros(parametrosMH):
    """Parsea los parámetros y devuelve un diccionario con claves y valores."""
    params = {}
    
    # Reemplazar ; por , para unificar separadores
    parametrosMH = parametrosMH.replace(';', ',')
    
    for param in parametrosMH.split(","):
        # Verificar si el parámetro tiene el formato clave:valor
        if ":" in param:
            try:
                key, value = param.split(":", 1)
                params[key.strip()] = value.strip()
            except Exception:
                log_error("Parseo de parámetros", "Ha ocurrido un error en el parseo de un parámetro.")
                raise ValueError("Error en el parseo de un parámetro.")
        else:
            log_error("Parseo de parámetros", "Ha ocurrido un error en el parseo de un parámetro.")
            raise ValueError("Error en el parseo de un parámetro.")
    
    return params

def verificar_y_crear_carpetas():
    """
    Verifica que las carpetas y subcarpetas dentro de 'Resultados' estén creadas.
    Si no existen, las crea automáticamente.
    """
    
    root_dir = _workspace_root()

    base_dir = os.path.join(root_dir, "Resultados")
    
    # Definir las subcarpetas necesarias
    subcarpetas = [
        "transitorio",
        "graficos",
        "best",
        "boxplot",
        "violinplot",
        "resumen",
        "fitness"
    ]

    # Crear las carpetas (seguro en paralelo)
    for subcarpeta in subcarpetas:
        ruta_completa = os.path.join(base_dir, subcarpeta)
        os.makedirs(ruta_completa, exist_ok=True)

    # Logs (incluye salida Slurm definida en run_main.sh)
    os.makedirs(os.path.join(root_dir, "Logs"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "Logs", "SSH"), exist_ok=True)

def asegurar_directorio(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# Util json

#"mhs": ["AOA", "EBWOA", "EOO", "FLO", "FOX", "GA", "GOA", "GWO", "HBA", "HLOA", "LOA", "NO", "PO", "POA", "PSO",
#    "QSO", "RSA", "SBOA", "SCA", "SHO", "TDO", "WOA", "WOM"]