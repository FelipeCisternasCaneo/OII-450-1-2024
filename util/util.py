import os
import json

from Util.log import log_error

def selectionSort(lista):
    posiciones = []
    
    for i in range(len(lista)):
        posiciones.append(i) 
        
    for i in range(len(lista)):
        lowest_value_index = i
        
        for j in range(i + 1, len(lista)):
            if lista[j] < lista[lowest_value_index]:
                lowest_value_index = j
            
        lista[i], lista[lowest_value_index] = lista[lowest_value_index], lista[i]
        
        posiciones[i], posiciones[lowest_value_index] = posiciones[lowest_value_index], posiciones[i]
        
    return posiciones

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
    if not os.path.exists(ruta_config):
        raise FileNotFoundError(f"El archivo de configuración '{ruta_config}' no existe.")
    
    with open(ruta_config, 'r') as archivo:
        return json.load(archivo)
    
def cargar_directorios():
    ruta_directorios = './util/json/dir.json'
    
    if not os.path.exists(ruta_directorios):
        raise FileNotFoundError(f"El archivo de directorios '{ruta_directorios}' no existe.")
    
    with open(ruta_directorios, 'r') as archivo:
        return json.load(archivo)
    
def cargar_configuracion_exp(CONFIG_FILE, EXPERIMENTS_FILE):
    """Carga la configuración y los experimentos desde archivos JSON."""
    with open(CONFIG_FILE, 'r') as config_file:
        config = json.load(config_file)
    with open(EXPERIMENTS_FILE, 'r') as experiments_file:
        experiments = json.load(experiments_file)
    return config, experiments
    
def parse_parametros(parametrosMH):
    """Parsea los parámetros y devuelve un diccionario con claves y valores."""
    params = {}
    
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
    
    base_dir = "./Resultados"
    
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

    # Crear las carpetas si no existen
    for subcarpeta in subcarpetas:
        ruta_completa = os.path.join(base_dir, subcarpeta)
        if not os.path.exists(ruta_completa):
            os.makedirs(ruta_completa)

def asegurar_directorio(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# Util json

#"mhs": ["AOA", "EBWOA", "EOO", "FLO", "FOX", "GA", "GOA", "GWO", "HBA", "HLOA", "LOA", "NO", "PO", "POA", "PSO",
#    "QSO", "RSA", "SBOA", "SCA", "SHO", "TDO", "WOA", "WOM"]