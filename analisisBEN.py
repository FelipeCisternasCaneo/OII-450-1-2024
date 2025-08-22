import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Util.util import cargar_configuracion_exp, writeTofile
from BD.sqlite import BD
from Util.log import escribir_resumenes

# === Carga de configuraciones ===
CONFIG_FILE = './util/json/dir.json'
EXPERIMENTS_FILE = './util/json/experiments_config.json'

CONFIG, EXPERIMENTS = cargar_configuracion_exp(CONFIG_FILE, EXPERIMENTS_FILE)

# === Definición de directorios ===
DIRS = CONFIG["dirs"]

DIR_FITNESS      = DIRS["fitness"]
DIR_RESUMEN      = DIRS["resumen"]
DIR_RESULTADO    = DIRS["base"]
DIR_TRANSITORIO  = DIRS["transitorio"]
DIR_GRAFICOS     = DIRS["graficos"]
DIR_BEST         = DIRS["best"]
DIR_BOXPLOT      = DIRS["boxplot"]
DIR_VIOLIN       = DIRS["violinplot"]

# === Parámetros generales ===
GRAFICOS = True # <-- Cambiar a True para generar gráficos para cada corrida
MHS_LIST = EXPERIMENTS["mhs"]
COLORS = ['r', 'g']

# === Clase de almacenamiento de resultados ===
class InstancesMhs:
    def __init__(self):
        self.div = []
        self.fitness = []
        self.time = []
        self.xpl = []
        self.xpt = []
        self.bestFitness = []
        self.bestTime = []

# === Inicialización ===
#mhs_instances = {name: InstancesMhs() for name in MHS_LIST}
bd = BD()

# === Función para actualizar Datos ===
def actualizar_datos(mhs_instances, mh, archivo_fitness, data):
    instancia = mhs_instances[mh]
    instancia.fitness.append(np.min(data['fitness']))
    instancia.time.append(np.round(np.sum(data['time']), 3))
    instancia.xpl.append(np.round(np.mean(data['XPL']), 2))
    instancia.xpt.append(np.round(np.mean(data['XPT']), 2))
    archivo_fitness.write(f'{mh}, {np.min(data["fitness"])}\n')

def graficar_datos(iteraciones, fitness, xpl, xpt, tiempo, mh, problem, corrida):
    """
    Genera gráficos de convergencia, porcentajes XPL/XPT y tiempo por iteración.
    Guarda los resultados en archivos PDF dentro del directorio configurado.
    """
    # Directorio base para los gráficos de esta corrida
    output_dir = os.path.join(DIR_GRAFICOS, 'BEN')
    os.makedirs(output_dir, exist_ok=True)

    # --- Gráfico de convergencia ---
    path_convergencia = os.path.join(output_dir, f'Convergence_{mh}_BEN_{problem}_{corrida}.pdf')
    _, ax = plt.subplots()
    ax.plot(iteraciones, fitness, marker='o')
    ax.set_title(f'Convergence {mh}\n {problem} - Run {corrida})')
    ax.set_ylabel("Fitness")
    ax.set_xlabel("Iteration")
    plt.tight_layout()
    plt.savefig(path_convergencia)
    plt.close()

    # --- Gráfico XPL vs XPT ---
    path_porcentaje = os.path.join(output_dir, f'Percentage_{mh}_BEN_{problem}_{corrida}.pdf')
    _, axPER = plt.subplots()
    axPER.plot(iteraciones, xpl, color="r", label=rf"$\overline{{XPL}}$: {np.round(np.mean(xpl), 2)}%")
    axPER.plot(iteraciones, xpt, color="b", label=rf"$\overline{{XPT}}$: {np.round(np.mean(xpt), 2)}%")
    axPER.set_title(f'XPL% - XPT% {mh}\ {problem} - Run {corrida}')
    axPER.set_ylabel("Percentage")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path_porcentaje)
    plt.close()

    # --- Gráfico de tiempo por iteración ---
    path_tiempo = os.path.join(output_dir, f'Time_{mh}_BEN_{problem}_{corrida}.pdf')
    _, axTime = plt.subplots()
    axTime.plot(iteraciones, tiempo, color='g', label='Time per Iteration')
    axTime.set_title(f'Time per Iteration {mh}\n {problem} - Run {corrida}')
    axTime.set_ylabel("Time (s)")
    axTime.set_xlabel("Iteration")
    axTime.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path_tiempo)
    plt.close()

def graficar_boxplot_violin(instancia):
    """
    Genera gráficos Boxplot y Violinplot para los valores de fitness
    según cada metaheurística aplicada en una instancia.

    Parámetros:
        instancia (str): Nombre o número de la instancia del problema.
    """

    # Ruta al archivo de datos
    direccion_datos = os.path.join(DIR_FITNESS, f'BEN/fitness_BEN_{instancia}.csv')

    # Cargar y validar los datos
    try:
        datos = pd.read_csv(direccion_datos)
        datos.columns = datos.columns.str.strip()  # Limpia espacios
        if 'FITNESS' not in datos.columns or 'MH' not in datos.columns:
            print(f"[ERROR] Columnas necesarias no encontradas en {direccion_datos}")
            return
    except FileNotFoundError:
        print(f"[ERROR] Archivo no encontrado: {direccion_datos}")
        return
    except Exception as e:
        print(f"[ERROR] Error al leer {direccion_datos}: {e}")
        return

    # --- Boxplot ---
    output_dir_box = os.path.join(DIR_BOXPLOT, 'BEN')
    os.makedirs(output_dir_box, exist_ok=True)
    file_path_box = os.path.join(output_dir_box, f'boxplot_fitness_BEN_{instancia}.pdf')

    sns.boxplot(x='MH', y='FITNESS', data=datos, hue='MH', palette='Set2', legend=False)
    plt.title(f'Boxplot Fitness\n {instancia}')
    plt.xlabel('Metaheurística')
    plt.ylabel('Fitness')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(file_path_box)
    plt.close()

    # --- Violinplot ---
    output_dir_violin = os.path.join(DIR_VIOLIN, 'BEN')
    os.makedirs(output_dir_violin, exist_ok=True)
    file_path_violin = os.path.join(output_dir_violin, f'violinplot_fitness_BEN_{instancia}.pdf')

    sns.violinplot(x='MH', y='FITNESS', data=datos, hue='MH', palette='Set3', legend=False)
    plt.title(f'Violinplot Fitness\n {instancia}')
    plt.xlabel('Metaheurística')
    plt.ylabel('Fitness')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(file_path_violin)
    plt.close()

def procesar_archivos(instancia, blob, archivo_fitness, mhs_instances):
    """
    Procesa una lista de archivos (blob), extrae sus datos y genera gráficos asociados
    a una instancia específica. Los resultados se guardan en archivos CSV y gráficos PDF.

    Parámetros:
        instancia (str): Identificador del problema a analizar.
        blob (list): Lista de tuplas (nombre_archivo, contenido).
        archivo_fitness (file): Archivo abierto para escribir resultados de fitness.
    """
    corrida = 1  # Contador de ejecuciones

    for nombre_archivo, contenido in blob:
        # Validar nombre del archivo
        try:
            mh, _ = nombre_archivo.split('_')[:2]
        except ValueError:
            print(f"[ADVERTENCIA] Archivo '{nombre_archivo}' con nombre inválido. Se omite.")
            continue

        # Guardar contenido temporal para su análisis
        direccion_destino = os.path.join(DIR_TRANSITORIO, f'{nombre_archivo}.csv')
        writeTofile(contenido, direccion_destino)

        # Intentar leer el archivo temporal
        try:
            data = pd.read_csv(direccion_destino)
        except Exception as e:
            print(f"[ERROR] Fallo al leer '{direccion_destino}': {e}")
            os.remove(direccion_destino)
            continue

        # Solo procesar si la metaheurística está registrada
        if mh in MHS_LIST:
            actualizar_datos(mhs_instances, mh, archivo_fitness, data)
            mhs_instances[mh].bestFitness = data['fitness']
            mhs_instances[mh].bestTime = data['time']

        # Generar gráficos por corrida
        if GRAFICOS:
            graficar_datos(
                iteraciones=data['iter'],
                fitness=data['fitness'],
                xpl=data['XPL'],
                xpt=data['XPT'],
                tiempo=data['time'],
                mh=mh,
                problem=instancia,
                corrida=corrida,
            )

        # Limpiar archivo temporal
        os.remove(direccion_destino)

        corrida += 1  # Siguiente corrida

    archivo_fitness.close()

def graficar_mejores_resultados(instancia, mhs_instances):
    """
    Genera gráficos de comparación para los mejores valores de fitness y tiempo
    alcanzados por cada metaheurística en una instancia.

    Parámetros:
        instancia (str): Nombre o identificador de la instancia.
        mhs_instances (dict): Diccionario con las instancias de cada MH.
    """
    
    mejor_fitness = float('inf')
    mejor_tiempo = float('inf')
    mh_mejor_fitness = ""
    mh_mejor_tiempo = ""

    # Buscar la mejor metaheurística en fitness y tiempo
    for name in MHS_LIST:
        mh = mhs_instances[name]
        min_fitness = min(mh.bestFitness)
        min_tiempo = min(mh.bestTime)

        if min_fitness < mejor_fitness:
            mejor_fitness = min_fitness
            mh_mejor_fitness = name

        if min_tiempo < mejor_tiempo:
            mejor_tiempo = min_tiempo
            mh_mejor_tiempo = name

    # Crear carpeta de salida si no existe
    output_dir = os.path.join(DIR_BEST, 'BEN')
    os.makedirs(output_dir, exist_ok=True)

    for name in MHS_LIST:
        mh = mhs_instances[name]
        plt.plot(range(len(mh.bestFitness)), mh.bestFitness, label=name)
    
    plt.title(f'Best Fitness per MH \n {instancia}\nMejor: {mh_mejor_fitness} ({mejor_fitness})')
    plt.ylabel("Fitness")
    plt.xlabel("Iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fitness_BEN_{instancia}.pdf'))
    plt.close()

    for name in MHS_LIST:
        mh = mhs_instances[name]
        plt.plot(range(len(mh.bestTime)), mh.bestTime, label=name)
    plt.title(f'Best Time per MH \n {instancia}\nMejor: {mh_mejor_tiempo} ({mejor_tiempo:.2f} s)')
    plt.ylabel("Time (s)")
    plt.xlabel("Iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'time_BEN_{instancia}.pdf'))
    plt.close()

# Crear carpeta transitorio si no existe
os.makedirs(DIR_TRANSITORIO, exist_ok=True)

def analizar_instancias():

    # Obtener lista de instancias
    lista_instancias = ', '.join([f'"{func}"' for func in EXPERIMENTS["instancias"]["BEN"]])
    instancias = bd.obtenerInstancias(lista_instancias)

    print("Iniciando procesamiento de instancias...\n")

    # Procesar cada instancia
    for instancia in instancias:
        instancia_id = instancia[1]
        print(f"[INFO] Procesando instancia: {instancia_id}")
        blob = bd.obtenerArchivos(instancia_id, incluir_binarizacion=False)

        if not blob:
            print(f"[ADVERTENCIA] La instancia {instancia_id} no tiene experimentos asociados. Saltando...\n")
            continue
        
        mhs_instances_local = {name: InstancesMhs() for name in MHS_LIST}

        # Preparar carpetas de salida
        output_dir_resumen = os.path.join(DIR_RESUMEN, 'BEN')
        output_dir_fitness = os.path.join(DIR_FITNESS, 'BEN')
        os.makedirs(output_dir_resumen, exist_ok=True)
        os.makedirs(output_dir_fitness, exist_ok=True)

        # Inicializar archivos de salida
        archivoResumenFitness = open(os.path.join(output_dir_resumen, f'resumen_fitness_BEN_{instancia_id}.csv'), 'w')
        archivoResumenTimes = open(os.path.join(output_dir_resumen, f'resumen_times_BEN_{instancia_id}.csv'), 'w')
        archivoResumenPercentage = open(os.path.join(output_dir_resumen, f'resumen_percentage_BEN_{instancia_id}.csv'), 'w')
        archivoFitness = open(os.path.join(output_dir_fitness, f'fitness_BEN_{instancia_id}.csv'), 'w')

        # Escribir encabezados
        archivoResumenFitness.write("instance, best, avg. fitness, std fitness\n")
        archivoResumenTimes.write("instance, min time (s), avg. time (s), std time (s)\n")
        archivoResumenPercentage.write("instance, avg. XPL%, avg. XPT%\n")
        archivoFitness.write("MH, FITNESS\n")

        # Procesar resultados y escribir datos
        procesar_archivos(instancia_id, blob, archivoFitness, mhs_instances_local)

        # Escribir resúmenes estadísticos
        escribir_resumenes(mhs_instances_local, archivoResumenFitness, archivoResumenTimes, archivoResumenPercentage, MHS_LIST)

        # Generar gráficos resumen
        graficar_mejores_resultados(instancia_id, mhs_instances_local)
        graficar_boxplot_violin(instancia_id)

        # Cerrar archivos
        archivoResumenFitness.close()
        archivoResumenTimes.close()
        archivoResumenPercentage.close()
        archivoFitness.close()

        print("")  # Separación visual entre instancias

    print("[INFO] Análisis BEN completado con éxito.")
    print("-" * 50)
