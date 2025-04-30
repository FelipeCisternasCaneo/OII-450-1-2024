import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Util.util import cargar_configuracion_exp, writeTofile
from BD.sqlite import BD
from Util.log import escribir_resumenes

# === Carga de Configuraciones ===
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
GRAFICOS = False # <-- Cambiar a True para generar gráficos para cada corrida
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
# mhs_instances = {name: InstancesMhs() for name in MHS_LIST}
bd = BD()

# === Función para actualizar Datos ===
def actualizar_datos(mhs_instances, mh, archivo_fitness, data):
    instancia = mhs_instances[mh]
    instancia.fitness.append(np.min(data['fitness']))
    instancia.time.append(np.round(np.sum(data['time']), 3))
    instancia.xpl.append(np.round(np.mean(data['XPL']), 2))
    instancia.xpt.append(np.round(np.mean(data['XPT']), 2))
    archivo_fitness.write(f'{mh}, {np.min(data["fitness"])}\n')

def graficar_datos(iteraciones, fitness, xpl, xpt, tiempo, mh, problem, corrida, binarizacion):
    """
    Genera gráficos de convergencia, porcentajes XPL/XPT y tiempo por iteración.
    Guarda los resultados en archivos PDF dentro del directorio configurado.
    """
    # Directorio base para los gráficos de esta corrida
    output_dir = os.path.join(DIR_GRAFICOS, 'SCP', str(binarizacion))
    os.makedirs(output_dir, exist_ok=True)

    # --- Gráfico de convergencia ---
    path_convergencia = os.path.join(output_dir, f'Convergence_{mh}_SCP_{problem}_{corrida}_{binarizacion}.pdf')
    _, ax = plt.subplots()
    ax.plot(iteraciones, fitness, marker='o')
    ax.set_title(f'Convergence {mh}\nscp{problem} - Run {corrida} - ({binarizacion})')
    ax.set_ylabel("Fitness")
    ax.set_xlabel("Iteration")
    plt.tight_layout()
    plt.savefig(path_convergencia)
    plt.close()

    # --- Gráfico XPL vs XPT ---
    path_porcentaje = os.path.join(output_dir, f'Percentage_{mh}_SCP_{problem}_{corrida}_{binarizacion}.pdf')
    _, axPER = plt.subplots()
    axPER.plot(iteraciones, xpl, color="r", label=rf"$\overline{{XPL}}$: {np.round(np.mean(xpl), 2)}%")
    axPER.plot(iteraciones, xpt, color="b", label=rf"$\overline{{XPT}}$: {np.round(np.mean(xpt), 2)}%")
    axPER.set_title(f'XPL% - XPT% {mh}\nscp{problem} - Run {corrida} - ({binarizacion})')
    axPER.set_ylabel("Percentage")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path_porcentaje)
    plt.close()

    # --- Gráfico de tiempo por iteración ---
    path_tiempo = os.path.join(output_dir, f'Time_{mh}_SCP_{problem}_{corrida}_{binarizacion}.pdf')
    _, axTime = plt.subplots()
    axTime.plot(iteraciones, tiempo, color='g', label='Time per Iteration')
    axTime.set_title(f'Time per Iteration {mh}\nscp{problem} - Run {corrida} - ({binarizacion})')
    axTime.set_ylabel("Time (s)")
    axTime.set_xlabel("Iteration")
    axTime.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path_tiempo)
    plt.close()

def graficar_boxplot_violin(instancia, binarizacion):
    """
    Genera gráficos Boxplot y Violinplot para los valores de fitness
    según cada metaheurística aplicada en una instancia del SCP.

    Parámetros:
        instancia (str): Nombre o número de la instancia del problema.
        binarizacion (str/int): Tipo de técnica de binarización usada.
    """

    # Ruta al archivo de datos
    direccion_datos = os.path.join(DIR_FITNESS, f'SCP/fitness_SCP_{instancia}_{binarizacion}.csv')

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
    output_dir_box = os.path.join(DIR_BOXPLOT, 'SCP')
    os.makedirs(output_dir_box, exist_ok=True)
    file_path_box = os.path.join(output_dir_box, f'boxplot_fitness_SCP_{instancia}_{binarizacion}.pdf')

    sns.boxplot(x='MH', y='FITNESS', data=datos, hue='MH', palette='Set2', legend=False)
    plt.title(f'Boxplot Fitness\nscp{instancia} - {binarizacion}')
    plt.xlabel('Metaheurística')
    plt.ylabel('Fitness')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(file_path_box)
    plt.close()

    # --- Violinplot ---
    output_dir_violin = os.path.join(DIR_VIOLIN, 'SCP')
    os.makedirs(output_dir_violin, exist_ok=True)
    file_path_violin = os.path.join(output_dir_violin, f'violinplot_fitness_SCP_{instancia}_{binarizacion}.pdf')

    sns.violinplot(x='MH', y='FITNESS', data=datos, hue='MH', palette='Set3', legend=False)
    plt.title(f'Violinplot Fitness\nscp{instancia} - {binarizacion}')
    plt.xlabel('Metaheurística')
    plt.ylabel('Fitness')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(file_path_violin)
    plt.close()

def procesar_archivos(instancia, blob, archivo_fitness, bin_actual, mhs_instances):
    """
    Procesa una lista de archivos (blob), extrae sus datos y genera gráficos asociados
    a una instancia específica del SCP, utilizando una binarización concreta.

    Parámetros:
        instancia (str): Identificador del problema SCP a analizar.
        blob (list): Lista de tuplas (nombre_archivo, contenido, binarizacion).
        archivo_fitness (file): Archivo abierto para escribir resultados de fitness.
        bin_actual (str): Tipo de binarización a procesar en esta ejecución.
    """
    corrida = 1  # Contador de ejecuciones

    for nombre_archivo, contenido, binarizacion in blob:
        # Validar nombre del archivo
        try:
            mh, _ = nombre_archivo.split('_')[:2]
        except ValueError:
            print(f"[ADVERTENCIA] Archivo '{nombre_archivo}' con nombre inválido. Se omite.")
            continue

        # Filtrar archivos según la binarización actual
        if binarizacion.strip() != bin_actual:
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
                binarizacion=binarizacion
            )

        # Limpiar archivo temporal
        os.remove(direccion_destino)

        corrida += 1  # Siguiente corrida

    archivo_fitness.close()

def graficar_mejores_resultados(instancia, mhs_instances, binarizacion):
    """
    Genera gráficos de comparación para los mejores valores de fitness y tiempo
    alcanzados por cada metaheurística en una instancia del SCP.

    Parámetros:
        instancia (str): Nombre o identificador de la instancia SCP.
        mhs_instances (dict): Diccionario con las instancias de cada MH.
        binarizacion (str): Nombre del esquema de binarización utilizado.
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
    output_dir = os.path.join(DIR_BEST, 'SCP')
    os.makedirs(output_dir, exist_ok=True)

    # --- Gráfico de mejores fitness ---
    
    for name in MHS_LIST:
        mh = mhs_instances[name]
        plt.plot(range(len(mh.bestFitness)), mh.bestFitness, label=name)
    
    plt.title(f'Best Fitness per MH \n scp{instancia} - {binarizacion}\nMejor: {mh_mejor_fitness} ({mejor_fitness})')
    plt.ylabel("Fitness")
    plt.xlabel("Iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fitness_SCP_{instancia}_{binarizacion}.pdf'))
    plt.close()

    # --- Gráfico de mejores tiempos ---
    for name in MHS_LIST:
        mh = mhs_instances[name]
        plt.plot(range(len(mh.bestTime)), mh.bestTime, label=name)
    plt.title(f'Best Time per MH \n scp{instancia} - {binarizacion}\nMejor: {mh_mejor_tiempo} ({mejor_tiempo:.2f} s)')
    plt.ylabel("Time (s)")
    plt.xlabel("Iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'time_SCP_{instancia}_{binarizacion}.pdf'))
    plt.close()


def analizar_instancias():
    """
    Función principal para analizar instancias del problema SCP.
    Procesa cada instancia, genera gráficos y resúmenes estadísticos.
    """

    # Crear carpeta transitorio si no existe
    os.makedirs(DIR_TRANSITORIO, exist_ok=True)

    # Obtener lista de instancias y binarizaciones
    lista_instancias = ', '.join([f'"{func}"' for func in EXPERIMENTS["instancias"]["SCP"]])
    lista_bin = EXPERIMENTS["DS_actions"]
    instancias = bd.obtenerInstancias(lista_instancias)

    print("Iniciando procesamiento de instancias...\n")

    # Procesar cada instancia
    for instancia in instancias:
        instancia_id = instancia[1]
        print(f"[INFO] Procesando instancia: scp{instancia_id}")
        blob = bd.obtenerArchivos(instancia_id)

        if not blob:
            print(f"[ADVERTENCIA] La instancia scp{instancia_id} no tiene experimentos asociados. Saltando...\n")
            continue

        for binarizacion in lista_bin:
            print(f"    ↳ Binarización: {binarizacion}")
            
            mhs_instances_local = {name: InstancesMhs() for name in MHS_LIST}

            # Preparar carpetas de salida
            output_dir_resumen = os.path.join(DIR_RESUMEN, 'SCP')
            output_dir_fitness = os.path.join(DIR_FITNESS, 'SCP')
            os.makedirs(output_dir_resumen, exist_ok=True)
            os.makedirs(output_dir_fitness, exist_ok=True)

            # Inicializar archivos de salida
            archivoResumenFitness = open(os.path.join(output_dir_resumen, f'resumen_fitness_SCP_{instancia_id}_{binarizacion}.csv'), 'w')
            archivoResumenTimes = open(os.path.join(output_dir_resumen, f'resumen_times_SCP_{instancia_id}_{binarizacion}.csv'), 'w')
            archivoResumenPercentage = open(os.path.join(output_dir_resumen, f'resumen_percentage_SCP_{instancia_id}_{binarizacion}.csv'), 'w')
            archivoFitness = open(os.path.join(output_dir_fitness, f'fitness_SCP_{instancia_id}_{binarizacion}.csv'), 'w')

            # Escribir encabezados
            archivoResumenFitness.write("instance, best, avg. fitness, std fitness\n")
            archivoResumenTimes.write("instance, min time (s), avg. time (s), std time (s)\n")
            archivoResumenPercentage.write("instance, avg. XPL%, avg. XPT%\n")
            archivoFitness.write("MH, FITNESS\n")

            # Procesar resultados y escribir datos
            procesar_archivos(instancia_id, blob, archivoFitness, binarizacion, mhs_instances_local)

            # Escribir resúmenes estadísticos
            escribir_resumenes(mhs_instances_local, archivoResumenFitness, archivoResumenTimes, archivoResumenPercentage, MHS_LIST)

            # Generar gráficos resumen
            graficar_mejores_resultados(instancia_id, mhs_instances_local, binarizacion)
            graficar_boxplot_violin(instancia_id, binarizacion)

            # Cerrar archivos
            archivoResumenFitness.close()
            archivoResumenTimes.close()
            archivoResumenPercentage.close()
            archivoFitness.close()

        print("")  # Separación visual entre instancias

    print("[INFO] Análisis SCP completado con éxito.")
    print("-" * 50)
