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
        self.ent = []
        self.divj_mean = []
        self.divj_min = []
        self.divj_max = []
        self.gap = []
        self.rdp = []
        # NEW: per-iteration series for best-style plots
        self.xpl_iter = None
        self.xpt_iter = None

# === Inicialización ===
#mhs_instances = {name: InstancesMhs() for name in MHS_LIST}
bd = BD()

# === Función para actualizar Datos ===
def actualizar_datos(mhs_instances, mh, archivo_fitness, data):
    instancia = mhs_instances[mh]

    # Core summaries
    instancia.fitness.append(np.min(data['best_fitness']))
    instancia.time.append(np.round(np.sum(data['time']), 3))
    instancia.xpl.append(np.round(np.mean(data['XPL']), 2))
    instancia.xpt.append(np.round(np.mean(data['XPT']), 2))
    archivo_fitness.write(f'{mh}, {np.min(data["best_fitness"])}\n')

    # --- NEW: diversity & gaps summaries per run (averages across iterations)
    # Use nanmean in case some columns contain NaN
    instancia.ent.append(float(np.nanmean(data['ENT'])))
    instancia.divj_mean.append(float(np.nanmean(data['Divj_mean'])))
    instancia.divj_min.append(float(np.nanmean(data['Divj_min'])))
    instancia.divj_max.append(float(np.nanmean(data['Divj_max'])))
    instancia.gap.append(float(np.nanmean(data['GAP'])))
    instancia.rdp.append(float(np.nanmean(data['RDP'])))

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
    ax.plot(iteraciones, fitness)
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

            # Decide which run to keep for "best-style" plots: keep the run with the
            # lowest final best_fitness value.
            new_final = float(data['best_fitness'].iloc[-1])

            prev_final = np.inf
            prev_series = getattr(mhs_instances[mh], 'bestFitness', None)
            if prev_series is not None:
                try:
                    # If it's a pandas Series
                    if len(prev_series) > 0:
                        prev_final = float(prev_series.iloc[-1])
                except AttributeError:
                    # If it's a list/ndarray
                    if len(prev_series) > 0:
                        prev_final = float(prev_series[-1])

            if new_final < prev_final:
                # Keep this run as the representative "best" for plots
                mhs_instances[mh].bestFitness = data['best_fitness']
                mhs_instances[mh].bestTime    = data['time']
                mhs_instances[mh].xpl_iter    = data['XPL']
                mhs_instances[mh].xpt_iter    = data['XPT']

        # Generar gráficos por corrida
        if GRAFICOS:
            graficar_datos(
                iteraciones=data['iter'],
                fitness=data['best_fitness'],
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
    Genera gráficos de comparación para los mejores valores de fitness, tiempo y
    promedios de exploración/explotación (XPL/XPT) alcanzados por cada metaheurística
    en una instancia.
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

    # === Gráfico comparativo de Fitness ===
    for name in MHS_LIST:
        mh = mhs_instances[name]
        plt.plot(range(len(mh.bestFitness)), mh.bestFitness, label=name)
    
    plt.title(f'Best Fitness per MH \n {instancia}\nBest: {mh_mejor_fitness} ({mejor_fitness})')
    plt.ylabel("Fitness")
    plt.xlabel("Iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fitness_BEN_{instancia}.pdf'))
    plt.close()

    # === Gráfico comparativo de Tiempos ===
    for name in MHS_LIST:
        mh = mhs_instances[name]
        plt.plot(range(len(mh.bestTime)), mh.bestTime, label=name)
    plt.title(f'Best Time per MH \n {instancia}\nBest: {mh_mejor_tiempo} ({mejor_tiempo:.2f} s)')
    plt.ylabel("Time (s)")
    plt.xlabel("Iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'time_BEN_{instancia}.pdf'))
    plt.close()

        # === Combined best-style plot: XPL% and XPT% per MH over iterations ===
    output_dir = os.path.join(DIR_BEST, 'BEN')
    os.makedirs(output_dir, exist_ok=True)

    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple",
              "tab:cyan", "tab:pink", "tab:brown", "tab:olive", "tab:gray"]
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(MHS_LIST)}

    plt.figure(figsize=(8, 5))
    any_series = False
    max_len = 0

    for name in MHS_LIST:
        mh = mhs_instances[name]
        if mh.xpl_iter is None or mh.xpt_iter is None:
            continue

        any_series = True
        color = color_map[name]
        x_idx = np.arange(len(mh.xpl_iter))

        # Solid = XPT, Dashed = XPL (same color per MH, sin puntos)
        plt.plot(x_idx, mh.xpt_iter, linestyle='-',  linewidth=2,
                 label=f'{name} XPT% (avg {np.round(np.mean(mh.xpt_iter), 2)}%)',
                 color=color)
        plt.plot(x_idx, mh.xpl_iter, linestyle='--', linewidth=2,
                 label=f'{name} XPL% (avg {np.round(np.mean(mh.xpl_iter), 2)}%)',
                 color=color)

        max_len = max(max_len, len(x_idx))

    if any_series:
        plt.title(f'Exploration (XPL) vs Exploitation (XPT) per MH\n{instancia}')
        plt.ylabel("Percentage (%)")
        plt.xlabel("Iteration")
        plt.ylim(0, 100)
        if max_len <= 1:
            plt.xlim(-0.5, 0.5)
        plt.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'xpl_xpt_BEN_{instancia}.pdf'))
        plt.close()
    else:
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
        archivoResumenDiversity = open(os.path.join(output_dir_resumen, f'resumen_diversity_BEN_{instancia_id}.csv'), 'w')
        archivoResumenGap = open(os.path.join(output_dir_resumen, f'resumen_gap_BEN_{instancia_id}.csv'), 'w')

        # Escribir encabezados
        archivoResumenFitness.write("MH, min best, avg. best, std. best\n")
        archivoResumenTimes.write("MH, min time (s), avg. time (s), std time (s)\n")
        archivoResumenPercentage.write("MH, avg. XPL%, avg. XPT%\n")
        archivoFitness.write("MH, FITNESS\n")
        archivoResumenDiversity.write("MH, avg. ENT, avg. Divj_mean, avg. Divj_min, avg. Divj_max\n")
        archivoResumenGap.write("MH, avg. GAP, avg. RDP\n")

        # Procesar resultados y escribir datos
        procesar_archivos(instancia_id, blob, archivoFitness, mhs_instances_local)

        # Escribir resúmenes estadísticos
        escribir_resumenes(mhs_instances_local, archivoResumenFitness, archivoResumenTimes, archivoResumenPercentage, MHS_LIST)

        # Generar gráficos resumen
        graficar_mejores_resultados(instancia_id, mhs_instances_local)
        graficar_boxplot_violin(instancia_id)
        
        # --- NEW: write diversity summary (ENT & Divj) and GAP/RDP summary ---
        for name in MHS_LIST:
            mh = mhs_instances_local[name]

            # Safety: if a MH has no data (shouldn't happen, but just in case)
            if len(mh.ent) == 0:
                continue

            # Diversity summary
            ent_avg       = np.round(np.mean(mh.ent), 6)
            divj_mean_avg = np.round(np.mean(mh.divj_mean), 6)
            divj_min_avg  = np.round(np.mean(mh.divj_min), 6)
            divj_max_avg  = np.round(np.mean(mh.divj_max), 6)
            archivoResumenDiversity.write(
                f"{name}, {ent_avg}, {divj_mean_avg}, {divj_min_avg}, {divj_max_avg}\n"
            )

            # GAP/RDP summary
            gap_avg = np.round(np.mean(mh.gap), 6)
            rdp_avg = np.round(np.mean(mh.rdp), 6)
            archivoResumenGap.write(f"{name}, {gap_avg}, {rdp_avg}\n")

        # Cerrar archivos
        archivoResumenFitness.close()
        archivoResumenTimes.close()
        archivoResumenPercentage.close()
        archivoFitness.close()
        archivoResumenDiversity.close()
        archivoResumenGap.close()

        print("")  # Separación visual entre instancias

    print("[INFO] Análisis BEN completado con éxito.")
    print("-" * 50)
