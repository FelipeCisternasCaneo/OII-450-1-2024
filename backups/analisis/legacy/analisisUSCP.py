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
GRAFICOS = False  # True si quieres gráficos por corrida
MHS_LIST = EXPERIMENTS["mhs"]

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
        # NEW: diversity & gaps
        self.ent = []
        self.divj_mean = []
        self.divj_min = []
        self.divj_max = []
        self.gap = []
        self.rdp = []
        # NEW: series representativas por iteración (para gráficos "best")
        self.xpl_iter = None
        self.xpt_iter = None

bd = BD()

# === Función para actualizar Datos ===
def actualizar_datos(mhs_instances, mh, archivo_fitness, data):
    inst = mhs_instances[mh]

    # Core summaries
    inst.fitness.append(np.min(data['best_fitness']))
    inst.time.append(np.round(np.sum(data['time']), 3))
    inst.xpl.append(np.round(np.mean(data['XPL']), 2))
    inst.xpt.append(np.round(np.mean(data['XPT']), 2))
    archivo_fitness.write(f'{mh}, {np.min(data["best_fitness"])}\n')

    # Diversity & gaps (promedios por corrida)
    inst.ent.append(float(np.nanmean(data['ENT'])))
    inst.divj_mean.append(float(np.nanmean(data['Divj_mean'])))
    inst.divj_min.append(float(np.nanmean(data['Divj_min'])))
    inst.divj_max.append(float(np.nanmean(data['Divj_max'])))
    inst.gap.append(float(np.nanmean(data['GAP'])))
    inst.rdp.append(float(np.nanmean(data['RDP'])))

def graficar_datos(iteraciones, fitness, xpl, xpt, tiempo, mh, problem, corrida, binarizacion):
    """ Gráficos por corrida (opcional). """
    output_dir = os.path.join(DIR_GRAFICOS, 'USCP', str(binarizacion))
    os.makedirs(output_dir, exist_ok=True)

    # Convergencia
    path_convergencia = os.path.join(output_dir, f'Convergence_{mh}_USCP_{problem}_{corrida}_{binarizacion}.pdf')
    _, ax = plt.subplots()
    ax.plot(iteraciones, fitness)
    ax.set_title(f'Convergence {mh}\n uscp{problem} - Run {corrida} - ({binarizacion})')
    ax.set_ylabel("Fitness")
    ax.set_xlabel("Iteration")
    plt.tight_layout()
    plt.savefig(path_convergencia)
    plt.close()

    # XPL vs XPT
    path_porcentaje = os.path.join(output_dir, f'Percentage_{mh}_USCP_{problem}_{corrida}_{binarizacion}.pdf')
    _, axPER = plt.subplots()
    axPER.plot(iteraciones, xpl, color="r", label=rf"$\overline{{XPL}}$: {np.round(np.mean(xpl), 2)}%")
    axPER.plot(iteraciones, xpt, color="b", label=rf"$\overline{{XPT}}$: {np.round(np.mean(xpt), 2)}%")
    axPER.set_title(f'XPL% - XPT% {mh}\n uscp{problem} - Run {corrida} - ({binarizacion})')
    axPER.set_ylabel("Percentage")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path_porcentaje)
    plt.close()

    # Tiempo por iteración
    path_tiempo = os.path.join(output_dir, f'Time_{mh}_USCP_{problem}_{corrida}_{binarizacion}.pdf')
    _, axTime = plt.subplots()
    axTime.plot(iteraciones, tiempo, color='g', label='Time per Iteration')
    axTime.set_title(f'Time per Iteration {mh}\n uscp{problem} - Run {corrida} - ({binarizacion})')
    axTime.set_ylabel("Time (s)")
    axTime.set_xlabel("Iteration")
    axTime.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path_tiempo)
    plt.close()

def graficar_boxplot_violin(instancia, binarizacion):
    """ Boxplot y Violinplot por instancia (USCP). """
    direccion_datos = os.path.join(DIR_FITNESS, f'USCP/fitness_USCP_{instancia}_{binarizacion}.csv')

    try:
        datos = pd.read_csv(direccion_datos)
        datos.columns = datos.columns.str.strip()
        if 'FITNESS' not in datos.columns or 'MH' not in datos.columns:
            print(f"[ERROR] Columnas necesarias no encontradas en {direccion_datos}")
            return
    except FileNotFoundError:
        print(f"[ERROR] Archivo no encontrado: {direccion_datos}")
        return
    except Exception as e:
        print(f"[ERROR] Error al leer {direccion_datos}: {e}")
        return

    # Boxplot
    output_dir_box = os.path.join(DIR_BOXPLOT, 'USCP')
    os.makedirs(output_dir_box, exist_ok=True)
    file_path_box = os.path.join(output_dir_box, f'boxplot_fitness_USCP_{instancia}_{binarizacion}.pdf')

    sns.boxplot(x='MH', y='FITNESS', data=datos, hue='MH', palette='Set2', legend=False)
    plt.title(f'Boxplot Fitness\n uscp{instancia} - {binarizacion}')
    plt.xlabel('Metaheurística')
    plt.ylabel('Fitness')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(file_path_box)
    plt.close()

    # Violinplot
    output_dir_violin = os.path.join(DIR_VIOLIN, 'USCP')
    os.makedirs(output_dir_violin, exist_ok=True)
    file_path_violin = os.path.join(output_dir_violin, f'violinplot_fitness_USCP_{instancia}_{binarizacion}.pdf')

    sns.violinplot(x='MH', y='FITNESS', data=datos, hue='MH', palette='Set3', legend=False)
    plt.title(f'Violinplot Fitness\n uscp{instancia} - {binarizacion}')
    plt.xlabel('Metaheurística')
    plt.ylabel('Fitness')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(file_path_violin)
    plt.close()

def procesar_archivos(instancia, blob, archivo_fitness, bin_actual, mhs_instances):
    """
    Procesa archivos (nombre, contenido, binarizacion) de una instancia USCP y
    actualiza estructuras + gráficos por corrida (si GRAFICOS=True).
    """
    corrida = 1

    for nombre_archivo, contenido, binarizacion in blob:
        # nombre archivo => mh_...
        try:
            mh, _ = nombre_archivo.split('_')[:2]
        except ValueError:
            print(f"[ADVERTENCIA] Archivo '{nombre_archivo}' con nombre inválido. Se omite.")
            continue

        # filtrar por binarización
        if binarizacion.strip() != bin_actual:
            continue

        # guardar y leer temporal
        direccion_destino = os.path.join(DIR_TRANSITORIO, f'{nombre_archivo}.csv')
        writeTofile(contenido, direccion_destino)

        try:
            data = pd.read_csv(direccion_destino)
        except Exception as e:
            print(f"[ERROR] Fallo al leer '{direccion_destino}': {e}")
            os.remove(direccion_destino)
            continue

        if mh in MHS_LIST:
            actualizar_datos(mhs_instances, mh, archivo_fitness, data)

            # Elegir corrida representativa por MH: la que termina con mejor best_fitness
            new_final = float(data['best_fitness'].iloc[-1])

            prev_final = np.inf
            prev_series = getattr(mhs_instances[mh], 'bestFitness', None)
            if prev_series is not None:
                try:
                    if len(prev_series) > 0:
                        prev_final = float(prev_series.iloc[-1])  # pandas Series
                except AttributeError:
                    if len(prev_series) > 0:
                        prev_final = float(prev_series[-1])       # list/ndarray

            if new_final < prev_final:
                mhs_instances[mh].bestFitness = data['best_fitness']
                mhs_instances[mh].bestTime    = data['time']
                mhs_instances[mh].xpl_iter    = data['XPL']
                mhs_instances[mh].xpt_iter    = data['XPT']

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
                binarizacion=binarizacion
            )

        os.remove(direccion_destino)
        corrida += 1

    archivo_fitness.close()

def graficar_mejores_resultados(instancia, mhs_instances, binarizacion):
    """
    Gráficos comparativos: mejor curva de fitness, mejor curva de tiempo y
    gráfico combinado XPL/XPT por MH (corrida representativa).
    """
    mejor_fitness = float('inf')
    mejor_tiempo = float('inf')
    mh_mejor_fitness = ""
    mh_mejor_tiempo = ""

    for name in MHS_LIST:
        mh = mhs_instances[name]
        min_fitness = min(mh.bestFitness)
        min_tiempo  = min(mh.bestTime)
        if min_fitness < mejor_fitness:
            mejor_fitness = min_fitness
            mh_mejor_fitness = name
        if min_tiempo < mejor_tiempo:
            mejor_tiempo = min_tiempo
            mh_mejor_tiempo = name

    output_dir = os.path.join(DIR_BEST, 'USCP')
    os.makedirs(output_dir, exist_ok=True)

    # Fitness comparativo
    for name in MHS_LIST:
        mh = mhs_instances[name]
        plt.plot(range(len(mh.bestFitness)), mh.bestFitness, label=name)
    plt.title(f'Best Fitness per MH \n uscp{instancia} - {binarizacion}\nBest: {mh_mejor_fitness} ({mejor_fitness})')
    plt.ylabel("Fitness")
    plt.xlabel("Iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fitness_USCP_{instancia}_{binarizacion}.pdf'))
    plt.close()

    # Tiempo comparativo
    for name in MHS_LIST:
        mh = mhs_instances[name]
        plt.plot(range(len(mh.bestTime)), mh.bestTime, label=name)
    plt.title(f'Best Time per MH \n uscp{instancia} - {binarizacion}\nBest: {mh_mejor_tiempo} ({mejor_tiempo:.2f} s)')
    plt.ylabel("Time (s)")
    plt.xlabel("Iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'time_USCP_{instancia}_{binarizacion}.pdf'))
    plt.close()

    # XPL/XPT combinado comparativo
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
        plt.plot(x_idx, mh.xpt_iter, linestyle='-',  linewidth=2,
                 label=f'{name} XPT% (avg {np.round(np.mean(mh.xpt_iter), 2)}%)',
                 color=color)
        plt.plot(x_idx, mh.xpl_iter, linestyle='--', linewidth=2,
                 label=f'{name} XPL% (avg {np.round(np.mean(mh.xpl_iter), 2)}%)',
                 color=color)
        max_len = max(max_len, len(x_idx))

    if any_series:
        plt.title(f'Exploration (XPL) vs Exploitation (XPT) per MH\nuscp{instancia} - {binarizacion}')
        plt.ylabel("Percentage (%)")
        plt.xlabel("Iteration")
        plt.ylim(0, 100)
        if max_len <= 1:
            plt.xlim(-0.5, 0.5)
        plt.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'xpl_xpt_USCP_{instancia}_{binarizacion}.pdf'))
        plt.close()
    else:
        plt.close()

def analizar_instancias():
    """ Orquestador principal para USCP. """
    os.makedirs(DIR_TRANSITORIO, exist_ok=True)

    lista_instancias = ', '.join([f'"{func}"' for func in EXPERIMENTS["instancias"]["USCP"]])
    lista_bin = EXPERIMENTS["DS_actions"]
    instancias = bd.obtenerInstancias(lista_instancias)

    print("Iniciando procesamiento de instancias...\n")

    for instancia in instancias:
        instancia_id = instancia[1]
        print(f"[INFO] Procesando instancia: uscp{instancia_id}")
        blob = bd.obtenerArchivos(instancia_id)

        if not blob:
            print(f"[ADVERTENCIA] La instancia uscp{instancia_id} no tiene experimentos asociados. Saltando...\n")
            continue

        for binarizacion in lista_bin:
            print(f"    -- Binarización: {binarizacion}")
            mhs_instances_local = {name: InstancesMhs() for name in MHS_LIST}

            # carpetas
            output_dir_resumen = os.path.join(DIR_RESUMEN, 'USCP')
            output_dir_fitness = os.path.join(DIR_FITNESS, 'USCP')
            os.makedirs(output_dir_resumen, exist_ok=True)
            os.makedirs(output_dir_fitness, exist_ok=True)

            # archivos de salida
            archivoResumenFitness   = open(os.path.join(output_dir_resumen, f'resumen_fitness_USCP_{instancia_id}_{binarizacion}.csv'), 'w')
            archivoResumenTimes     = open(os.path.join(output_dir_resumen, f'resumen_times_USCP_{instancia_id}_{binarizacion}.csv'), 'w')
            archivoResumenPercentage= open(os.path.join(output_dir_resumen, f'resumen_percentage_USCP_{instancia_id}_{binarizacion}.csv'), 'w')
            archivoResumenDiversity = open(os.path.join(output_dir_resumen, f'resumen_diversity_USCP_{instancia_id}_{binarizacion}.csv'), 'w')
            archivoResumenGap       = open(os.path.join(output_dir_resumen, f'resumen_gap_USCP_{instancia_id}_{binarizacion}.csv'), 'w')
            archivoFitness          = open(os.path.join(output_dir_fitness, f'fitness_USCP_{instancia_id}_{binarizacion}.csv'), 'w')

            # headers
            archivoResumenFitness.write("MH, min best, avg. best, std. best\n")
            archivoResumenTimes.write("MH, min time (s), avg. time (s), std time (s)\n")
            archivoResumenPercentage.write("MH, avg. XPL%, avg. XPT%\n")
            archivoResumenDiversity.write("MH, avg. ENT, avg. Divj_mean, avg. Divj_min, avg. Divj_max\n")
            archivoResumenGap.write("MH, avg. GAP, avg. RDP\n")
            archivoFitness.write("MH, FITNESS\n")

            # procesar
            procesar_archivos(instancia_id, blob, archivoFitness, binarizacion, mhs_instances_local)

            # resúmenes core
            escribir_resumenes(mhs_instances_local, archivoResumenFitness, archivoResumenTimes, archivoResumenPercentage, MHS_LIST)

            # gráficos comparativos
            graficar_mejores_resultados(instancia_id, mhs_instances_local, binarizacion)
            graficar_boxplot_violin(instancia_id, binarizacion)

            # resúmenes de diversidad y GAP/RDP
            for name in MHS_LIST:
                mh = mhs_instances_local[name]

                if len(mh.ent) or len(mh.divj_mean) or len(mh.divj_min) or len(mh.divj_max):
                    ent_avg       = np.round(np.mean(mh.ent), 6)        if len(mh.ent)       else np.nan
                    divj_mean_avg = np.round(np.mean(mh.divj_mean), 6)  if len(mh.divj_mean) else np.nan
                    divj_min_avg  = np.round(np.mean(mh.divj_min), 6)   if len(mh.divj_min)  else np.nan
                    divj_max_avg  = np.round(np.mean(mh.divj_max), 6)   if len(mh.divj_max)  else np.nan
                    archivoResumenDiversity.write(
                        f"{name}, {ent_avg}, {divj_mean_avg}, {divj_min_avg}, {divj_max_avg}\n"
                    )

                if len(mh.gap) or len(mh.rdp):
                    gap_avg = np.round(np.mean(mh.gap), 6) if len(mh.gap) else np.nan
                    rdp_avg = np.round(np.mean(mh.rdp), 6) if len(mh.rdp) else np.nan
                    archivoResumenGap.write(f"{name}, {gap_avg}, {rdp_avg}\n")

            # cerrar
            archivoResumenFitness.close()
            archivoResumenTimes.close()
            archivoResumenPercentage.close()
            archivoResumenDiversity.close()
            archivoResumenGap.close()
            archivoFitness.close()

        print("")

    print("[INFO] Análisis USCP completado con éxito.")
    print("-" * 50)
