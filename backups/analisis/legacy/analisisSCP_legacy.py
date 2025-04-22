import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import json

from Util import util
from BD.sqlite import BD
from Util.log import escribir_resumenes

CONFIG_FILE = './util/json/dir.json'
EXPERIMENTS_FILE = './util/json/experiments_config.json'

with open(CONFIG_FILE, 'r') as config_file:
    CONFIG = json.load(config_file)

with open(EXPERIMENTS_FILE, 'r') as experiments_file:
    EXPERIMENTS = json.load(experiments_file)

# Directorios
DIRS = CONFIG["dirs"]
DIR_RESULTADO = DIRS["base"]
DIR_TRANSITORIO = DIRS["transitorio"]
DIR_GRAFICOS = DIRS["graficos"]
DIR_BEST = DIRS["best"]
DIR_BOXPLOT = DIRS["boxplot"]
DIR_VIOLIN = DIRS["violinplot"]

GRAFICOS = True
MHS_LIST = EXPERIMENTS["mhs"]
COLORS = ['r', 'g']

# Clase para almacenar resultados
class InstancesMhs:
    def __init__(self):
        self.div, self.fitness, self.time = [], [], []
        self.xpl, self.xpt = [], []
        self.bestFitness, self.bestTime = [], []

# Inicializa las metaheurísticas
mhs_instances = {name: InstancesMhs() for name in MHS_LIST}

bd = BD()

def actualizar_datos(mhs_instances, mh, archivo_fitness, data):
    instancia_mh = mhs_instances[mh]
    instancia_mh.fitness.append(np.min(data['fitness']))
    instancia_mh.time.append(np.round(np.sum(data['time']), 3))
    instancia_mh.xpl.append(np.round(np.mean(data['XPL']), 2))
    instancia_mh.xpt.append(np.round(np.mean(data['XPT']), 2))
    
    archivo_fitness.write(f'{mh}, {np.min(data["fitness"])}\n')

def graficar_datos(iteraciones, fitness, xpl, xpt, tiempo, mh, problem, corrida):
    # Gráfico de convergencia
    _, ax = plt.subplots()
    ax.plot(iteraciones, fitness, marker='o')  # Usa un marcador para visualizar cada iteración
    ax.set_title(f'Convergence {mh} \n scp{problem} - Run {corrida} - ()')
    ax.set_ylabel("Fitness")
    ax.set_xlabel("Iteration")
    plt.savefig(f'{DIR_GRAFICOS}Convergence_{mh}_SCP_{problem}_{corrida}_.pdf')
    plt.close('all')

    # Gráfico XPL vs XPT
    _, axPER = plt.subplots()
    axPER.plot(iteraciones, xpl, color="r", label=r"$\overline{XPL}$" + ": " + str(np.round(np.mean(xpl), decimals=2)) + "%")
    axPER.plot(iteraciones, xpt, color="b", label=r"$\overline{XPT}$" + ": " + str(np.round(np.mean(xpt), decimals=2)) + "%")
    axPER.set_title(f'XPL% - XPT% {mh} \n scp{problem} Run {corrida} - ')
    axPER.set_ylabel("Percentage")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc='upper right')
    plt.savefig(f'{DIR_GRAFICOS}Percentage_{mh}_SCP_{problem}_{corrida}_.pdf')
    plt.close('all')
    
    # Gráfico de tiempo por iteración
    _, axTime = plt.subplots()
    axTime.plot(iteraciones, tiempo, color='g', label='Time per Iteration')
    axTime.set_title(f'Time per Iteration {mh} \n scp{problem} Run {corrida} -')
    axTime.set_ylabel("Time (s)")
    axTime.set_xlabel("Iteration")
    axTime.legend(loc='upper right')
    plt.savefig(f'{DIR_GRAFICOS}Time_{mh}_SCP_{problem}_{corrida}_.pdf')
    plt.close('all')


def graficar_boxplot_violin(instancia):
    direccion_datos = f'{DIR_RESULTADO}fitness_SCP_{instancia}_.csv'
    datos = pd.read_csv(direccion_datos)
    
    # Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='MH', y=' FITNESS', data=datos)
    plt.title(f'Boxplot Fitness \n scp{instancia} -')
    plt.savefig(f'{DIR_BOXPLOT}boxplot_fitness_SCP_{instancia}_.pdf')
    plt.close()
    
    # Violinplot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='MH', y=' FITNESS', data=datos)
    plt.title(f'Violinplot Fitness \n scp{instancia} - ')
    plt.savefig(f'{DIR_VIOLIN}violinplot_fitness_SCP_{instancia}_.pdf')
    plt.close()
    
# Procesar archivos de resultados
def procesar_archivos(instancia, blob, archivo_fitness):
    corrida = 1
    
    for nombre_archivo, contenido in blob:
        # Extraer la metaheurística y la instancia desde el nombre del archivo
        try:
            mh, _ = nombre_archivo.split('_')[:2] # Confirmar que el nombre del archivo tiene el formato esperado
        except ValueError:
            print(f"[ADVERTENCIA] El archivo '{nombre_archivo}' no tiene el formato esperado. Se omite.")
            continue

        # Procesar el archivo
        direccion_destino = f'{DIR_TRANSITORIO}{nombre_archivo}.csv'
        util.writeTofile(contenido, direccion_destino)

        try:
            data = pd.read_csv(direccion_destino)
        except Exception as e:
            print(f"[ERROR] No se pudo leer el archivo '{direccion_destino}': {e}")
            os.remove(direccion_destino)
            continue

        if mh in MHS_LIST:
            # Actualizar datos para la metaheurística
            actualizar_datos(mhs_instances, mh, archivo_fitness, data)
            mhs_instances[mh].bestFitness = data['fitness']
            mhs_instances[mh].bestTime = data['time']

        # Generar gráficos
        """if GRAFICOS:
            graficar_datos(data['iter'], data['fitness'], data['XPL'], data['XPT'], data['time'], mh, instancia, corrida)"""
        
        os.remove(direccion_destino)
        
        corrida += 1

    archivo_fitness.close()

def graficar_mejores_resultados(instancia, mhs_instances):
    mejor_fitness = float('inf')
    mejor_tiempo = float('inf')
    mh_mejor_fitness = ""
    mh_mejor_tiempo = ""

    for name in MHS_LIST:
        mh = mhs_instances[name]
        
        # Obtener el mejor fitness y el mejor tiempo
        min_fitness = min(mh.bestFitness)
        min_tiempo = min(mh.bestTime)
        
        if min_fitness < mejor_fitness:
            mejor_fitness = min_fitness
            mh_mejor_fitness = name
        
        if min_tiempo < mejor_tiempo:
            mejor_tiempo = min_tiempo
            mh_mejor_tiempo = name

    for name in MHS_LIST:
        mh = mhs_instances[name]
        
        # Gráfico de mejores fitness
        plt.plot(range(len(mh.bestFitness)), mh.bestFitness, color='r', label='Best Fitness')
        plt.title(f'Best Fitness \n scp{instancia} - Mejor: {mh_mejor_fitness}')
        plt.ylabel("Fitness")
        plt.xlabel("Iteration")
        plt.legend()
        plt.savefig(f'{DIR_BEST}fitness_SCP_{instancia}_.pdf')
        plt.close()

        # Gráfico de mejores tiempos
        plt.plot(range(len(mh.bestTime)), mh.bestTime, color='b', label='Best Time')
        plt.title(f'Best Time (s) \n scp{instancia} - Mejor: {mh_mejor_tiempo}')
        plt.ylabel("Time (s)")
        plt.xlabel("Iteration")
        plt.legend()
        plt.savefig(f'{DIR_BEST}time_SCP_{instancia}_.pdf')
        plt.close()

# Procesar cada instancia
lista_instancias = ', '.join([f'"{func}"' for func in EXPERIMENTS["instancias"]["SCP"]])
lista_bin = EXPERIMENTS["DS_actions"]

instancias = bd.obtenerInstancias(lista_instancias)

for instancia in instancias:
    print(f"Procesando instancia: scp{instancia[1]}")
    blob = bd.obtenerArchivos(instancia[1])
    
    #print (blob)

    if not blob:
        print(f"Advertencia: La instancia scp'{instancia[1]}' no tiene experimentos asociados. Saltando esta instancia...")
        continue

    # Crear archivos de salida
    archivoResumenFitness = open(f'{DIR_RESULTADO}resumen_fitness_SCP_{instancia[1]}_.csv', 'w')
    archivoResumenTimes = open(f'{DIR_RESULTADO}resumen_times_SCP_{instancia[1]}_.csv', 'w')
    archivoResumenPercentage = open(f'{DIR_RESULTADO}resumen_percentage_SCP_{instancia[1]}_.csv', 'w')
    archivoFitness = open(f'{DIR_RESULTADO}fitness_SCP_{instancia[1]}_.csv', 'w')

    # Escribir encabezados
    archivoResumenFitness.write("instance, best, avg. fitness, std fitness\n")
    archivoResumenTimes.write("instance, min time (s), avg. time (s), std time (s)\n")
    archivoResumenPercentage.write("instance, avg. XPL%, avg. XPT%\n")
    archivoFitness.write("MH, FITNESS\n")

    # Procesar archivos
    procesar_archivos(instancia[1], blob, archivoFitness)

    # Resumen
    escribir_resumenes(mhs_instances, archivoResumenFitness, archivoResumenTimes, archivoResumenPercentage, MHS_LIST)

    # Gráficos finales
    graficar_mejores_resultados(instancia[1], mhs_instances)
    graficar_boxplot_violin(instancia[1])

    # Cerrar archivos
    archivoResumenFitness.close()
    archivoResumenTimes.close()
    archivoResumenPercentage.close()
    
    print("")

print("Procesamiento completado con éxito.")