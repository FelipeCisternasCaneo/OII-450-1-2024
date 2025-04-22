import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import json

from Util import util
from BD.sqlite import BD

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
    fig, ax = plt.subplots()
    ax.plot(iteraciones, fitness)
    ax.set_title(f'Convergence {mh} \n {problem} run {corrida}')
    ax.set_ylabel("Fitness")
    ax.set_xlabel("Iteration")
    plt.savefig(f'{DIR_GRAFICOS}Convergence_{mh}_SCP_{problem}_{corrida}.pdf')
    plt.close('all')

    # Gráfico XPL vs XPT
    figPER, axPER = plt.subplots()
    axPER.plot(iteraciones, xpl, color="r", label=r"$\overline{XPL}$" + ": " + str(np.round(np.mean(xpl), decimals=2)) + "%")
    axPER.plot(iteraciones, xpt, color="b", label=r"$\overline{XPT}$" + ": " + str(np.round(np.mean(xpt), decimals=2)) + "%")
    axPER.set_title(f'XPL% - XPT% {mh} \n {problem} run {corrida}')
    axPER.set_ylabel("Percentage")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc='upper right')
    plt.savefig(f'{DIR_GRAFICOS}Percentage_{mh}_SCP_{problem}_{corrida}.pdf')
    plt.close('all')
    
    # Gráfico de tiempo por iteración
    figTime, axTime = plt.subplots()
    axTime.plot(iteraciones, tiempo, color='g', label='Time per Iteration')
    axTime.set_title(f'Time per Iteration {mh} \n {problem} run {corrida}')
    axTime.set_ylabel("Time (s)")
    axTime.set_xlabel("Iteration")
    axTime.legend(loc='upper right')
    plt.savefig(f'{DIR_GRAFICOS}Time_{mh}_SCP_{problem}_{corrida}.pdf')
    plt.close('all')

def graficar_boxplot_violin(instancia):
    direccion_datos = f'{DIR_RESULTADO}fitness_SCP_{instancia}.csv'
    datos = pd.read_csv(direccion_datos)
    
    # Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='MH', y=' FITNESS', data=datos)
    plt.title(f'Boxplot Fitness {instancia}')
    plt.savefig(f'{DIR_BOXPLOT}boxplot_fitness_SCP_{instancia}.pdf')
    plt.close()
    
    # Violinplot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='MH', y=' FITNESS', data=datos)
    plt.title(f'Violinplot Fitness {instancia}')
    plt.savefig(f'{DIR_VIOLIN}violinplot_fitness_SCP_{instancia}.pdf')
    plt.close()
    
def escribir_resumenes(mhs_instances, archivoResumenFitness, archivoResumenTimes, archivoResumenPercentage):
    for name in MHS_LIST:
        mh = mhs_instances[name]
        archivoResumenFitness.write(f"{name}, {np.min(mh.fitness)}, {np.round(np.mean(mh.fitness), 3)}, {np.round(np.std(mh.fitness), 3)}\n")
        archivoResumenTimes.write(f"{name}, {np.min(mh.time)}, {np.round(np.mean(mh.time), 3)}, {np.round(np.std(mh.time), 3)}\n")
        archivoResumenPercentage.write(f"{name}, {np.round(np.mean(mh.xpl), 3)}, {np.round(np.mean(mh.xpt), 3)}\n")

# Procesar archivos de resultados
def procesar_archivos(instancia, blob, archivo_fitness):
    corrida = 1
    for nombre_archivo, contenido in blob:
        direccion_destino = f'{DIR_TRANSITORIO}{nombre_archivo}.csv'
        util.writeTofile(contenido, direccion_destino)

        # Leer archivo
        data = pd.read_csv(direccion_destino)
        mh, problem = nombre_archivo.split('_')[0], instancia

        # Actualizar datos
        if mh in MHS_LIST:
            actualizar_datos(mhs_instances, mh, archivo_fitness, data)
            mhs_instances[mh].bestFitness = data['fitness']
            mhs_instances[mh].bestTime = data['time']

        # Generar gráficos
        if GRAFICOS:
            graficar_datos(data['iter'], data['fitness'], data['XPL'], data['XPT'], data['time'], mh, problem, corrida)

        os.remove(direccion_destino)
        corrida += 1
        
    archivoFitness.close()

def graficar_mejores_resultados(instancia, mhs_instances):
    for name in MHS_LIST:
        mh = mhs_instances[name]
        
        # Gráfico de mejores fitness
        plt.plot(range(len(mh.bestFitness)), mh.bestFitness, color = 'r', label = 'Best Fitness')
        plt.title(f'Best Fitness \n {instancia}')
        plt.ylabel("Fitness")
        plt.xlabel("Iteration")
        plt.legend()
        plt.savefig(f'{DIR_BEST}fitness_SCP_{instancia}.pdf')
        plt.close()

        # Gráfico de mejores tiempos
        plt.plot(range(len(mh.bestTime)), mh.bestTime, color = 'b', label = 'Best Time')
        plt.title(f'Time (s) \n {instancia}')
        plt.ylabel("Time (s)")
        plt.xlabel("Iteration")
        plt.legend()
        plt.savefig(f'{DIR_BEST}time_SCP_{instancia}.pdf')
        plt.close()

lista_instancias = ', '.join([f'"{func}"' for func in EXPERIMENTS["instancias"]["SCP"]])

# Procesar cada instancia
instancias = bd.obtenerInstancias(lista_instancias)

for instancia in instancias:
    print(f"Procesando instancia: {instancia[1]}")
    blob = bd.obtenerArchivos(instancia[1])

    # Validar si hay experimentos para la instancia
    if not blob:
        print(f"Advertencia: La instancia '{instancia[1]}' no tiene experimentos asociados. Saltando esta instancia...")
        
        continue

    # Crear archivos de salida
    archivoResumenFitness = open(f'{DIR_RESULTADO}resumen_fitness_SCP_{instancia[1]}.csv', 'w')
    archivoResumenTimes = open(f'{DIR_RESULTADO}resumen_times_SCP_{instancia[1]}.csv', 'w')
    archivoResumenPercentage = open(f'{DIR_RESULTADO}resumen_percentage_SCP_{instancia[1]}.csv', 'w')
    archivoFitness = open(f'{DIR_RESULTADO}fitness_SCP_{instancia[1]}.csv', 'w')

    # Escribir encabezados
    archivoResumenFitness.write("instance, best, avg. fitness, std fitness\n")
    archivoResumenTimes.write("instance, min time (s), avg. time (s), std time (s)\n")
    archivoResumenPercentage.write("instance, avg. XPL%, avg. XPT%\n")
    archivoFitness.write("MH, FITNESS\n")

    # Procesar archivos
    procesar_archivos(instancia[1], blob, archivoFitness)

    # Resumen
    escribir_resumenes(mhs_instances, archivoResumenFitness, archivoResumenTimes, archivoResumenPercentage)

    # Gráficos finales
    graficar_mejores_resultados(instancia[1], mhs_instances)
    
    graficar_boxplot_violin(instancia[1])

    # Cerrar archivos
    archivoResumenFitness.close()
    archivoResumenTimes.close()
    archivoResumenPercentage.close()

print("Procesamiento completado con éxito.")
