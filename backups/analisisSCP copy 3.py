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

def actualizar_datos(mhs_instances, mh, archivo_fitness, data, initial_fitness=None):
    instancia_mh = mhs_instances[mh]
    
    if initial_fitness is not None:
        instancia_mh.fitness.append(initial_fitness)  # Agregar el fitness inicial como iteración 0

    instancia_mh.fitness.extend(data['fitness'])
    instancia_mh.time.extend(data['time'])
    instancia_mh.xpl.extend(data['XPL'])
    instancia_mh.xpt.extend(data['XPT'])

    archivo_fitness.write(f'{mh}, {np.min(data["fitness"])}\n')

def graficar_datos(iteraciones, fitness, xpl, xpt, tiempo, mh, problem, corrida, binarizacion):
    # Asegurarse de que las listas tengan la misma longitud
    min_len = min(len(iteraciones), len(fitness), len(xpl), len(xpt), len(tiempo))
    iteraciones = iteraciones[:min_len]
    fitness = fitness[:min_len]
    xpl = xpl[:min_len]
    xpt = xpt[:min_len]
    tiempo = tiempo[:min_len]

    # Depuración: Imprimir las dimensiones para confirmar
    print(f"DEBUG: iteraciones={len(iteraciones)}, fitness={len(fitness)}, xpl={len(xpl)}, xpt={len(xpt)}, tiempo={len(tiempo)}")

    # Gráfico de convergencia
    fig, ax = plt.subplots()
    ax.plot(iteraciones, fitness, marker='o')  # Usa un marcador para visualizar cada iteración
    ax.set_title(f'Convergence {mh} \n {problem} run {corrida} ({binarizacion})')
    ax.set_ylabel("Fitness")
    ax.set_xlabel("Iteration")
    plt.savefig(f'{DIR_GRAFICOS}Convergence_{mh}_SCP_{problem}_{corrida}_{binarizacion}.pdf')
    plt.close('all')

    # Gráfico XPL vs XPT
    figPER, axPER = plt.subplots()
    axPER.plot(iteraciones, xpl, color="r", label=r"$\overline{XPL}$" + ": " + str(np.round(np.mean(xpl), decimals=2)) + "%")
    axPER.plot(iteraciones, xpt, color="b", label=r"$\overline{XPT}$" + ": " + str(np.round(np.mean(xpt), decimals=2)) + "%")
    axPER.set_title(f'XPL% - XPT% {mh} \n {problem} run {corrida} ({binarizacion})')
    axPER.set_ylabel("Percentage")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc='upper right')
    plt.savefig(f'{DIR_GRAFICOS}Percentage_{mh}_SCP_{problem}_{corrida}_{binarizacion}.pdf')
    plt.close('all')
    
    # Gráfico de tiempo por iteración
    figTime, axTime = plt.subplots()
    axTime.plot(iteraciones, tiempo, color='g', label='Time per Iteration')
    axTime.set_title(f'Time per Iteration {mh} \n {problem} run {corrida} ({binarizacion})')
    axTime.set_ylabel("Time (s)")
    axTime.set_xlabel("Iteration")
    axTime.legend(loc='upper right')
    plt.savefig(f'{DIR_GRAFICOS}Time_{mh}_SCP_{problem}_{corrida}_{binarizacion}.pdf')
    plt.close('all')




def procesar_archivos(instancia, blob, archivo_fitness, bin_actual):
    corrida = 1

    for nombre_archivo, contenido, binarizacion in blob:
        try:
            mh, _ = nombre_archivo.split('_')[:2]
        except ValueError:
            print(f"[ADVERTENCIA] El archivo '{nombre_archivo}' no tiene el formato esperado. Se omite.")
            continue

        if binarizacion.strip() != bin_actual:
            continue

        direccion_destino = f'{DIR_TRANSITORIO}{nombre_archivo}.csv'
        util.writeTofile(contenido, direccion_destino)

        try:
            data = pd.read_csv(direccion_destino)
        except Exception as e:
            print(f"[ERROR] No se pudo leer el archivo '{direccion_destino}': {e}")
            os.remove(direccion_destino)
            continue

        if mh in MHS_LIST:
            initial_fitness = data['fitness'][0]  # Fitness inicial como iteración 0
            actualizar_datos(mhs_instances, mh, archivo_fitness, data, initial_fitness)

        if GRAFICOS:
            iteraciones = [0] + list(data['iter'].unique())
            graficar_datos(iteraciones, mhs_instances[mh].fitness, mhs_instances[mh].xpl, mhs_instances[mh].xpt, mhs_instances[mh].time, mh, instancia, corrida, binarizacion)
        
        os.remove(direccion_destino)
        corrida += 1

    archivo_fitness.close()

# Procesar cada instancia
lista_instancias = ', '.join([f'"{func}"' for func in EXPERIMENTS["instancias"]["SCP"]])
lista_bin = EXPERIMENTS["DS_actions"]

instancias = bd.obtenerInstancias(lista_instancias)

for instancia in instancias:
    print(f"Procesando instancia: scp{instancia[1]}")
    blob = bd.obtenerArchivos(instancia[1])

    if not blob:
        print(f"Advertencia: La instancia scp'{instancia[1]}' no tiene experimentos asociados. Saltando esta instancia...")
        continue

    for binarizacion in lista_bin:
        print(f"Procesando binarización: {binarizacion}")

        archivoResumenFitness = open(f'{DIR_RESULTADO}resumen_fitness_SCP_{instancia[1]}_{binarizacion}.csv', 'w')
        archivoResumenTimes = open(f'{DIR_RESULTADO}resumen_times_SCP_{instancia[1]}_{binarizacion}.csv', 'w')
        archivoResumenPercentage = open(f'{DIR_RESULTADO}resumen_percentage_SCP_{instancia[1]}_{binarizacion}.csv', 'w')
        archivoFitness = open(f'{DIR_RESULTADO}fitness_SCP_{instancia[1]}_{binarizacion}.csv', 'w')

        archivoResumenFitness.write("instance, best, avg. fitness, std fitness\n")
        archivoResumenTimes.write("instance, min time (s), avg. time (s), std time (s)\n")
        archivoResumenPercentage.write("instance, avg. XPL%, avg. XPT%\n")
        archivoFitness.write("MH, FITNESS\n")

        procesar_archivos(instancia[1], blob, archivoFitness, binarizacion)

        escribir_resumenes(mhs_instances, archivoResumenFitness, archivoResumenTimes, archivoResumenPercentage, MHS_LIST)

        archivoResumenFitness.close()
        archivoResumenTimes.close()
        archivoResumenPercentage.close()

    print("")

print("Procesamiento completado con éxito.")
