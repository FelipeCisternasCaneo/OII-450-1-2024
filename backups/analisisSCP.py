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

def graficar_datos(iteraciones, fitness, xpl, xpt, tiempo, mh, problem, corrida, binarizacion):
    # Gráfico de convergencia
    fig, ax = plt.subplots()
    ax.plot(iteraciones, fitness)
    ax.set_title(f'Convergencia {mh} \n {problem} ({binarizacion}) corrida {corrida}')
    ax.set_ylabel("Fitness")
    ax.set_xlabel("Iteración")
    plt.savefig(f'{DIR_GRAFICOS}Convergence_{mh}_SCP_{problem}_{binarizacion}_{corrida}.pdf')
    plt.close('all')

    # Gráfico XPL vs XPT
    figPER, axPER = plt.subplots()
    axPER.plot(iteraciones, xpl, color="r", label=r"$\overline{XPL}$" + ": " + str(np.round(np.mean(xpl), decimals=2)) + "%")
    axPER.plot(iteraciones, xpt, color="b", label=r"$\overline{XPT}$" + ": " + str(np.round(np.mean(xpt), decimals=2)) + "%")
    axPER.set_title(f'XPL% - XPT% {mh} \n {problem} ({binarizacion}) corrida {corrida}')
    axPER.set_ylabel("Porcentaje")
    axPER.set_xlabel("Iteración")
    axPER.legend(loc='upper right')
    plt.savefig(f'{DIR_GRAFICOS}Percentage_{mh}_SCP_{problem}_{binarizacion}_{corrida}.pdf')
    plt.close('all')
    
    # Gráfico de tiempo por iteración
    figTime, axTime = plt.subplots()
    axTime.plot(iteraciones, tiempo, color='g', label='Tiempo por Iteración')
    axTime.set_title(f'Tiempo por Iteración {mh} \n {problem} ({binarizacion}) corrida {corrida}')
    axTime.set_ylabel("Tiempo (s)")
    axTime.set_xlabel("Iteración")
    axTime.legend(loc='upper right')
    plt.savefig(f'{DIR_GRAFICOS}Time_{mh}_SCP_{problem}_{binarizacion}_{corrida}.pdf')
    plt.close('all')

def graficar_boxplot_violin(instancia, binarizacion):
    direccion_datos = f'{DIR_RESULTADO}fitness_SCP_{instancia}_{binarizacion}.csv'
    datos = pd.read_csv(direccion_datos)
    
    # Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='MH', y=' FITNESS', data=datos)
    plt.title(f'Boxplot Fitness {instancia} ({binarizacion})')
    plt.savefig(f'{DIR_BOXPLOT}boxplot_fitness_SCP_{instancia}_{binarizacion}.pdf')
    plt.close()
    
    # Violinplot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='MH', y=' FITNESS', data=datos)
    plt.title(f'Violinplot Fitness {instancia} ({binarizacion})')
    plt.savefig(f'{DIR_VIOLIN}violinplot_fitness_SCP_{instancia}_{binarizacion}.pdf')
    plt.close()

def escribir_resumenes(mhs_instances, archivoResumenFitness, archivoResumenTimes, archivoResumenPercentage):
    for name in MHS_LIST:
        mh = mhs_instances[name]
        archivoResumenFitness.write(f"{name}, {np.min(mh.fitness)}, {np.round(np.mean(mh.fitness), 3)}, {np.round(np.std(mh.fitness), 3)}\n")
        archivoResumenTimes.write(f"{name}, {np.min(mh.time)}, {np.round(np.mean(mh.time), 3)}, {np.round(np.std(mh.time), 3)}\n")
        archivoResumenPercentage.write(f"{name}, {np.round(np.mean(mh.xpl), 3)}, {np.round(np.mean(mh.xpt), 3)}\n")

# Procesar archivos de resultados
def procesar_archivos(instancia, blob, archivo_fitness, binarizacion):
    corrida = 1
    archivos_validos = False  # Variable para verificar si hay archivos válidos

    for nombre_archivo, contenido in blob:
        direccion_destino = f'{DIR_TRANSITORIO}{nombre_archivo}.csv'
        util.writeTofile(contenido, direccion_destino)

        # Extraer binarización desde el nombre del archivo
        partes_nombre = nombre_archivo.split('_')
        
        if len(partes_nombre) > 2:
            archivo_binarizacion = partes_nombre[-1].replace('.csv', '')  # Suponiendo que la binarización está al final
        else:
            archivo_binarizacion = "UNKNOWN"
            
        mh = partes_nombre[0]  # Metaheurística
        
        # Procesar solo si la binarización coincide con la actual
        if archivo_binarizacion != binarizacion:
            print(f"[DEBUG] No coincide la binarización: Archivo {nombre_archivo} - Detectada: {archivo_binarizacion} - Esperada: {binarizacion}")
            os.remove(direccion_destino)  # Eliminar el archivo temporal si no corresponde
            continue

        archivos_validos = True  # Hay al menos un archivo válido
        # Actualizar datos
        if mh in MHS_LIST:
            data = pd.read_csv(direccion_destino)
            actualizar_datos(mhs_instances, mh, archivo_fitness, data)
            mhs_instances[mh].bestFitness = data['fitness']
            mhs_instances[mh].bestTime = data['time']

            # Generar gráficos por corrida
            if GRAFICOS:
                graficar_datos(
                    data['iter'], data['fitness'], data['XPL'], data['XPT'], data['time'], 
                    mh, f"{instancia}_{binarizacion}", corrida, binarizacion
                )

        os.remove(direccion_destino)
        corrida += 1

    archivo_fitness.close()

    if not archivos_validos:
        print(f"[ADVERTENCIA] No se encontraron archivos válidos para la binarización '{binarizacion}' en la instancia '{instancia}'.")

def graficar_mejores_resultados(instancia, mhs_instances, binarizacion):
    for name in MHS_LIST:
        mh = mhs_instances[name]
        
        # Gráfico de mejores fitness
        plt.plot(range(len(mh.bestFitness)), mh.bestFitness, color='r', label='Best Fitness')
        plt.title(f'Best Fitness \n {instancia} ({binarizacion})')
        plt.ylabel("Fitness")
        plt.xlabel("Iteration")
        plt.legend()
        plt.savefig(f'{DIR_BEST}fitness_SCP_{instancia}_{binarizacion}.pdf')
        plt.close()

        # Gráfico de mejores tiempos
        plt.plot(range(len(mh.bestTime)), mh.bestTime, color='b', label='Best Time')
        plt.title(f'Time (s) \n {instancia} ({binarizacion})')
        plt.ylabel("Time (s)")
        plt.xlabel("Iteration")
        plt.legend()
        plt.savefig(f'{DIR_BEST}time_SCP_{instancia}_{binarizacion}.pdf')
        plt.close()


lista_instancias = ', '.join([f'"{func}"' for func in EXPERIMENTS["instancias"]["SCP"]])
lista_bin = EXPERIMENTS["DS_actions"]

print("lista binarizaciones: ", lista_bin)

# Procesar cada instancia
instancias = bd.obtenerInstancias(lista_instancias)

# Limpia los datos de las metaheurísticas
def limpiar_datos_mhs(mhs_instances):
    for name in mhs_instances:
        mh = mhs_instances[name]
        mh.div.clear()
        mh.fitness.clear()
        mh.time.clear()
        mh.xpl.clear()
        mh.xpt.clear()
        mh.bestFitness.clear()
        mh.bestTime.clear()

# Procesar cada instancia
for instancia in instancias:
    print(f"Procesando instancia: {instancia[1]}")
    blob = bd.obtenerArchivos(instancia[1])

    # Validar si hay experimentos para la instancia
    if not blob:
        print(f"Advertencia: La instancia '{instancia[1]}' no tiene experimentos asociados. Saltando esta instancia...")
        continue

    for binarizacion in lista_bin:
        print(f"Procesando binarización: {binarizacion}")
        
        # Limpiar datos de metaheurísticas antes de procesar la binarización actual
        limpiar_datos_mhs(mhs_instances)
        
        # Crear archivos de salida específicos para la binarización
        archivoResumenFitness = open(f'{DIR_RESULTADO}resumen_fitness_SCP_{instancia[1]}_{binarizacion}.csv', 'w')
        archivoResumenTimes = open(f'{DIR_RESULTADO}resumen_times_SCP_{instancia[1]}_{binarizacion}.csv', 'w')
        archivoResumenPercentage = open(f'{DIR_RESULTADO}resumen_percentage_SCP_{instancia[1]}_{binarizacion}.csv', 'w')
        archivoFitness = open(f'{DIR_RESULTADO}fitness_SCP_{instancia[1]}_{binarizacion}.csv', 'w')

        # Escribir encabezados
        archivoResumenFitness.write("instance, best, avg. fitness, std fitness\n")
        archivoResumenTimes.write("instance, min time (s), avg. time (s), std time (s)\n")
        archivoResumenPercentage.write("instance, avg. XPL%, avg. XPT%\n")
        archivoFitness.write("MH, FITNESS\n")

        # Procesar archivos solo para la binarización actual
        procesar_archivos(instancia[1], blob, archivoFitness, binarizacion)

        # Solo generar resumenes si hay datos en las listas
        if any(len(mhs_instances[mh].fitness) > 0 for mh in mhs_instances):
            # Resumen
            escribir_resumenes(mhs_instances, archivoResumenFitness, archivoResumenTimes, archivoResumenPercentage)

            # Gráficos finales
            graficar_mejores_resultados(instancia[1], mhs_instances, binarizacion)
            graficar_boxplot_violin(instancia[1], binarizacion)
        else:
            print(f"[ADVERTENCIA] No se generaron datos para la binarización '{binarizacion}' en la instancia '{instancia[1]}'.")

        # Cerrar archivos
        archivoResumenFitness.close()
        archivoResumenTimes.close()
        archivoResumenPercentage.close()

print("Procesamiento completado con éxito.")
