import numpy as np

from colorama import Fore, init

from datetime import datetime

init(autoreset=True)

def obtener_fecha_hora():
    # Obtener la fecha y hora actual
    current_time = datetime.now()

    # Formatear la fecha
    formatted_time = current_time.strftime("%A %d %B %Y, %H:%M:%S")

    return formatted_time

def log_message(iter, bestFitness, optimo, timeEjecuted, XPT, XPL, div_t, results):
    msg = (
        f"Iteración: {iter:<4} | "
        f"Mejor Fitness: {bestFitness:>7.2e} | "
        f"Óptimo: {optimo:>9.2e} | "
        f"Tiempo (s): {timeEjecuted:>4.3f} | "
        f"XPT: {XPT:>6.2f} | "
        f"XPL: {XPL:>6.2f} | "
        f"DIV: {div_t:>5.2f}"
    )
    print(msg)
    
    if results:
        try:
            results.write(f"{iter},{bestFitness:.2e},{round(timeEjecuted, 3)},{XPL},{XPT},{div_t}\n")
        except Exception as e:
            print(f"Error al escribir en el archivo de resultados: {e}")

def log_progress(iter, maxIter, bestFitness, optimo, timeEjecuted, XPT, XPL, div_t, results):
    # Siempre escribir en el archivo
    try:
        results.write(f"{iter},{bestFitness:.2e},{round(timeEjecuted, 3)},{XPL},{XPT},{div_t}\n")
    except Exception as e:
        print(f"Error al escribir en el archivo de resultados: {e}")

    # Imprimir en consola solo según la condición
    if (iter) % (maxIter // 4) == 0:
        msg = (
            f"Iteración: {iter:<4} | "
            f"Mejor Fitness: {bestFitness:>7.2e} | "
            f"Óptimo: {optimo:>9.2e} | "
            f"Tiempo (s): {timeEjecuted:>4.3f} | "
            f"XPT: {XPT:>6.2f} | "
            f"XPL: {XPL:>6.2f} | "
            f"DIV: {div_t:>5.2f}"
        )
        print(msg)

def initial_log(function, dim, mh, bestFitness, optimo, initializationTime1, initializationTime2, XPT, XPL, maxDiversity, results):
    print(f"{function} {dim} {mh} - Best Fitness Inicial: {bestFitness:.2e}")
    print("------------------------------------------------------------------------------------------------------")
    log_message(0, bestFitness, optimo, initializationTime2 - initializationTime1, XPT, XPL, maxDiversity, results)

def initial_log_scp_uscp(instance, DS, bestFitness, instances, initializationTime1, initializationTime2, XPT, XPL, maxDiversity, results):
    print(f"{instances} - {DS} - {instance.getBlockSizes()} - Best Fitness Inicial: {bestFitness:.2e}")
    print("------------------------------------------------------------------------------------------------------")
    log_message(0, bestFitness, instance.getOptimum(), initializationTime2 - initializationTime1, XPT, XPL, maxDiversity, results)

def final_log(bestFitness, initialTime, finalTime):
    print("------------------------------------------------------------------------------------------------------")
    print(f"{Fore.GREEN}Tiempo de ejecución (s): {(finalTime - initialTime):.2f}")
    print(f"{Fore.GREEN}Best Fitness: {bestFitness:.2e}")
    print("------------------------------------------------------------------------------------------------------")
        
def final_log_scp(bestFitness, subsSelected, initialTime, finalTime):
    print("------------------------------------------------------------------------------------------------------")
    print(f"{Fore.GREEN}Tiempo de ejecución (s): {(finalTime - initialTime):.2f}")
    print(f"{Fore.GREEN}Best Fitness: {bestFitness:.2e} ({bestFitness})")
    print(f"{Fore.GREEN}Subconjuntos seleccionados: {subsSelected}")
    print("------------------------------------------------------------------------------------------------------")

def log_experimento(data):
    """Log para mostrar el inicio del procesamiento de un experimento."""
    print(f"Procesando Experimento ID: {data[0][0]}")
    print(f"Instancia: {data[0][1]}")
    print(f"{Fore.CYAN}Metaheurística: {data[0][2]}")
    print(f"Binarización: {data[0][3]}")
    print(f"Parámetros: {data[0][4]}")
    print(f"Estado: {data[0][9]}")
    print("------------------------------------------------------------------------------------------------------")

def log_error(id, mensaje):
    """Log para mostrar errores en rojo."""
    print(f"{Fore.RED}Error al ejecutar el experimento ID: {id}")
    print(f"{Fore.RED}Mensaje: {mensaje}")
    print("------------------------------------------------------------------------------------------------------")

def log_final(total_time):
    """Log para mostrar el final del procesamiento con tiempo total."""
    
    if total_time < 0.01:
        print("------------------------------------------------------------------------------------------------------")
        print(f"{Fore.RED}No se han ejecutado experimentos.")
        print("------------------------------------------------------------------------------------------------------")
        return

    print(f"{Fore.GREEN}Se han ejecutado todos los experimentos pendientes.")
    
    if total_time < 60:
        print(f"{Fore.GREEN}Tiempo total de ejecución: {total_time:.2f} segundos")
    elif total_time < 3600:
        print(f"{Fore.GREEN}Tiempo total de ejecución: {total_time / 60:.2f} minutos")
    else:
        horas = int(total_time // 3600)
        minutos = int((total_time % 3600) // 60)
        print(f"{Fore.GREEN}Tiempo total de ejecución: {horas} horas y {minutos} minutos")
    
    print("------------------------------------------------------------------------------------------------------")
    
def resumen_experimentos(log_resumen, cantidad):
    """Muestra un resumen detallado de los experimentos."""
    
    if cantidad == 0:
        print("No se han ingresado experimentos.")
        
        return
    
    print("\n" + "-" * 100)
    print(f"{'RESUMEN DETALLADO DE EXPERIMENTOS':^100}")
    print("-" * 100)

    print(f"{'Problema':<10} {'Instancia':<12} {'Dimensión':<15} {'MH':<10} "
          f"{'Iteraciones':<12} {'Población':<10} {'DS':<10} {'# Experimentos':<15}")
    print("-" * 100)

    for log in log_resumen:
        ds_value = "-"
        if 'Binarización' in log and log["Problema"] != "BEN":
            ds_value = log['Binarización']

        print(f"{log['Problema']:<10} {log['Instancia']:<12} {log['Dimensión']:<15} {log['MH']:<10} "
              f"{log['Iteraciones']:<12} {log['Población']:<10} {ds_value:<10} {log['Total Experimentos']:<15}")

    print("-" * 100)
    print(f"TOTAL EXPERIMENTOS INGRESADOS: {cantidad}")
    print("-" * 100)
    
def escribir_resumenes(mhs_instances, archivoResumenFitness, archivoResumenTimes, archivoResumenPercentage, MHS_LIST):
    for name in MHS_LIST:
        mh = mhs_instances[name]
        
        archivoResumenFitness.write(f"{name}, {np.min(mh.fitness)}, {np.round(np.mean(mh.fitness), 3)}, {np.round(np.std(mh.fitness), 3)}\n")
        archivoResumenTimes.write(f"{name}, {np.min(mh.time)}, {np.round(np.mean(mh.time), 3)}, {np.round(np.std(mh.time), 3)}\n")
        archivoResumenPercentage.write(f"{name}, {np.round(np.mean(mh.xpl), 3)}, {np.round(np.mean(mh.xpt), 3)}\n")

def log_fecha_hora(evento):
    """Muestra la fecha y hora actual."""
    
    print(f"{Fore.GREEN}{evento}: {obtener_fecha_hora()}")
    print("------------------------------------------------------------------------------------------------------")