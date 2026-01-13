"""
SCP/USCP Solver with Chaotic Maps

Este solver es una versión ESPECIALIZADA que REQUIERE mapas caóticos.
NO reemplaza solverSCP.py original.

Diferencias clave:
- Usa population_SCP_Chaotic.py (versión con mapas caóticos)
- Genera secuencia caótica al inicio
- Parámetro 'chaotic_map_name' es REQUERIDO (no opcional)

Uso:
    from Solver.solverSCP_Chaotic import solverSCP_Chaotic
    
    solverSCP_Chaotic(
        id=1,
        mh='GWO',
        maxIter=500,
        pop=50,
        instances='scp41',
        DS='V1-STD',
        repairType='complex',
        param='0.4;mut:0.50',
        unicost=False,
        chaotic_map_name='LOG'  # ← REQUERIDO
    )
"""

import numpy as np
import os
import time

from Problem.SCP.problem import SCP
from Problem.USCP.problem import USCP
from Metaheuristics.imports import IterarPO
from Diversity.Codes.diversity import initialize_diversity, calculate_diversity
from Diversity.imports import compute_gap_rdp, diversity_per_dimension, population_entropy
from Discretization import discretization as b
from Util.console_logging import print_initial, print_iteration, print_final
from Util.util import convert_into_binary

# IMPORTAR MÓDULOS CAÓTICOS
from Solver.population.population_SCP_Chaotic import (
    initialize_population_chaotic,
    evaluate_population_chaotic,
    binarize_and_evaluate_chaotic,
    update_best_solution_chaotic,
    iterate_population_scp_chaotic
)

from ChaoticMaps import get_chaotic_map, CHAOTIC_MAP_NAMES

from BD.sqlite import BD


def solverSCP_Chaotic(id, mh, maxIter, pop, instances, DS, repairType, param, unicost, chaotic_map_name):
    """
    Solver para SCP/USCP con mapas caóticos (VERSIÓN ESPECIALIZADA).
    
    REQUISITOS:
    - chaotic_map_name debe ser válido: 'LOG', 'SINE', 'TENT', 'CIRCLE', 'SINGER', 'SINU', 'PIECE'
    - El módulo ChaoticMaps debe estar disponible
    
    Args:
        id (int): ID del experimento
        mh (str): Metaheurística a usar
        maxIter (int): Máximo de iteraciones
        pop (int): Tamaño de población
        instances (str): Nombre de la instancia (e.g., 'scp41')
        DS (str): Función de transferencia-binarización (e.g., 'V1-STD')
        repairType (str): Tipo de reparación ('simple' o 'complex')
        param (str): Parámetros adicionales de MH (e.g., '0.4;mut:0.50')
        unicost (bool): True para USCP, False para SCP
        chaotic_map_name (str): Nombre del mapa caótico (REQUERIDO)
    
    Raises:
        ValueError: Si chaotic_map_name es inválido o None
    """
    
    # ========== VALIDACIÓN MAPA CAÓTICO ==========
    if not chaotic_map_name:
        raise ValueError(
            "solverSCP_Chaotic requiere un mapa caótico. "
            "Usa solverSCP.py estándar si no quieres mapas caóticos."
        )

    valid_maps = ['LOG', 'SINE', 'TENT', 'CIRCLE', 'SINGER', 'SINU', 'PIECE', 'CHEB', 'GAUS']
    if chaotic_map_name not in valid_maps:
        raise ValueError(
            f"Mapa caótico '{chaotic_map_name}' inválido. "
            f"Opciones: {valid_maps}"
        )
    
    # ========== SETUP INICIAL ==========
    bd = BD()
    dirResult = './Resultados/Transitorio/'
    os.makedirs(dirResult, exist_ok=True)

    # Cargar instancia
    instance = USCP(instances) if unicost else SCP(instances)
    dim = instance.getColumns()
    
    # ========== GENERAR SECUENCIA CAÓTICA ==========
    print(f"[] Generando mapa caótico: {CHAOTIC_MAP_NAMES[chaotic_map_name]}")
    
    # Calcular cantidad de valores necesarios
    # Fórmula: maxIter * pop * dim
    quantity_elements = maxIter * pop * dim
    
    try:
        chaotic_func = get_chaotic_map(chaotic_map_name)
        chaotic_map = chaotic_func(x0=0.7, quantity=quantity_elements)
        
        print(f"[] Secuencia caótica generada: {quantity_elements:,} valores")
        print(f"    Rango: [{chaotic_map.min():.6f}, {chaotic_map.max():.6f}]")
    
    except Exception as e:
        raise RuntimeError(f"Error al generar mapa caótico: {e}")
    
    # ========== TIEMPOS ==========
    initialTime = time.time()
    initializationTime1 = time.time()

    # ========== ARCHIVOS DE SALIDA ==========
    base_name = f"{mh}_{instances.split('.')[0]}_{chaotic_map_name}_{id}"
    results_path = os.path.join(dirResult, f"{base_name}.csv")
    results_divj_path = os.path.join(dirResult, f"{base_name}_divj.csv")

    results = open(results_path, "w")
    results.write("iter,best_fitness,mean_fitness,std_fitness,time,XPL,XPT,DIV,GAP,RDP,ENT,Divj_mean,Divj_min,Divj_max\n")

    results_divj = open(results_divj_path, "w")
    divj_header = ",".join([f"Divj_{j+1}" for j in range(dim)])
    results_divj.write(f"iter,{divj_header}\n")

    # ========== INICIALIZACIÓN POBLACIÓN ==========
    population, vel, pBestScore, pBest = initialize_population_chaotic(mh, pop, instance)
    maxDiversity, XPL, XPT = initialize_diversity(population)

    # ========== EVALUACIÓN INICIAL ==========
    fitness = np.zeros(pop)
    fitness, best, bestFitness, pBest, pBestScore = evaluate_population_chaotic(
        mh, population, fitness, instance, pBest, pBestScore, repairType
    )

    initializationTime2 = time.time()

    # ========== MÉTRICAS INICIALES (iter=0) ==========
    meanFitness0 = float(np.mean(fitness))
    stdFitness0 = float(np.std(fitness))
    gap0, rdp0 = compute_gap_rdp(bestFitness, instance.getOptimum())

    lb = [0.0] * dim
    ub = [1.0] * dim

    divj_vec0, divj_mean0, divj_min0, divj_max0 = diversity_per_dimension(population)
    ent_avg0, _ = population_entropy(population, bins=20, lb=lb, ub=ub)

    time0 = initializationTime2 - initializationTime1
    results.write(
        f"0,{bestFitness:.6e},{meanFitness0:.6f},{stdFitness0:.6f},"
        f"{time0:.3f},{XPL:.6f},{XPT:.6f},{maxDiversity:.6f},"
        f"{gap0:.6f},{rdp0:.6f},{ent_avg0:.6f},{divj_mean0:.6f},{divj_min0:.6f},{divj_max0:.6f}\n"
    )
    results_divj.write("0," + ",".join([f"{v:.6f}" for v in divj_vec0]) + "\n")

    # ========== LOGGING INICIAL ==========
    print_initial(
        f"{instances} | cols: {dim} | {mh} |  {chaotic_map_name}", 
        bestFitness
    )

    matrixBin = population.copy()
    posibles_mejoras = None

    # ========== FUNCIÓN OBJETIVO ==========
    def fo(x):
        # Para fo() usamos indexación simple ya que no tiene contexto de iter/i
        x_bin = b.aplicarBinarizacion(
            x, DS, best, matrixBin[len(population) - 1],
            chaotic_map=chaotic_map, 
            chaotic_index=0
        )
        x_bin = instance.repair(x_bin, repairType)
        return x_bin, instance.fitness(x_bin)

    if mh == 'PO':
        iterarPO = IterarPO(fo, dim, pop, maxIter, 0, 1)

    # ========== BUCLE PRINCIPAL ==========
    for iter in range(1, maxIter + 1):
        timerStart = time.time()

        # PO especial
        if mh == 'PO':
            iterarPO.pob(population, iter)
            population = iterarPO.optimizer(iter)
            if not isinstance(population, np.ndarray):
                population = np.array(population)

        # Movimiento de la población
        population, vel, posibles_mejoras, pBest = iterate_population_scp_chaotic(
            mh=mh,
            population=population,
            iter=iter,
            maxIter=maxIter,
            instance=instance,
            fitness=fitness,
            best=best,
            vel=vel,
            pBest=pBest,
            fo=fo,
            param=param
        )

        # BINARIZACIÓN CON MAPAS CAÓTICOS
        population, fitness, pBest = binarize_and_evaluate_chaotic(
            mh=mh,
            population=population,
            fitness=fitness,
            DS=DS,
            best=best,
            matrixBin=matrixBin,
            instance=instance,
            repairType=repairType,
            pBest=pBest,
            pBestScore=pBestScore,
            posibles_mejoras=posibles_mejoras,
            fo=fo,
            chaotic_map=chaotic_map,  # ← Secuencia caótica
            iter=iter,
            pop_size=pop,
            maxIter=maxIter
        )

        # Actualizar mejor solución
        best, bestFitness = update_best_solution_chaotic(population, fitness, best, bestFitness)
        matrixBin = population.copy()

        # Diversidad
        div_t, maxDiversity, XPL, XPT = calculate_diversity(population, maxDiversity)

        # Métricas adicionales
        meanFitness = float(np.mean(fitness))
        stdFitness = float(np.std(fitness))
        gap, rdp = compute_gap_rdp(bestFitness, instance.getOptimum())

        divj_vec, divj_mean, divj_min, divj_max = diversity_per_dimension(population)
        ent_avg, _ = population_entropy(population, bins=20, lb=lb, ub=ub)

        # Tiempo iteración
        timerFinal = time.time()
        dt = timerFinal - timerStart

        # Escribir resultados
        results.write(
            f"{iter},{bestFitness:.6e},{meanFitness:.6f},{stdFitness:.6f},"
            f"{dt:.3f},{XPL:.6f},{XPT:.6f},{div_t:.6f},"
            f"{gap:.6f},{rdp:.6f},{ent_avg:.6f},{divj_mean:.6f},{divj_min:.6f},{divj_max:.6f}\n"
        )
        results_divj.write(f"{iter}," + ",".join([f"{v:.6f}" for v in divj_vec]) + "\n")

        # Consola
        print_iteration(iter, maxIter, bestFitness, instance.getOptimum(), dt, XPT, XPL, div_t)

    # ========== CIERRE ==========
    finalTime = time.time()
    print_final(bestFitness, initialTime, finalTime)

    results.flush()
    results.close()
    results_divj.flush()
    results_divj.close()

    # ========== GUARDADO EN BD ==========
    binary = convert_into_binary(results_path)
    fileName = f"{mh}_{instances.split('.')[0]}_{chaotic_map_name}"
    
    bd.insertarIteraciones(fileName, binary, id)
    bd.insertarResultados(bestFitness, finalTime - initialTime, best, id)
    bd.actualizarExperimento(id, 'terminado')

    # Limpieza
    os.remove(results_path)
    os.remove(results_divj_path)
    
    print(f"[] Experimento completado: {fileName}")