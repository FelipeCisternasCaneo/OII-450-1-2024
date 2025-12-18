import numpy as np
import os
import time

from Problem.SCP.problem import SCP
from Problem.USCP.problem import USCP
from Metaheuristics.imports import IterarPO
from Diversity.Codes.diversity import initialize_diversity, calculate_diversity
from Diversity.imports import compute_gap_rdp, diversity_per_dimension, population_entropy  # NEW
from Discretization import discretization as b
from Util.console_logging import print_initial, print_iteration, print_final          # NEW
from Util.util import convert_into_binary                                             # keep (o tu util.convert_into_binary)
from Solver.population.population_SCP import (
    initialize_population, evaluate_population, binarize_and_evaluate,
    update_best_solution, iterate_population_scp
)

from BD.sqlite import BD

def solverSCP(id, mh, maxIter, pop, instances, DS, repairType, param, unicost):
    bd = BD()
    dirResult = './Resultados/Transitorio/'
    os.makedirs(dirResult, exist_ok=True)

    # Cargar instancia
    instance = USCP(instances) if unicost else SCP(instances)
    dim = instance.getColumns()

    # Tiempos
    initialTime = time.time()
    initializationTime1 = time.time()

    # Archivos de salida
    base_name = f"{mh}_{instances.split('.')[0]}_{id}"
    results_path     = os.path.join(dirResult, f"{base_name}.csv")
    results_divj_path= os.path.join(dirResult, f"{base_name}_divj.csv")

    results = open(results_path, "w")
    results.write("iter,best_fitness,mean_fitness,std_fitness,time,XPL,XPT,DIV,GAP,RDP,ENT,Divj_mean,Divj_min,Divj_max\n")

    results_divj = open(results_divj_path, "w")
    divj_header = ",".join([f"Divj_{j+1}" for j in range(dim)])
    results_divj.write(f"iter,{divj_header}\n")

    # Inicialización de población
    population, vel, pBestScore, pBest = initialize_population(mh, pop, instance)
    maxDiversity, XPL, XPT = initialize_diversity(population)

    # Evaluación inicial
    fitness = np.zeros(pop)
    fitness, best, bestFitness, pBest, pBestScore = evaluate_population(
        mh, population, fitness, instance, pBest, pBestScore, repairType
    )

    initializationTime2 = time.time()

    # === Fila iter 0 con TODAS las métricas ===
    meanFitness0 = float(np.mean(fitness))
    stdFitness0  = float(np.std(fitness))
    gap0, rdp0   = compute_gap_rdp(bestFitness, instance.getOptimum())

    # Para SCP los límites son binarios 0..1
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

    # Logging consola (igual a BEN)
    print_initial(f"{instances} | cols: {dim} | {mh}", bestFitness)

    matrixBin = population.copy()
    posibles_mejoras = None

    # FO para PO/HBA/GOA/etc (igual al original)
    def fo(x):
        x = b.aplicarBinarizacion(x, DS, best, matrixBin[population.__len__() - 1])
        x = instance.repair(x, repairType)
        return x, instance.fitness(x)

    if mh == 'PO':
        iterarPO = IterarPO(fo, dim, pop, maxIter, 0, 1)

    # === Bucle principal ===
    for iter in range(1, maxIter + 1):
        timerStart = time.time()

        # PO especial
        if mh == 'PO':
            iterarPO.pob(population, iter)
            population = iterarPO.optimizer(iter)
            if not isinstance(population, np.ndarray):
                population = np.array(population)

        # Movimiento de la población
        population, vel, posibles_mejoras, pBest = iterate_population_scp(
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

        # Binarizar + reparar + evaluar
        population, fitness, pBest = binarize_and_evaluate(
            mh, population, fitness, DS, best, matrixBin, instance,
            repairType, pBest, pBestScore, posibles_mejoras, fo
        )

        # Actualizar mejor solución
        best, bestFitness = update_best_solution(population, fitness, best, bestFitness)
        matrixBin = population.copy()

        # Diversidad “Hussain” + XPL/XPT (igual a BEN)
        div_t, maxDiversity, XPL, XPT = calculate_diversity(population, maxDiversity)

        # Métricas adicionales
        meanFitness = float(np.mean(fitness))
        stdFitness  = float(np.std(fitness))
        gap, rdp    = compute_gap_rdp(bestFitness, instance.getOptimum())

        divj_vec, divj_mean, divj_min, divj_max = diversity_per_dimension(population)
        ent_avg, _ = population_entropy(population, bins=20, lb=lb, ub=ub)

        # Tiempo iter
        timerFinal = time.time()
        dt = timerFinal - timerStart

        # Escribir fila iter t
        results.write(
            f"{iter},{bestFitness:.6e},{meanFitness:.6f},{stdFitness:.6f},"
            f"{dt:.3f},{XPL:.6f},{XPT:.6f},{div_t:.6f},"
            f"{gap:.6f},{rdp:.6f},{ent_avg:.6f},{divj_mean:.6f},{divj_min:.6f},{divj_max:.6f}\n"
        )
        results_divj.write(f"{iter}," + ",".join([f"{v:.6f}" for v in divj_vec]) + "\n")

        # Consola (cada 1/4 de iteraciones)
        print_iteration(iter, maxIter, bestFitness, instance.getOptimum(), dt, XPT, XPL, div_t)

    # Cierre
    finalTime = time.time()
    print_final(bestFitness, initialTime, finalTime)

    results.close()
    results_divj.close()

    # Guardado en BD (igual)
    binary = convert_into_binary(results_path)
    fileName = f"{mh}_{instances.split('.')[0]}"
    bd.insertarIteraciones(fileName, binary, id)
    bd.insertarResultados(bestFitness, finalTime - initialTime, best, id)
    bd.actualizarExperimento(id, 'terminado')

    # Limpieza
    os.remove(results_path)