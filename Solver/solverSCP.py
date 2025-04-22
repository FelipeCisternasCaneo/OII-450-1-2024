import numpy as np
import os
import time

from Problem.SCP.problem import SCP
from Problem.USCP.problem import USCP
from Metaheuristics.imports import IterarPO
from Diversity.Codes.diversity import initialize_diversity, calculate_diversity
from Discretization import discretization as b
from Util import util
from Solver.population.population_SCP import (initialize_population, evaluate_population, binarize_and_evaluate, 
                                update_best_solution, iterate_population_scp)

from BD.sqlite import BD
from Util.log import initial_log_scp_uscp, log_progress, final_log_scp

def solverSCP(id, mh, maxIter, pop, instances, DS, repairType, param, unicost):
    
    bd = BD()
    
    dirResult = './Resultados/Transitorio/'
    
    if unicost:
        instance = USCP(instances)
    else:
        instance = SCP(instances)
    
    # tomo el tiempo inicial de la ejecucion
    initialTime = time.time()
    initializationTime1 = time.time()
    
    results = open(dirResult + mh + "_" + instances.split(".")[0] + "_" + str(id) + ".csv", "w")
    results.write(f'iter,fitness,time,XPL,XPT,DIV\n')
    
    # Inicializo la población
    population, vel, pBestScore, pBest = initialize_population(mh, pop, instance)
    
    maxDiversity, XPL, XPT = initialize_diversity(population)
    
    # Genero un vector donde almacenaré los fitness de cada individuo
    fitness = np.zeros(pop)

    # Evaluo la población inicial
    fitness, best, bestFitness, pBest, pBestScore = evaluate_population(
        mh, population, fitness, instance, pBest, pBestScore, repairType)
    
    matrixBin = population.copy()
    
    i = population.__len__() - 1
    
    initializationTime2 = time.time()
    
    initial_log_scp_uscp(instance, DS, bestFitness, instances, initializationTime1, initializationTime2, XPT, XPL, maxDiversity, results)
    
    posibles_mejoras = None
    
    # Función objetivo para GOA, HBA, TDO, SHO y SBOA
    def fo(x):
        x = b.aplicarBinarizacion(x, DS, best, matrixBin[i])
        x = instance.repair(x, repairType) # Reparación de soluciones
        
        return x, instance.fitness(x) # Return de la solución reparada y valor de función objetivo
    
    if mh == 'PO':
        iterarPO = IterarPO(fo, instance.getColumns(), pop, maxIter, 0, 1)
        
    for iter in range(1, maxIter + 1):
        # obtengo mi tiempo inicial
        timerStart = time.time()
        
        # --- Manejo especial para PO (se mantiene fuera de iterate_population_scp) ---
        
        if mh == 'PO':
            # 'population' no fue modificada por iterate_population_scp en este caso
            iterarPO.pob(population, iter)
            population = iterarPO.optimizer(iter)
            if not isinstance(population, np.ndarray):
                population = np.array(population)
                
        # --- Fin Manejo PO ---

        population, vel, posibles_mejoras = iterate_population_scp(
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
        
        # Binarizo, calculo de factibilidad de cada individuo y calculo del fitness
        population, fitness, pBest = binarize_and_evaluate(
            mh, population, fitness, DS, best, matrixBin, instance, 
            repairType, pBest, pBestScore, posibles_mejoras, fo)

        # Actualizo mi mejor solucion
        best, bestFitness = update_best_solution(population, fitness, best, bestFitness)
        
        matrixBin = population.copy()

        # Calculo de diversidad
        div_t, maxDiversity, XPL, XPT = calculate_diversity(population, maxDiversity)

        timerFinal = time.time()
        
        # calculo mi tiempo para la iteracion t
        timeExecuted = timerFinal - timerStart
        
        log_progress(iter, maxIter, bestFitness, instance.getOptimum(), timeExecuted, XPT, XPL, div_t, results)
        
    finalTime = time.time()
    
    numberOfSubsets = str(sum(best))
    
    final_log_scp(bestFitness, numberOfSubsets, initialTime, finalTime)
    
    results.close()
    
    binary = util.convert_into_binary(dirResult + mh + "_" + instances.split(".")[0] + "_" + str(id) + ".csv")
    
    fileName = mh + "_" + instances.split(".")[0]

    bd.insertarIteraciones(fileName, binary, id)
    bd.insertarResultados(bestFitness, finalTime - initialTime, best, id)
    bd.actualizarExperimento(id, 'terminado')
    
    os.remove(dirResult + mh + "_" + instances.split(".")[0] + "_" + str(id) + ".csv")