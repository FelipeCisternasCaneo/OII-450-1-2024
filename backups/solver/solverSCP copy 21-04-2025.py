import numpy as np
import os
import time

from Problem.SCP.problem import SCP

from Metaheuristics.imports import iterarGWO, iterarSCA, iterarWOA, iterarPSA, iterarGA
from Metaheuristics.imports import iterarPSO, iterarFOX, iterarEOO, iterarRSA, iterarGOA
from Metaheuristics.imports import iterarHBA, iterarTDO, iterarSHO, iterarSBOA
from Metaheuristics.imports import iterarEBWOA, iterarFLO, iterarHLOAScp, iterarLOA, iterarNO
from Metaheuristics.imports import iterarPOA, IterarPO, iterarWOM, iterarQSO, iterarAOA

from Diversity.Codes.diversity import initialize_diversity, calculate_diversity
from Discretization import discretization as b

from Util import util
from Solver.population.population_SCP import (initialize_population, evaluate_population, binarize_and_evaluate, 
                                update_best_solution)

from BD.sqlite import BD

from Util.log import initial_log_scp_uscp, log_progress, final_log

def solverSCP(id, mh, maxIter, pop, instances, DS, repairType, param):
    dirResult = './Resultados/Transitorio/'
    
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

        # perturbo la poblacion con la metaheuristica, pueden usar SCA y GWO
        # en las funciones internas tenemos los otros dos for, for de individuos y for de dimensiones
        
        if mh == "SCA":
            population = iterarSCA(maxIter, iter, instance.getColumns(), population, best)
            
        if mh == "GWO":
            population = iterarGWO(maxIter, iter, instance.getColumns(), population, fitness, 'MIN')
            
        if mh == 'WOA':
            population = iterarWOA(maxIter, iter, instance.getColumns(), population, best)
            
        if mh == 'PSA':
            population = iterarPSA(maxIter, iter, instance.getColumns(), population, best)
            
        if mh == "GA":
            partes = param.split(";")
            
            cross = float(partes[0])
            muta = float(partes[1].split(":")[1])
            
            population = iterarGA(population, fitness, cross, muta)
            
        if mh == 'PSO':
            population, vel = iterarPSO(maxIter, iter, instance.getColumns(), population, best, pBest, vel, 1)
            
        if mh == 'FOX':
            population = iterarFOX(maxIter, iter, instance.getColumns(), population, best)
            
        if mh == 'EOO':
            population = iterarEOO(maxIter, iter, population.tolist(), best.tolist())
            
        if mh == 'RSA':
            population = iterarRSA(maxIter, iter, instance.getColumns(), population, best, 0, 1)
            
        if mh == 'GOA':
            population = iterarGOA(maxIter, iter, instance.getColumns(), population, best, fitness, fo, 'MIN')
            
        if mh == 'HBA':
            population = iterarHBA(maxIter, iter, instance.getColumns(), population, best, fitness, fo, 'MIN')
            
        if mh == 'TDO':
            population = iterarTDO(maxIter, iter, instance.getColumns(), population, fitness, fo, 'MIN')
            
        if mh == 'SHO':
            population = iterarSHO(maxIter, iter, instance.getColumns(), population, best, fo, 'MIN')
            
        if mh == 'SBOA':
            population = iterarSBOA(maxIter, iter, instance.getColumns(), population, fitness, best, fo)
            
        if mh == 'EBWOA':
            population = iterarEBWOA(maxIter, iter, instance.getColumns(), population, best, 0, 1)
            
        if mh == 'FLO': 
            population = iterarFLO(maxIter, iter, instance.getColumns(), population, fitness, best, fo, 'MIN', 0, 1)
            
        if mh == 'HLOA':
            population = iterarHLOAScp(maxIter, iter, instance.getColumns(), population, best, 0, 1)
            
        if mh == "LOA":
            population, posibles_mejoras = iterarLOA(maxIter, population, best, 0, 1, iter, instance.getColumns())
            
        if mh == 'NO':
            population = iterarNO(maxIter, iter, instance.getColumns(), population, best)
            
        if mh == 'POA':
            population = iterarPOA(maxIter, iter, instance.getColumns(), population, fitness, fo, 0, 1, 'MIN')
            
        if mh == 'PO':
            iterarPO.pob(population, iter)
            population = iterarPO.optimizer(iter)
            
        if mh == 'WOM':
            lb = [0] * instance.getColumns()
            ub = [1] * instance.getColumns()
            
            population = iterarWOM(maxIter, iter, instance.getColumns(), population, fitness, lb, ub, fo)
        
        if mh == 'QSO':
            population = iterarQSO(maxIter, iter, instance.getColumns(), population, best, 0, 1)
            
        if mh == 'AOA':
            population = iterarAOA(maxIter, iter, instance.getColumns(), population, best)
        
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
    
    final_log(bestFitness, initialTime, finalTime)
    
    results.close()
    
    binary = util.convert_into_binary(dirResult + mh + "_" + instances.split(".")[0] + "_" + str(id) + ".csv")
    
    fileName = mh + "_" + instances.split(".")[0]

    bd = BD()
    bd.insertarIteraciones(fileName, binary, id)
    bd.insertarResultados(bestFitness, finalTime - initialTime, best, id)
    bd.actualizarExperimento(id, 'terminado')
    
    os.remove(dirResult + mh + "_" + instances.split(".")[0] + "_" + str(id) + ".csv")