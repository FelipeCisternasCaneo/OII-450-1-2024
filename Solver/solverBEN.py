import time
import numpy as np
import os
import json

from Diversity.Codes.diversity import initialize_diversity, calculate_diversity
from Metaheuristics.imports import metaheuristics, IterarPO
from Problem.Benchmark.Problem import fitness as f

from Solver.population.population_BEN import initialize_population, evaluate_population, update_population, iterate_population
from Util.log import initial_log, log_progress, final_log
from Util.util import convert_into_binary

from BD.sqlite import BD

def solverBEN(id, mh, maxIter, pop, function, lb, ub, dim):
    dirResult = './Resultados/Transitorio/'
    
    os.makedirs(dirResult, exist_ok = True)
    
    bd = BD()
    
    if not isinstance(lb, list):
        lb = [lb] * dim
        
    if not isinstance(ub, list):
        ub = [ub] * dim
    
    initialTime = time.time()
    optimo = bd.obtenerOptimoInstancia(function)[0][0]
    
    if function == 'F8':
        optimo = optimo * dim
    
    initializationTime1 = time.time()
    
    def fo_vectorized(x):
        x = np.clip(x, lb, ub)
        return x, float(f(function, x))
    
    population, vel, pBestScore, pBest = initialize_population(mh, pop, dim, lb, ub)
    
    fitness = np.zeros(pop)
    
    fitness, best, bestFitness, pBest, pBestScore = evaluate_population(
        mh, population, fitness, dim, lb, ub, function)
    
    # Inicializamos el archivo de resultados
    results = open(dirResult + f"{mh}_{function}_{id}.csv", "w")
    results.write('iter,fitness,time,XPL,XPT,DIV\n')
    
    maxDiversity, XPL, XPT = initialize_diversity(population)

    initializationTime2 = time.time()
    
    initial_log(function, dim, mh, bestFitness, optimo, 
                initializationTime1, initializationTime2, XPT,
                XPL, maxDiversity, results)
    
    if mh == 'PO':
        iterarPO = IterarPO(fo_vectorized, dim, pop, maxIter, lb[0], ub[0])
        
    # Bucle de iteraciones
    for iter in range(1, maxIter + 1):
        timerStart = time.time()
        
        population, vel, posibles_mejoras = iterate_population(
            mh,               # 1
            population,       # 2
            iter,             # 3
            maxIter,          # 4
            dim,              # 5
            fitness,          # 6
            best,             # 7
            vel=vel,          # 8
            pBest=pBest,      # 9
            ub=ub,            # 10
            lb=lb,            # 11
            fo=fo_vectorized  # 12
        )
        
        if mh == 'PO':
            iterarPO.pob(population, iter)
            population = iterarPO.optimizer(iter)
            
        population, fitness, best, bestFitness, div_t = update_population(
        population, fitness, dim, lb, ub, function, best, bestFitness, pBest, pBestScore, mh, posibles_mejoras)
        
        div_t, maxDiversity, XPL, XPT = calculate_diversity(population, maxDiversity)
        
        timerFinal = time.time()
        
        log_progress(iter, maxIter, bestFitness, optimo, timerFinal - timerStart, XPT, XPL, div_t, results)

    finalTime = time.time()
    
    final_log(bestFitness, initialTime, finalTime)
    
    results.close()
    binary = convert_into_binary(dirResult + f"{mh}_{function}_{id}.csv")
    
    bd.insertarIteraciones(f"{mh}_{function}", binary, id)
    bd.insertarResultados(bestFitness, finalTime - initialTime, best, id)
    bd.actualizarExperimento(id, 'terminado')
    
    os.remove(dirResult + f"{mh}_{function}_{id}.csv")