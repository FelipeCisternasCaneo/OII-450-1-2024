import time
import numpy as np
import os
import random

from traitlets import Instance
from Diversity.imports import diversidadHussain, porcentajesXLPXPT

from Metaheuristics.imports import iterarGWO, iterarPSA, iterarSCA, iterarWOA, iterarPSO
from Metaheuristics.imports import iterarFOX, iterarEOO, iterarRSA, iterarGOA, iterarHBA
from Metaheuristics.imports import iterarTDO, iterarSHO, iterarSBOA, iterarEHO, iterarEBWOA
from Metaheuristics.imports import iterarFLO, iterarHLOABen, iterarLOA, iterarNO, iterarPOA
from Metaheuristics.imports import IterarPO, iterarWOM, iterarQSO

from Problem.Benchmark.Problem import fitness as f
from Util import util
from BD.sqlite import BD

# Diccionario de metaheurísticas
metaheuristics = {
    "SCA": iterarSCA, "GWO": iterarGWO,
    "WOA": iterarWOA, "PSA": iterarPSA,
    "PSO": iterarPSO, "FOX": iterarFOX,
    "EOO": iterarEOO, "RSA": iterarRSA,
    "GOA": iterarGOA, "HBA": iterarHBA,
    "TDO": iterarTDO, "SHO": iterarSHO,
    "SBOA": iterarSBOA, "EHO": iterarEHO,
    "EBWOA": iterarEBWOA, "FLO": iterarFLO,
    "HLOA": iterarHLOABen, "LOA": iterarLOA,
    "NO": iterarNO, "POA": iterarPOA,
    "PO": IterarPO, "WOM": iterarWOM,
    "QSO": iterarQSO
}

def initialize_population(mh, pop, dim, lb, ub):
    vel, pBestScore, pBest = None, None, None
    
    if mh == 'PSO':
        vel = np.zeros((pop, dim))
        pBestScore = np.zeros(pop)
        pBestScore.fill(float("inf"))
        pBest = np.zeros((pop, dim))
    
    population = np.zeros((pop, dim))
    
    for i in range(dim):
        population[:, i] = (np.random.uniform(0, 1, pop) * (ub[i] - lb[i]) + lb[i])
    
    return population, vel, pBestScore, pBest

def evaluate_population(mh, population, fitness, dim, lb, ub, function):
    pBest, pBestScore = None, None
    if mh == 'PSO':
        pBest = np.zeros_like(population)
        pBestScore = np.full(population.shape[0], float("inf"))
    
    for i in range(population.shape[0]):
        population[i] = np.clip(population[i], lb, ub)
        fitness[i] = f(function, population[i])
        
        if mh == 'PSO' and pBestScore[i] > fitness[i]:
            pBestScore[i] = fitness[i]
            pBest[i] = population[i].copy()

    solutionsRanking = np.argsort(fitness)
    bestIndex = solutionsRanking[0]
    bestFitness = fitness[bestIndex]
    best = population[bestIndex].copy()
    
    return fitness, best, bestFitness, pBest, pBestScore

def iterate_population(mh, metaheuristics, population, iter, maxIter, dim, fitness, best, vel=None, pBest=None, ub=None, lb=None, fo=None):
    if mh in metaheuristics:
        if mh == 'PSO':
            population, vel = metaheuristics[mh](maxIter, iter, dim, population.tolist(), best.tolist(), pBest.tolist(), vel, ub[0])
        elif mh in ['GOA', 'HBA', 'TDO', 'SHO', 'POA']:
            population = metaheuristics[mh](maxIter, iter, dim, population, best.tolist(), fitness.tolist(), fo, 'MIN')
        elif mh == 'SBOA':  # Caso específico para SBOA
            population = metaheuristics[mh](maxIter, iter, dim, population.tolist(), fitness.tolist(), best.tolist(), fo)
        elif mh == 'PO':
            pass
        else:
            population = metaheuristics[mh](maxIter, iter, dim, population.tolist(), best.tolist())
    else:
        raise ValueError(f"Metaheurística {mh} no está soportada.")
    
    return np.array(population), vel

def update_population(population, fitness, dim, lb, ub, function, best, bestFitness, pBest=None, pBestScore=None, mh=None, posibles_mejoras=None):
    for i in range(population.shape[0]):
        population[i] = np.clip(population[i], lb, ub)
        fitness[i] = f(function, population[i])
        
        if mh == 'LOA' and posibles_mejoras is not None:
            fitn = f(function, posibles_mejoras[i])
            if fitn < fitness[i]:
                population[i] = posibles_mejoras[i]
        
        if mh == 'PSO' and pBestScore is not None and fitness[i] < pBestScore[i]:
            pBest[i] = np.copy(population[i])
            pBestScore[i] = fitness[i]
    
    solutionsRanking = np.argsort(fitness)
    if fitness[solutionsRanking[0]] < bestFitness:
        bestFitness = fitness[solutionsRanking[0]]
        best = population[solutionsRanking[0]].copy()
    
    div_t = diversidadHussain(population)
    return population, fitness, best, bestFitness, div_t

def log_progress(iter, maxIter, bestFitness, optimo, timeEjecuted, XPT, XPL, div_t, results):
    if (iter + 1) % (maxIter // 4) == 0:
        print(f"iter: {iter + 1}, best: {bestFitness:.2e}, optimo: {optimo}, time (s): {round(timeEjecuted, 3)}, XPT: {XPT}, XPL: {XPL}, DIV: {div_t}")
    
    results.write(f'{iter + 1},{bestFitness:.2e},{round(timeEjecuted, 3)},{XPL},{XPT},{div_t}\n')

def solverBEN(id, mh, maxIter, pop, function, lb, ub, dim):
    dirResult = './Resultados/'
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
    
    def fo(x):
        for j in range(len(x)):
            x[j] = np.clip(x[j], lb[j], ub[j])
            
        return x, float(f(function, x))
    
    # Inicialización
    population, vel, pBestScore, pBest = initialize_population(mh, pop, dim, lb, ub)
    
    fitness = np.zeros(pop)
    
    fitness, best, bestFitness, pBest, pBestScore = evaluate_population(
        mh, population, fitness, dim, lb, ub, function
    )
    
    # Inicializamos el archivo de resultados
    results = open(dirResult + f"{mh}_{function}_{id}.csv", "w")
    results.write('iter,fitness,time,XPL,XPT,DIV\n')
    
    maxDiversity = diversidadHussain(population)
    XPL, XPT, state = porcentajesXLPXPT(maxDiversity, maxDiversity)

    initializationTime2 = time.time()
    
    # mostramos nuestro fitness iniciales
    print("------------------------------------------------------------------------------------------------------")
    print(f"{function} {str(dim)} {mh} - best fitness inicial: {str(bestFitness)}")
    print("------------------------------------------------------------------------------------------------------")
    print("iter: " +
            str(0) +
            ", best: " + str(format(bestFitness, ".2e")) +
            ", optimo: " + str(optimo) +
            ", time (s): " + str(round(initializationTime2 - initializationTime1, 3)) +
            ", XPT: " + str(XPT) +
            ", XPL: " + str(XPL) +
            ", DIV: " + str(maxDiversity))
    
    results.write(f'0,{str(format(bestFitness, ".2e"))},{str(round(initializationTime2 - initializationTime1, 3))},{str(XPL)},{str(XPT)},{str(maxDiversity)}\n')
    
    if mh == 'PO':
        iterarPO = IterarPO(fo, dim, pop, maxIter, lb[0], ub[0])
        
    # Bucle de iteraciones
    for iter in range(0, maxIter):
        timerStart = time.time()
        
        population, vel = iterate_population(mh, metaheuristics, population, iter, maxIter, dim, fitness, best, vel, pBest, ub, lb, fo)
        
        if mh == 'PO':
            iterarPO.pob(population.tolist(), iter)
            population = iterarPO.optimizer(iter)
        
        population, fitness, best, bestFitness, div_t = update_population(
            population, fitness, dim, lb, ub, function, best, bestFitness, pBest, pBestScore, mh
        )
        
        div_t = diversidadHussain(population)

        if maxDiversity < div_t: maxDiversity = div_t
            
        XPL, XPT, state = porcentajesXLPXPT(div_t, maxDiversity)
        
        timerFinal = time.time()
        log_progress(iter, maxIter, bestFitness, optimo, timerFinal - timerStart, XPT, XPL, div_t, results)

    finalTime = time.time()
    
    print("------------------------------------------------------------------------------------------------------")
    print(f"Tiempo de ejecucion (s): {finalTime - initialTime}")
    print(f"Best fitness: {bestFitness}")
    print("------------------------------------------------------------------------------------------------------")
    
    results.close()
    binary = util.convert_into_binary(dirResult + f"{mh}_{function}_{id}.csv")
    bd.insertarIteraciones(f"{mh}_{function}", binary, id)
    bd.insertarResultados(bestFitness, finalTime - initialTime, best, id)
    bd.actualizarExperimento(id, 'terminado')
    os.remove(dirResult + f"{mh}_{function}_{id}.csv")