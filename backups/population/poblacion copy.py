import numpy as np
from Diversity.imports import diversidadHussain
from Problem.Benchmark.Problem import fitness as f

def initialize_population(mh, pop, dim, lb, ub):
    vel, pBestScore, pBest = None, None, None
    
    if mh == 'PSO':
        vel = np.zeros((pop, dim))
        pBestScore = np.full(pop, float("inf"))  # Más directo
        pBest = np.zeros((pop, dim))
    
    # Vectorización para generar población inicial
    lb = np.array(lb)
    ub = np.array(ub)
    
    population = np.random.uniform(0, 1, (pop, dim)) * (ub - lb) + lb
    
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

def update_population(population, fitness, dim, lb, ub, function, best, bestFitness, pBest=None, pBestScore=None, mh=None, posibles_mejoras=None):
    # Aplicar límites a toda la población
    population = np.clip(population, lb, ub)

    # Evaluar fitness de toda la población (bucle explícito)
    for i in range(population.shape[0]):
        fitness[i] = f(function, population[i])

    # Comparar y actualizar posibles mejoras para LOA
    if mh == 'LOA' and posibles_mejoras is not None:
        posibles_mejoras = np.clip(posibles_mejoras, lb, ub)
        
        for i in range(posibles_mejoras.shape[0]):
            mejora_fitness = f(function, posibles_mejoras[i])
            
            if mejora_fitness < fitness[i]:
                population[i] = posibles_mejoras[i]
                fitness[i] = mejora_fitness

    # Actualizar pBest para PSO
    if mh == 'PSO':
        for i in range(population.shape[0]):
            
            if fitness[i] < pBestScore[i]:
                pBestScore[i] = fitness[i]
                pBest[i] = population[i]

    # Encontrar el mejor fitness y solución
    bestIndex = np.argmin(fitness)
    
    if fitness[bestIndex] < bestFitness:
        bestFitness = fitness[bestIndex]
        best = population[bestIndex].copy()

    # Calcular diversidad
    div_t = diversidadHussain(population)

    return population, fitness, best, bestFitness, div_t

def iterate_population(mh, metaheuristics, population, iter, maxIter, dim, fitness, best, vel=None, pBest=None, ub=None, lb=None, fo=None):
    """
    Itera sobre la población usando la metaheurística especificada.
    """
    
    if mh == 'PO':
        return np.array(population), vel, None
    
    if mh not in metaheuristics:
        raise ValueError(f"Metaheurística {mh} no está soportada.")
    
    # Diccionario de configuraciones específicas para metaheurísticas
    metaheuristic_params = {
        'PSO': lambda: metaheuristics[mh](maxIter, iter, dim, population, best, pBest, vel, ub[0]),
        'GOA': lambda: metaheuristics[mh](maxIter, iter, dim, population, best, fitness, fo, 'MIN'),
        'HBA': lambda: metaheuristics[mh](maxIter, iter, dim, population, best, fitness, fo, 'MIN'),
        'SBOA': lambda: metaheuristics[mh](maxIter, iter, dim, population, fitness, best, fo),
        'GWO': lambda: metaheuristics[mh](maxIter, iter, dim, population, fitness, 'MIN'),
        'EOO': lambda: metaheuristics[mh](maxIter, iter, population, best),
        'RSA': lambda: metaheuristics[mh](maxIter, iter, dim, population, best, lb[0], ub[0]),
        'TDO': lambda: metaheuristics[mh](maxIter, iter, dim, population, fitness, fo, 'MIN'),
        'SHO': lambda: metaheuristics[mh](maxIter, iter, dim, population, best, fo, 'MIN'),
        'EHO': lambda: metaheuristics[mh](maxIter, iter, dim, population, best, lb, ub, fitness),
        'EBWOA': lambda: metaheuristics[mh](maxIter, iter, dim, population, best, lb[0], ub[0]),
        'FLO': lambda: metaheuristics[mh](maxIter, iter, dim, population, fitness, best, fo, 'MIN', lb[0], ub[0]),
        'HLOA': lambda: metaheuristics[mh](maxIter, iter, dim, population, best, lb, ub),
        'POA': lambda: metaheuristics[mh](maxIter, iter, dim, population, fitness, fo, lb[0], ub[0], 'MIN'),
        'WOM': lambda: metaheuristics[mh](maxIter, iter, dim, population, fitness, lb, ub, fo),
        'QSO': lambda: metaheuristics[mh](maxIter, iter, dim, population, best, lb, ub),
        'LOA': lambda: metaheuristics[mh](maxIter, population, best, lb[0], ub[0], iter, dim),
        'DEFAULT': lambda: metaheuristics[mh](maxIter, iter, dim, population, best)
    }

    result = metaheuristic_params.get(mh, metaheuristic_params['DEFAULT'])()
    
    if mh == 'LOA':
        population, posibles_mejoras = result
        
        return np.array(population), vel, posibles_mejoras

    # Manejo estándar para otras metaheurísticas
    if isinstance(result, tuple) and len(result) == 2:
        population, vel = result
    
    else:
        population = result
        vel = None

    return np.array(population), vel, None