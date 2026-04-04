"""
Population Management for SCP/USCP with Chaotic Maps Support

Este módulo es una versión ESPECIALIZADA para usar mapas caóticos.
NO modifica population_SCP.py original.

Uso:
    from Solver.population.population_SCP_Chaotic import (
        initialize_population_chaotic,
        binarize_and_evaluate_chaotic
    )
"""

import numpy as np

from Discretization import discretization as b
from Metaheuristics.imports import metaheuristics, MH_ARG_MAP

# ========== FUNCIONES REUTILIZABLES DEL ORIGINAL ==========
# Estas funciones son idénticas al original (sin cambios)

def initialize_population_chaotic(mh, pop, instance):
    """
    Idéntico al original. Inicializa población binaria.
    """
    vel, pBestScore, pBest = None, None, None
    
    population = np.random.randint(low=0, high=2, size=(pop, instance.getColumns()))
    
    if mh in ['PSO', 'TJO']:
        pBestScore = np.full(pop, float("inf"))
        pBest = population.copy()
    
    if mh == 'PSO':
        vel = np.zeros((pop, instance.getColumns()))
    
    return population, vel, pBestScore, pBest


def evaluate_population_chaotic(mh, population, fitness, instance, pBest, pBestScore, repairType):
    """
    Idéntico al original. Evalúa población inicial.
    """
    n_pop = len(population)
    
    factibilityTest = instance.factibilityTest
    repair = instance.repair
    fitness_func = instance.fitness
    
    for i in range(n_pop):
        flag, _ = factibilityTest(population[i])
        
        if not flag:
            population[i] = repair(population[i], repairType)
            
        fitness[i] = fitness_func(population[i])
        
        if mh == 'PSO':
            if pBestScore[i] > fitness[i]:
                pBestScore[i] = fitness[i]
                pBest[i, :] = population[i, :].copy()
        
        if mh == 'TJO':
            pBest[i, :] = population[i, :].copy()
        
    solutionsRanking = np.argsort(fitness)
    bestRowAux = solutionsRanking[0]
    best = population[bestRowAux].copy()
    bestFitness = fitness[bestRowAux]
    
    return fitness, best, bestFitness, pBest, pBestScore


def iterate_population_scp_chaotic(mh, population, iter, maxIter, instance, fitness, best,
                                   vel=None, pBest=None, fo=None, param=None, userData=None):
    """
    Idéntico al original. Itera población con metaheurísticas.
    """
    if mh == 'PO':
        return np.array(population), vel, None, pBest

    if mh == 'GA':
        if param is None:
            raise ValueError("Parámetros 'cross' y 'muta' no proporcionados para GA.")
        
        partes = param.split(";")
        cross = float(partes[0])
        muta = float(partes[1].split(":")[1])

        new_population = metaheuristics['GA'](population=population, fitness=fitness, cross=cross, muta=muta)
        if not isinstance(new_population, np.ndarray):
            new_population = np.array(new_population)
            
        return new_population, vel, None, pBest
    
    if mh == 'HLOA':
        mh = 'HLOA_SCP'
    
    if mh not in metaheuristics:
        raise ValueError(f"Metaheurística '{mh}' no encontrada en 'metaheuristics'.")
    if mh not in MH_ARG_MAP:
        raise ValueError(f"Mapa de argumentos MH_ARG_MAP para '{mh}' no definido.")

    lb0_val = 0
    ub0_val = 1
    dim = instance.getColumns()
    lb_arr = np.zeros(dim)
    ub_arr = np.ones(dim)

    context = {
        'maxIter': maxIter,
        'iter': iter,
        'dim': dim,
        'population': population,
        'fitness': fitness,
        'best': best,
        'vel': vel,
        'pBest': pBest,
        'ub': ub_arr,
        'lb': lb_arr,
        'ub0': ub0_val,
        'lb0': lb0_val,
        'fo': fo,
        'userData': userData,
        'objective_type': 'MIN'
    }

    required_args_names = MH_ARG_MAP[mh]

    kwargs = {}
    for arg_name in required_args_names:
        if arg_name not in context:
            raise KeyError(f"Argumento '{arg_name}' requerido por {mh} no encontrado.")
        kwargs[arg_name] = context[arg_name]

    mh_function = metaheuristics[mh]
    try:
        result = mh_function(**kwargs)
    except TypeError as e:
        raise TypeError(f"Error al llamar a {mh}. Revisa MH_ARG_MAP['{mh}'].") from e

    new_population = None
    new_vel = vel
    posibles_mejoras = None
    updated_pBest = pBest

    if mh == 'TJO':
        if isinstance(result, tuple) and len(result) == 3:
            new_population, updated_fitness, updated_pBest = result
        else:
            raise TypeError(f"Retorno inesperado de TJO.")
    
    elif mh == 'LOA':
        if isinstance(result, tuple) and len(result) == 2:
            new_population, posibles_mejoras = result
        else:
             raise TypeError(f"Retorno inesperado de LOA.")

    elif isinstance(result, tuple) and len(result) == 2:
        new_population, new_vel = result

    elif isinstance(result, (np.ndarray, list)):
        new_population = result

    else:
        raise TypeError(f"Tipo de retorno inesperado de {mh}.")

    if not isinstance(new_population, np.ndarray):
       new_population = np.array(new_population)

    return new_population, new_vel, posibles_mejoras, updated_pBest


# ========== FUNCIÓN ESPECIALIZADA PARA MAPAS CAÓTICOS ==========

def binarize_and_evaluate_chaotic(mh, population, fitness, DS, best, matrixBin, instance, 
                                  repairType, pBest, pBestScore, posibles_mejoras, fo,
                                  chaotic_map, iter, pop_size, maxIter):
    """
    Versión ESPECIALIZADA para mapas caóticos.
    
     DIFERENCIAS CON LA VERSIÓN ESTÁNDAR:
    - Usa secuencias caóticas pregeneradas en lugar de np.random
    - Indexación: chaotic_index = (iter * pop_size * dim) + (i * dim)
    - Requiere chaotic_map (no es opcional)
    
    Args:
        mh (str): Metaheurística
        population (np.ndarray): Población actual
        fitness (np.ndarray): Fitness de la población
        DS (str): Función de transferencia-binarización (e.g., "S1-STD")
        best (np.ndarray): Mejor solución global
        matrixBin (np.ndarray): Matriz binaria de referencia
        instance: Instancia del problema (SCP/USCP)
        repairType (str): Tipo de reparación ('simple' o 'complex')
        pBest (np.ndarray): Mejor solución personal (PSO/TJO)
        pBestScore (np.ndarray): Fitness de pBest
        posibles_mejoras (np.ndarray): Soluciones candidatas (LOA)
        fo (callable): Función objetivo
        chaotic_map (np.ndarray): Secuencia caótica pregenerada (REQUERIDO)
        iter (int): Iteración actual
        pop_size (int): Tamaño de la población
        maxIter (int): Iteraciones máximas
    
    Returns:
        tuple: (population, fitness, pBest) actualizados
    """
    n_pop = len(population)
    dim = instance.getColumns()
    
    # Cachear métodos
    factibilityTest = instance.factibilityTest
    repair = instance.repair
    fitness_func = instance.fitness
    
    # ========== BINARIZACIÓN CON MAPAS CAÓTICOS ==========
    if mh != "GA":
        for i in range(n_pop):
            #  INDEXACIÓN CAÓTICA (fórmula del paper original)
            # Cada individuo en cada iteración tiene un índice único
            chaotic_index = ((iter - 1) * pop_size * dim) + (i * dim)
            
            # Aplicar binarización con valores caóticos
            population[i] = b.aplicarBinarizacion(
                population[i], 
                DS, 
                best, 
                matrixBin[i],
                chaotic_map=chaotic_map,
                chaotic_index=chaotic_index
            )
    
    # ========== REPARACIÓN Y EVALUACIÓN ==========
    for i in range(n_pop):
        flag, _ = factibilityTest(population[i])
        
        if not flag:
            population[i] = repair(population[i], repairType)
            
        fitness[i] = fitness_func(population[i])

        # Actualización de pBest (PSO/TJO)
        if mh in ['PSO', 'TJO']:
            if fitness[i] < pBestScore[i]:
                pBestScore[i] = fitness[i]
                pBest[i] = np.copy(population[i])
                
        # Manejo especial para LOA
        if mh == 'LOA' and posibles_mejoras is not None:
            _, fitn = fo(posibles_mejoras[i])
            
            if fitn < fitness[i]:
                population[i] = posibles_mejoras[i]
    
    return population, fitness, pBest


def update_best_solution_chaotic(population, fitness, best, bestFitness):
    """
    Idéntico al original. Actualiza mejor solución global.
    """
    solutionsRanking = np.argsort(fitness)
    
    if fitness[solutionsRanking[0]] < bestFitness:
        bestFitness = fitness[solutionsRanking[0]]
        best = population[solutionsRanking[0]]
    
    return best, bestFitness
