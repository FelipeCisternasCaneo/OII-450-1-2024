import numpy as np

from Diversity.imports import diversidadHussain
from Problem.Benchmark.Problem import fitness as f
from Metaheuristics.imports import metaheuristics, MH_ARG_MAP

def initialize_population(mh, pop, dim, lb, ub):
    vel, pBestScore, pBest = None, None, None
    
     # Inicialización de Memoria (pBest) - Común para PSO y TJO
    if mh in ['PSO', 'TJO']:
        pBestScore = np.full(pop, np.inf)
        pBest = np.zeros((pop, dim))
    
    # Inicialización de Velocidad (vel) - Exclusivo para PSO
    if mh == 'PSO':
        vel = np.zeros((pop, dim))
    
    # Generación de la población inicial (Común para todos)
    lb = np.array(lb)
    ub = np.array(ub)
    population = np.random.uniform(0, 1, (pop, dim)) * (ub - lb) + lb
    
    return population, vel, pBestScore, pBest

def evaluate_population(mh, population, fitness, _, lb, ub, function, nfe_counter):
    """Evalúa fitness inicial de la población."""
    pBest, pBestScore = None, None
    
    if mh == 'PSO':
        pBest = np.zeros_like(population)
        pBestScore = np.full(population.shape[0], float("inf"))
    
    # ========== INICIALIZACIÓN PARA TJO ==========
    if mh == 'TJO':
        pBest = np.zeros_like(population)  # FlockMemoryX
    # ============================================
    
    for i in range(population.shape[0]):
        population[i] = np.clip(population[i], lb, ub)
        fitness[i] = f(function, population[i])
        nfe_counter[0] += 1
        
        if mh == 'PSO' and pBestScore[i] > fitness[i]:
            pBestScore[i] = fitness[i]
            pBest[i] = population[i].copy()
        
        # ========== INICIALIZACIÓN PARA TJO ==========
        if mh == 'TJO':
            pBest[i] = population[i].copy()  # FlockMemoryX = x inicial
        # ============================================

    solutionsRanking = np.argsort(fitness)
    bestIndex = solutionsRanking[0]
    bestFitness = fitness[bestIndex]
    best = population[bestIndex].copy()
    
    return fitness, best, bestFitness, pBest, pBestScore

def update_population(population, fitness, _, lb, ub, function, best, bestFitness, 
                     pBest=None, pBestScore=None, mh=None, posibles_mejoras=None, nfe_counter=None):
    """Actualiza población y evalúa fitness."""
    
    # ========== EXCEPCIÓN PARA CLO Y TJO ==========
    # CLO y TJO ya evaluaron y seleccionaron dentro de sus funciones
    if mh in ['CLO', 'TJO']:
        # Solo actualizar el mejor global
        bestIndex = np.argmin(fitness)
        if fitness[bestIndex] < bestFitness:
            bestFitness = fitness[bestIndex]
            best = population[bestIndex].copy()
        
        div_t = diversidadHussain(population)
        return population, fitness, best, bestFitness, div_t
    # ==============================================
    
    # Para el resto de algoritmos continuar normal...
    population = np.clip(population, lb, ub)

    for i in range(population.shape[0]):
        fitness[i] = f(function, population[i])
        nfe_counter[0] += 1

    if mh == 'LOA' and posibles_mejoras is not None:
        posibles_mejoras = np.clip(posibles_mejoras, lb, ub)
        
        for i in range(posibles_mejoras.shape[0]):
            mejora_fitness = f(function, posibles_mejoras[i])
            nfe_counter[0] += 1  # ← Y esto (evaluación extra de LOA)
            
            if mejora_fitness < fitness[i]:
                population[i] = posibles_mejoras[i]
                fitness[i] = mejora_fitness

    if mh == 'PSO' or mh == 'TJO':
        for i in range(population.shape[0]):
            
            if fitness[i] < pBestScore[i]:
                pBestScore[i] = fitness[i]
                pBest[i] = population[i]

    bestIndex = np.argmin(fitness)
    
    if fitness[bestIndex] < bestFitness:
        bestFitness = fitness[bestIndex]
        best = population[bestIndex].copy()

    div_t = diversidadHussain(population)

    return population, fitness, best, bestFitness, div_t

def iterate_population(mh, population, iter, maxIter, dim, fitness, best, vel=None, pBest=None, ub=None, lb=None, fo=None, userData=None):
    """
    Itera sobre la población usando la metaheurística especificada ('mh'),
    construyendo los argumentos dinámicamente basados en MH_ARG_MAP.
    """
    # --- Manejo especial para PO ---
    if mh == 'PO':
        return np.array(population), vel, None

    if mh == 'HLOA':
        mh = 'HLOA_BEN'
    
    # --- Verificaciones ---
    if mh not in metaheuristics:
        raise ValueError(f"""Metaheurística '{mh}' no encontrada en el módulo de metaheurísticas.""")
    if mh not in MH_ARG_MAP:
        raise ValueError(f"""Metaheurística '{mh}' no encontrada en el mapa de argumentos requeridos (MH_ARG_MAP).""")

    # --- Preparar lb0 y ub0 ---
    lb0_val = None
    ub0_val = None
    if lb is not None and len(lb) > 0: lb0_val = lb[0]
    if ub is not None and len(ub) > 0: ub0_val = ub[0]

    context = {
        'maxIter': maxIter,
        'iter': iter,
        'dim': dim,
        'population': population,
        'fitness': fitness,
        'best': best,
        'vel': vel,
        'pBest': pBest,
        'ub': ub,
        'lb': lb,
        'ub0': ub0_val,
        'lb0': lb0_val,
        'fo': fo,
        'objective_type': 'MIN',
        'userData': userData,
    }
    
    if userData:
        context.update(userData)

    required_args_names = MH_ARG_MAP[mh]
    
    kwargs = {}
    for arg_name in required_args_names:
        if arg_name not in context:
            # Si falta un argumento esperado en el contexto, es un error interno
            raise KeyError(f"Error Interno: El argumento '{arg_name}' requerido por {mh} (según MH_ARG_MAP) no se encontró en el diccionario 'context' de iterate_population.")

        kwargs[arg_name] = context[arg_name]

    mh_function = metaheuristics[mh]
    try:
        # print(f"Iter {iter}: Llamando a {mh} con args: {list(kwargs.keys())}") # Debug
        result = mh_function(**kwargs)
    except TypeError as e:
        raise TypeError(f"Error de tipo al llamar a {mh}. Revisa MH_ARG_MAP['{mh}'].") from e

    new_population = None
    new_vel = None
    posibles_mejoras = None

    if mh == 'LOA':
        if isinstance(result, tuple) and len(result) == 2:
            new_population, posibles_mejoras = result
            new_vel = vel
        else:
             raise TypeError(f"Retorno inesperado de {mh}. Se esperaba (population, posibles_mejoras), se obtuvo {type(result)}")
         
    elif mh == 'GOAT':
        if isinstance(result, tuple) and len(result) == 3:
            new_population, fitness, best = result
            new_vel = vel
        else:
            raise TypeError(f"Retorno inesperado de GOAT. Se esperaba (population, fitness, best)")
    
    elif mh == 'TJO':
        # TJO retorna (población, fitness, pBest)
        if isinstance(result, tuple) and len(result) == 3:
            new_population, fitness, pBest = result
            new_vel = vel
        else:
            raise TypeError(f"Retorno inesperado de TJO. Se esperaba (population, fitness, pBest)")
    
    elif mh in ['APO', 'CLO']:
        # APO y CLO solo retornan población, no modifican vel
        if isinstance(result, np.ndarray):
            new_population = result
            new_vel = vel
        else:
            raise TypeError(f"Retorno inesperado de {mh}. Se esperaba np.ndarray")
    
    elif isinstance(result, tuple) and len(result) == 2:
        new_population, new_vel = result
        
    elif isinstance(result, (np.ndarray, list)):
        new_population = result
        new_vel = vel
    
    else:
        raise TypeError(f"Tipo de retorno inesperado de la metaheurística {mh}: {type(result)}")

    if not isinstance(new_population, np.ndarray):
        new_population = np.array(new_population)

    return new_population, new_vel, posibles_mejoras