import numpy as np

from Discretization import discretization as b
from Metaheuristics.imports import metaheuristics, MH_ARG_MAP

def initialize_population(mh, pop, instance):
    vel, pBestScore, pBest = None, None, None
    
    if mh == 'PSO':
        vel = np.zeros((pop, instance.getColumns()))
        pBestScore = np.full(pop, float("inf"))  # Más directo
        pBest = np.zeros((pop, instance.getColumns()))
    
    # Genero una población inicial binaria, esto ya que nuestro problema es binario
    population = np.random.randint(low = 0, high = 2, size = (pop, instance.getColumns()))
    
    return population, vel, pBestScore, pBest

def evaluate_population(mh, population, fitness, instance, pBest, pBestScore, repairType):
    # Calculo de factibilidad de cada individuo y calculo del fitness inicial
    for i in range(population.__len__()):
        flag, _ = instance.factibilityTest(population[i])
        
        if not flag: #solucion infactible
            population[i] = instance.repair(population[i], repairType)
            
        fitness[i] = instance.fitness(population[i])
        
        if mh == 'PSO':
            if pBestScore[i] > fitness[i]:
                pBestScore[i] = fitness[i]
                pBest[i, :] = population[i, :].copy()
        
    solutionsRanking = np.argsort(fitness) # rankings de los mejores fitnes
    bestRowAux = solutionsRanking[0] # DETERMINO MI MEJOR SOLUCION Y LA GUARDO 
    best = population[bestRowAux].copy()
    bestFitness = fitness[bestRowAux]
    
    return fitness, best, bestFitness, pBest, pBestScore

def iterate_population_scp(mh, population, iter, maxIter, instance, fitness, best,
                           vel=None, pBest=None, fo=None, param=None):
    """
    Itera sobre la población para SCP usando la metaheurística especificada ('mh'),
    construyendo los argumentos dinámicamente basados en MH_ARG_MAP.
    Maneja casos especiales como PO y GA.
    """
    # --- Manejo especial para PO ---
    if mh == 'PO':
        return np.array(population), vel, None

    # --- Manejo especial para GA ---
    if mh == 'GA':
        if param is None:
            raise ValueError("Parámetros 'cross' y 'muta' no proporcionados para GA.")
        
        partes = param.split(";")
        cross = float(partes[0])
        muta = float(partes[1].split(":")[1])

        new_population = metaheuristics['GA'](population=population, fitness=fitness, cross=cross, muta=muta)
        # Asegurar que sea array numpy
        if not isinstance(new_population, np.ndarray):
            new_population = np.array(new_population)
            
        return new_population, vel, None
    
    # --- Mapeo específico para HLOA ---
    # Si se llama con 'HLOA', usar la versión SCP
    if mh == 'HLOA':
        mh = 'HLOA_SCP'
    
    # --- Verificaciones esenciales (para MHs no especiales) ---
    if mh not in metaheuristics:
        raise ValueError(f"Metaheurística '{mh}' no encontrada en 'metaheuristics' (Metaheuristics/imports.py).")
    if mh not in MH_ARG_MAP:
        raise ValueError(f"Mapa de argumentos MH_ARG_MAP para '{mh}' no definido (Metaheuristics/imports.py).")

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
        'ub': ub_arr,           # Array de 1s
        'lb': lb_arr,           # Array de 0s
        'ub0': ub0_val,         # Escalar 1
        'lb0': lb0_val,         # Escalar 0
        'fo': fo,
        'objective_type': 'MIN'
    }

    # 2. Obtener los nombres de los argumentos requeridos para esta MH
    required_args_names = MH_ARG_MAP[mh]

    # 3. Construir diccionario 'kwargs' solo con los argumentos necesarios
    kwargs = {}
    for arg_name in required_args_names:
        if arg_name not in context:
            raise KeyError(f"Error Interno: Argumento '{arg_name}' requerido por {mh} (según MH_ARG_MAP) no encontrado en 'context'.")
        kwargs[arg_name] = context[arg_name]

    mh_function = metaheuristics[mh]
    try:
        # print(f"Iter {iter}: Llamando a {mh} con args: {list(kwargs.keys())}") # Debug
        result = mh_function(**kwargs)
    except TypeError as e:
        raise TypeError(f"Error de tipo al llamar a la función para {mh}. Revisa MH_ARG_MAP['{mh}'] y la definición de la función.") from e

    new_population = None
    new_vel = vel
    posibles_mejoras = None

    if mh == 'LOA':
        if isinstance(result, tuple) and len(result) == 2:
            new_population, posibles_mejoras = result
        else:
             raise TypeError(f"Retorno inesperado de LOA (SCP). Se esperaba (population, posibles_mejoras), se obtuvo {type(result)}")

    elif isinstance(result, tuple) and len(result) == 2:
        new_population, new_vel = result

    elif isinstance(result, (np.ndarray, list)):
        new_population = result

    else:
        raise TypeError(f"Tipo de retorno inesperado de {mh} (SCP): {type(result)}. Se esperaba ndarray, list o tupla.")

    if not isinstance(new_population, np.ndarray):
       new_population = np.array(new_population)

    return new_population, new_vel, posibles_mejoras

def binarize_and_evaluate(mh, population, fitness, DS, best, matrixBin, instance, repairType, pBest, pBestScore, posibles_mejoras, fo):
    # Binarizo, calculo de factibilidad de cada individuo y calculo del fitness
    for i in range(population.__len__()):

        if mh != "GA":
            population[i] = b.aplicarBinarizacion(population[i], DS, best, matrixBin[i])

        flag, _ = instance.factibilityTest(population[i])
        
        if not flag: #solucion infactible
            population[i] = instance.repair(population[i], repairType)
            
        fitness[i] = instance.fitness(population[i])

        if mh == 'PSO':
            if fitness[i] < pBestScore[i]:
                pBest[i] = np.copy(population[i])
                
        if mh == 'LOA':
            _, fitn = fo(posibles_mejoras[i])
            
            if fitn < fitness[i]:
                population[i] = posibles_mejoras[i]
    
    return population, fitness, pBest

def update_best_solution(population, fitness, best, bestFitness):
    # Genero un vector de donde tendré mis soluciones rankeadas
    solutionsRanking = np.argsort(fitness) # rankings de los mejores fitness
    
    # conservo el best
    if fitness[solutionsRanking[0]] < bestFitness:
        bestFitness = fitness[solutionsRanking[0]]
        best = population[solutionsRanking[0]]
    
    return best, bestFitness