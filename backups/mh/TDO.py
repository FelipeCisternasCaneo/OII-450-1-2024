import numpy as np

# Tasmanian Devil Optimization (TDO)
# https://doi.org/10.1109/ACCESS.2022.3151641

def iterarTDO(maxIter, it, dim, population, fitness, function, typeProblem):
    N = len(population)
    population = np.array(population)
    fitness = np.array(fitness)
    
    R = 0.01 * (1 - (it / maxIter))  # Parámetro para explotación

    # Selección aleatoria de demonios
    k = np.random.choice(N, size=N, replace=True)
    CP = population[k]  # Selecciona un demonio aleatorio para cada individuo

    # Matriz para nuevas posiciones
    xNew = np.copy(population)

    # Verifica condiciones para acercarse o alejarse
    evaluated_CP = np.array([function(ind)[1] for ind in CP])
    if typeProblem == 'MIN':
        condition = evaluated_CP < fitness
    else:
        condition = evaluated_CP > fitness

    # Actualización de posiciones (Acercarse o Alejarse)
    I = np.random.randint(1, 3, size=(N, 1))  # I es aleatorio en [1, 2]
    random_factors = np.random.uniform(0.0, 1.0, size=(N, dim))

    xNew[condition] += random_factors[condition] * (CP[condition] - I[condition] * population[condition])
    xNew[~condition] += random_factors[~condition] * (population[~condition] - CP[~condition])

    # Evalúa todas las nuevas posiciones y separa resultados
    xNew_results = [function(ind) for ind in xNew]
    xNew_solutions = np.array([result[0] for result in xNew_results])  # Extrae las soluciones
    xNew_fitness = np.array([result[1] for result in xNew_results])    # Extrae los valores de fitness

    # Actualiza la población si la nueva solución es mejor
    if typeProblem == 'MIN':
        improvement = xNew_fitness < fitness
    else:
        improvement = xNew_fitness > fitness

    population[improvement] = xNew_solutions[improvement]
    fitness[improvement] = xNew_fitness[improvement]

    xLocal = population + (2 * np.random.uniform(0.0, 1.0, size=(N, dim)) - 1) * R * xNew
    xLocal_results = [function(ind) for ind in xLocal]
    xLocal_solutions = np.array([result[0] for result in xLocal_results])  # Extrae soluciones locales
    xLocal_fitness = np.array([result[1] for result in xLocal_results])    # Extrae fitness local

    # Actualiza donde mejora la explotación local
    if typeProblem == 'MIN':
        exploitation_improvement = xLocal_fitness < fitness
    else:
        exploitation_improvement = xLocal_fitness > fitness

    population[exploitation_improvement] = xLocal_solutions[exploitation_improvement]
    fitness[exploitation_improvement] = xLocal_fitness[exploitation_improvement]

    return population
