import numpy as np

# Pufferfish Optimization Algorithm (POA)
# https://doi.org/10.3390/biomimetics9020065

def actualizarPosicionP1(i, X, SP, r, I):
    # Actualiza la posición del individuo i en la fase 1 (exploración)
    X_new = np.copy(X[i])
    if SP is not None:
        X_new = X[i] + r * (X[SP] - I * X[i])
    return X_new

def actualizarPosicionP2(individual, r, t, lb, ub):
    # Actualiza la posición del individuo en la fase 2 (explotación)
    return individual + (1 - 2 * r) * ((ub - lb) / (t + 1))

def iterarPOA(maxIter, iter, dim, population, fitness, function, lb, ub, typeProblem):
    population = np.array(population)
    fitness = np.array(fitness)
    
    N = population.shape[0]  # Tamaño de la población

    # Vamos a iterar sobre cada individuo de la población
    for i in range(N):
        # Fase 1: Ataque del depredador hacia el pez globo (fase de exploración)

        # Determina los peces globo candidatos para el individuo i
        if typeProblem == 'MIN':
            CP = np.where((fitness < fitness[i]) & (np.arange(N) != i))[0]
        
        elif typeProblem == 'MAX':
            CP = np.where((fitness > fitness[i]) & (np.arange(N) != i))[0]

        # Selecciona el pez globo candidato para el individuo
        SP = np.random.choice(CP) if len(CP) > 0 else None
    
        r = np.random.rand(dim)
        I = np.random.choice([1, 2], size=dim)

        # Probar nueva posición fase 1
        newValuesP1 = actualizarPosicionP1(i, population, SP, r, I)
        newValuesP1, newFitnessP1 = function(newValuesP1)
        
        # Verifica si la nueva posición mejora el fitness (dependiendo de si es minimización o maximización)
        if (typeProblem == 'MIN' and newFitnessP1 < fitness[i]) or (typeProblem == 'MAX' and newFitnessP1 > fitness[i]):
            population[i] = newValuesP1  # Actualizamos la posición si mejora
            fitness[i] = newFitnessP1  # Actualizamos el fitness también

        # Fase 2: Mecanismo de defensa del pez globo contra los depredadores (fase de explotación)
        r = np.random.rand(dim)
        
        # Probar nueva posición fase 2
        newValuesP2 = actualizarPosicionP2(population[i], r, iter + 1, lb, ub)
        newValuesP2, newFitnessP2 = function(newValuesP2)

        # Verifica si la nueva posición en fase 2 mejora el fitness
        if (typeProblem == 'MIN' and newFitnessP2 < fitness[i]) or (typeProblem == 'MAX' and newFitnessP2 > fitness[i]):
            population[i] = newValuesP2  # Actualizamos la posición si mejora
            fitness[i] = newFitnessP2  # Actualizamos el fitness también

    return population