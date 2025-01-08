import numpy as np

# Reptile Search Algorithm (RSA)
# https://doi.org/10.1007/s11831-023-09990-1

def iterarRSA(maxIter, it, dim, population, bestSolution, LB, UB):
    '''
    maxIter: Máximo de iteraciones 
    it: iteración actual
    dim: Dimensión de las soluciones
    population: población actual de soluciones
    bestSolution: Mejor individuo obtenido hasta ahora
    LB: Margen inferior
    UB: Margen superior
    '''
    # Parámetros
    alfa = 0.1
    beta = 0.1
    eps = 1e-10  # Pequeño valor para evitar divisiones por cero
    r3 = np.random.randint(-1, 2)  # -1, 0 o 1
    ES = 2 * r3 * (1 - (1 / maxIter))

    population = np.array(population)
    bestSolution = np.array(bestSolution)

    N = len(population)
    R = (bestSolution - population[np.random.randint(N, size=N)]) / (bestSolution + eps)
    P = alfa + (population - np.mean(population, axis=1, keepdims=True)) / (UB - LB + eps)
    Eta = bestSolution * P
    rand = np.random.random((N, dim))

    if it < maxIter / 4:
        # Ecuación 1
        population = bestSolution - Eta * beta - R * rand
    elif it < (2 * maxIter) / 4:
        # Ecuación 2
        r1_indices = np.random.randint(N, size=N)
        population = bestSolution * population[r1_indices] * ES * rand
    elif it < (3 * maxIter) / 4:
        # Ecuación 3
        population = bestSolution * P * rand
    else:
        # Ecuación 4
        population = bestSolution - Eta * eps - R * rand

    return np.clip(population, LB, UB)
