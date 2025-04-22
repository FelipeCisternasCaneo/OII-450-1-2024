import numpy as np

# Reptile Search Algorithm (RSA)
# https://doi.org/10.1007/s11831-023-09990-1

def iterarRSA(maxIter, iter, dim, population, best, lb0, ub0):
    '''
    maxIter: Máximo de iteraciones 
    it: iteración actual
    dim: Dimensión de las soluciones
    population: población actual de soluciones
    best: Mejor individuo obtenido hasta ahora
    lb: Margen inferior
    ub: Margen superior
    '''
    # Parámetros
    alfa = 0.1
    beta = 0.1
    eps = 1e-10  # Pequeño valor para evitar divisiones por cero
    r3 = np.random.randint(-1, 2)  # -1, 0 o 1
    ES = 2 * r3 * (1 - (1 / maxIter))

    population = np.array(population)
    best = np.array(best)

    N = len(population)
    R = (best - population[np.random.randint(N, size=N)]) / (best + eps)
    P = alfa + (population - np.mean(population, axis=1, keepdims=True)) / (ub0 - lb0 + eps)
    Eta = best * P
    rand = np.random.random((N, dim))

    if iter < maxIter / 4:
        # Ecuación 1
        population = best - Eta * beta - R * rand
    elif iter < (2 * maxIter) / 4:
        # Ecuación 2
        r1_indices = np.random.randint(N, size=N)
        population = best * population[r1_indices] * ES * rand
    elif iter < (3 * maxIter) / 4:
        # Ecuación 3
        population = best * P * rand
    else:
        # Ecuación 4
        population = best - Eta * eps - R * rand

    return np.clip(population, lb0, ub0)
