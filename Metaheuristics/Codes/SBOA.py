import numpy as np
import math

# Secretary Bird Optimization Algorithm (SBOA)
# https://doi.org/10.1007/s10462-024-10729-y

def levy(dim, beta=1.5):
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    return u / np.abs(v) ** (1 / beta)

def iterarSBOA(maxIter, iterActual, dim, population, fitness, bestSolution, function):
    best_position = np.asarray(bestSolution)
    N = len(population)

    CF = (1 - iterActual / maxIter) ** (2 * iterActual / maxIter)

    # Etapas: búsqueda de presas, aproximación, ataque
    if iterActual < maxIter / 3:  # Etapa de búsqueda de presas
        R1 = np.random.rand(N, dim)
        indices = np.random.randint(N, size=(N, 2))
        X1 = population + (population[indices[:, 0]] - population[indices[:, 1]]) * R1

    elif maxIter / 3 <= iterActual < 2 * maxIter / 3:  # Etapa de aproximación
        RB = np.random.randn(N, dim)
        exp_term = np.exp((iterActual / maxIter) ** 4)
        X1 = best_position + exp_term * (RB - 0.5) * (best_position - population)

    else:  # Etapa de ataque
        RL = 0.5 * levy(dim)
        X1 = best_position + CF * population * RL

    # Evaluación del fitness
    for i in range(N):
        X1[i], f_newP1 = function(X1[i])
        
        if f_newP1 <= fitness[i]:
            population[i] = X1[i]
            fitness[i] = f_newP1

    # Estrategia de escape
    r = np.random.rand()
    k = np.random.randint(N)
    Xrandom = population[k]

    RB = np.random.rand(N, dim)
    R2 = np.random.rand(N, dim)
    K = np.round(1 + np.random.rand(N))

    if r < 0.5:
        escape_term = (1 - iterActual / maxIter) ** 2 * (2 * RB - 1)
        X2 = best_position + escape_term * population
    
    else:
        X2 = population + R2 * (Xrandom - K[:, None] * population)

    # Evaluación del fitness en la estrategia de escape
    for i in range(N):
        X2[i], f_newP2 = function(X2[i])
        
        if f_newP2 <= fitness[i]:
            population[i] = X2[i]
            fitness[i] = f_newP2

    return population