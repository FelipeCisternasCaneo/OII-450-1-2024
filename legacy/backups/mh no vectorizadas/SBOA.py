import numpy as np
import math

# Secretary Bird Optimization Algorithm (SBOA)
# https://doi.org/10.1007/s10462-024-10729-y

def levy(dim):
    beta = 1.5
    
    sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
            (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    
    step = u / np.abs(v) ** (1 / beta)
    
    return step

def iterarSBOA(maxIter, iterActual, dim, population, fitness, bestSolution, function):
    if not isinstance(population, np.ndarray):
        population = np.array(population)
    
    N = len(population)
    fitness = np.array(fitness)
    best_position = np.array(bestSolution)

    CF = (1 - iterActual / maxIter) ** (2 * iterActual / maxIter)
    
    for i in range(N):
        if iterActual < maxIter / 3:  # Etapa de búsqueda de presas
            X_random_1 = np.random.randint(N)
            X_random_2 = np.random.randint(N)
            R1 = np.random.rand(dim)
            X1 = population[i] + (population[X_random_1] - population[X_random_2]) * R1
            
        elif maxIter / 3 <= iterActual < 2 * maxIter / 3:  # Etapa de aproximación
            RB = np.random.randn(dim)
            X1 = best_position + np.exp((iterActual / maxIter) ** 4) * (RB - 0.5) * (best_position - population[i])
            
        else:  # Etapa de ataque
            RL = 0.5 * levy(dim)
            X1 = best_position + CF * population[i] * RL

        X1, f_newP1 = function(X1)
        
        if f_newP1 <= fitness[i]:
            population[i] = X1
            fitness[i] = f_newP1

    # Estrategia de escape
    r = np.random.rand()
    k = np.random.randint(N)
    
    Xrandom = population[k]
    
    for i in range(N):
        if r < 0.5:
            RB = np.random.rand(dim) 
            X2 = best_position + (1 - iterActual / maxIter) ** 2 * (2 * RB - 1) * population[i]
            
        else:
            K = np.round(1 + np.random.rand())
            R2 = np.random.rand(dim) 
            X2 = population[i] + R2 * (Xrandom - K * population[i])

        X2, f_newP2 = function(X2)
        
        if f_newP2 <= fitness[i]:
            population[i] = X2
            fitness[i] = f_newP2

    return population