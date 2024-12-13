import random
import numpy as np

# Reptile Search Algorithm (RSA)
# https://doi.org/10.1007/s11831-023-09990-1

def iterarRSA(maxIter, it, dim, population, bestSolution, LB, UB):
    '''
    maxIter: Máximo de iteraciones 
    it: iteración actual
    dim: Dimensión de las soluciones
    population: population actual de soluciones
    bestSolution: Mejor individuo obtenido hasta ahora
    LB: Margen inferior
    UB: Margen superior
    '''
    # PARAM
    alfa = 0.1
    beta = 0.1
    
    # Small value epsilon
    eps = 1e-10
    
    # Actualización de valor ES
    r3 = random.randint(-1, 1) # r3 denotes to a random integer number between −1 and 1, pag4
    ES = 2 * r3 * (1 - (1 / maxIter))

    # Pob size
    N = len(population)

    # for de población
    for i in range(N):
        # for de dimensión
        for j in range(dim):
            # actualización de valores de la metaheurística
            r2 = random.randint(0, N - 1)
            R = (bestSolution[j] - population[r2][j]) / (bestSolution[j] + eps)
            P = alfa + (population[i][j] - np.mean(population[i])) / (UB - LB + eps)
            Eta = bestSolution[j] * P
            rand = random.random()
            
            # ecuaciones de movimiento

            # ec1
            if it < maxIter / 4:
                population[i][j] = bestSolution[j] - Eta * beta - R * rand
            # ec2
            
            elif it < (2 * maxIter) / 4 and it >= maxIter / 4:
                r1 = random.randint(0, N - 1)
                population[i][j] = bestSolution[j] * population[r1][j] * ES * rand
            # ec3
            
            elif it < (maxIter * 3) / 4 and it >= (2 * it) / 4:
                population[i][j] = bestSolution[j] * P * rand
            # ec4
            
            else:
                population[i][j] = bestSolution[j] - Eta * eps - R * rand
        # fin for dim

    return np.array(population)