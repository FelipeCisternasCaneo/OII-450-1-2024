import random
import numpy as np

# Particle Swarm Optimization


def obtenerRandom(chaotic_map):
    return random.choice(chaotic_map)
    

def iterarPSO(maxIter, it, dim, population, gBest, pBest, vel, ub):
    '''
    maxIter: Máximo de iteraciones 
    it: iteración actual
    dim: Dimensión de las soluciones
    population: population actual de soluciones
    bestSolution: Mejor individuo obtenido hasta ahora
    bestPop: Mejores partículas obtenidas hasta ahora
    '''

    Vmax = ub * 0.1
    wMax = 0.9
    wMin = 0.1
    c1 = 2
    c2 = 2
    
    # Vmax = 6
    # wMax = 0.9
    # wMin = 0.2
    # c1 = 2
    # c2 = 2

    # Update the W of PSO
    w = wMax - it * ((wMax - wMin) / maxIter)
    
    #For de población
    for i in range(population.__len__()):
        #For de dimensión
        for j in range(dim):
            r1 = random.random()
            r2 = random.random()
            #actualización de la velocidad de las partículas
            vel[i, j] = (
                w * vel[i, j]
                + c1 * r1 * (pBest[i][j] - population[i][j])
                + c2 * r2 * (gBest[j] - population[i][j])
            )

            #Se mantiene la velocidad en sus márgenes mínimos y máximos
            if vel[i, j] > Vmax:
                vel[i, j] = Vmax

            if vel[i, j] < -Vmax:
                vel[i, j] = -Vmax

            #se actualiza la población utilizando las velocidades calculadas
            population[i][j] = population[i][j] + vel[i][j]
    return np.array(population), np.array(vel)