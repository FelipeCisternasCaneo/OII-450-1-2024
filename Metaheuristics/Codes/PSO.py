import random
import numpy as np

def iterarPSO(maxIter, it, dim, population, bestSolution,bestPop):
    '''
    maxIter: Máximo de iteraciones 
    it: iteración actual
    dim: Dimensión de las soluciones
    population: population actual de soluciones
    bestSolution: Mejor individuo obtenido hasta ahora
    bestPop: Mejores partículas obtenidas hasta ahora
    '''

    Vmax = 5000
    wMax = 0.9
    wMin = 0.2
    c1 = 2
    c2 = 2

    vel = np.zeros((population.__len__(), dim))
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
                + c1 * r1 * (bestPop[i][j] - population[i][j])
                + c2 * r2 * (bestSolution[j] - population[i][j])
            )

            #Se mantiene la velocidad en sus márgenes mínimos y máximos
            if vel[i, j] > Vmax:
                vel[i, j] = Vmax

            if vel[i, j] < -Vmax:
                vel[i, j] = -Vmax
            
            #se actualiza la población utilizando las velocidades calculadas
            population[i][j] = population[i][j] + vel[i][j]


    return np.array(population)