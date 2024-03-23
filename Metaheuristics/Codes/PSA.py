#PSA
import random
import numpy as np
def iterarPSA(maxIter, t, dimension, population, bestSolution):
    for i in range(population.__len__()):
        for j in range(dimension):
            rand = random.random()
            pend = 2 * np.exp( -t / maxIter ) * ( np.cos( 2 * np.pi * rand ) )
            population[i][j] = population[i][j] + ( pend * ( bestSolution[j] - population[i][j] ) )
    return np.array(population)