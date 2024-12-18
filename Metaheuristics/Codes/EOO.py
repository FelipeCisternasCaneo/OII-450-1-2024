import random
import numpy as np

# Eurasian Oystercatcher Optimiser (EOO)
# doi.org/10.1515/jisys-2022-0017

def iterarEOO(maxIter, it, population, bestSolution):
    population = np.array(population, dtype = np.float64)
    bestSolution = np.array(bestSolution, dtype = np.float64)

    it = maxIter - it + 1
    n = population.__len__()

    for i in range(n):
        L = random.uniform(3, 5)
        T = (((L - 5)/(5 - 3)) * 10) - 5
        E = ((it - 1)/(n - 1)) - 0.5 if it > 1 else (1 / (n - 1)) - 0.5
        C = (((L-3)/(5-3)) * 2 ) + 0.6
        
        r = random.uniform(0, 1)
        Y = T + E + L * r * (bestSolution - population[i])
        population[i] *= C
        population[i] += Y

    return population