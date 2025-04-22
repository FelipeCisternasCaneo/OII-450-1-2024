import random
import numpy as np

# Pendulum Search Algorithm (PSA)
# https://doi.org/10.3390/a15060214

def iterarPSA(maxIter, iter, dim, population, best):
    for i in range(population.__len__()):
        for j in range(dim):
            rand = random.random()
            pend = 2 * np.exp(-iter / maxIter) * (np.cos( 2 * np.pi * rand))
            population[i][j] = population[i][j] + (pend * (best[j] - population[i][j]))

    return np.array(population)