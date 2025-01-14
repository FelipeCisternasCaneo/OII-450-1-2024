import numpy as np
import random

from Util import util

# Grey Wolf Optimizer (GWO)
# https://doi.org/10.1016/j.advengsoft.2013.12.007

def iterarGWO(maxIter, t, dimension, population, fitness, typeProblem):
    a = 2 - t * ((2) / maxIter) # a decreases linearly fron 2 to 0
    sortedPositions = util.selectionSort(fitness)
    Xalfa  = []
    Xbeta  = []
    Xdelta = []
    # eq. 3.6
    
    if typeProblem == "MIN":
        Xalfa  = population[sortedPositions[0]]
        Xbeta  = population[sortedPositions[1]]
        Xdelta = population[sortedPositions[2]]
        
    if typeProblem == "MAX":
        Xalfa  = population[sortedPositions[population.__len__()-1]]
        Xbeta  = population[sortedPositions[population.__len__()-2]]
        Xdelta = population[sortedPositions[population.__len__()-3]]
        
    for i in range(population.__len__()):
        for j in range(dimension):
            # r1 is a random number in [0,1]
            r1 = random.uniform(0.0, 1.0)
            # r2 is a random number in [0,1]
            r2 = random.uniform(0.0, 1.0)
            # aplicacion de la ecuacion 3.3
            A1 = 2 * a * r1 - a
            # aplicacion de la ecuacion 3.4
            C1 = 2 * r2
            # aplicacion de la ecuacion 3.5
            dalfa = abs((C1 * Xalfa[j]) - population[i][j])
            # aplicacion de la ecuacion 3.6
            X1 = Xalfa[j] - (A1 * dalfa)
            # r1 is a random number in [0,1]
            r1 = random.uniform(0.0, 1.0)
            # r2 is a random number in [0,1]
            r2 = random.uniform(0.0, 1.0)
            # aplicacion de la ecuacion 3.3
            A2 = 2 * a * r1 - a
            # aplicacion de la ecuacion 3.4
            C2 = 2 * r2
            # aplicacion de la ecuacion 3.5
            dbeta = abs((C2 * Xbeta[j]) - population[i][j])
            # aplicacion de la ecuacion 3.6
            X2 = Xbeta[j] - (A2 * dbeta)
            # r1 is a random number in [0,1]
            r1 = random.uniform(0.0, 1.0) 
            # r2 is a random number in [0,1]
            r2 = random.uniform(0.0, 1.0) 
            # aplicacion de la ecuacion 3.3
            A3 = 2 * a * r1 - a
            # aplicacion de la ecuacion 3.4
            C3 = 2 * r2
            # aplicacion de la ecuacion 3.5
            ddelta = abs((C3*Xdelta[j]) - population[i][j])
            # aplicacion de la ecuacion 3.6
            X3 = Xdelta[j] - (A3 * ddelta) 
            # aplicacion de la ecuacion 3.7
            population[i][j] = (X1 + X2 + X3) / 3
            
    return np.array(population)