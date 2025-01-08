import math
import random
import numpy as np

# Whale Optimization Algorithm (WOA)
# doi.org/10.1016/j.advengsoft.2016.01.008

def iterarWOA(maxIter, t, dimension, population, bestSolution):
    a = 2 - ((2 * t) / maxIter)
    a = 2 - ((2 * t) / maxIter)
    b = 1
    
    
    for i in range(population.__len__()):
        #  p is a random number into [0,1]
        p = random.uniform(0.0, 1.0)
        # aplicacion de ecuacion 2.3
        r = random.uniform(0.0, 1.0)
        A = 2 * a * (r - a) 
        # aplicacion de ecuacion 2.4
        r = random.uniform(0.0, 1.0)
        C =  2 * r
        # l is a random number into [-1,1]
        l = random.uniform(-1.0, 1.0)
        # aplicacion de ecuacion 2.6
        
        if p < 0.5:
            if abs(A) < 1: 
                # aplicacion ecuacion de movimiento 2.1
                for j in range(dimension):
                    D = abs((C * bestSolution[j]) - population[i][j])
                    # aplicacion ecuacion de movimiento 2.2
                    population[i][j] = bestSolution[j] - ( A * D )
            
            else:
                randomPos = random.randint(0, population.__len__() - 1) # seleccionar un individuo al azar
                for j in range(dimension):
                    # aplicacion de ecuacion 2.7
                    D = abs((C * population[randomPos][j]) - population[i][j])
                    D = abs((C * population[randomPos][j]) - population[i][j])
                    # aplicacion de ecuacion 2.8
                    population[i][j] = population[randomPos][j] - (A * D)
        
        else:
            for j in range(dimension):
                # aplicacion de ecuacion de movimiento 2.5
                DPrima = bestSolution[j] - population[i][j]
                population[i][j] = (DPrima * math.exp(b * l) * math.cos(2 * math.pi * l) ) + bestSolution[j]
    
    return np.array(population)