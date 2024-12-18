import random
import numpy as np
import math

# Honey Badger Algorithm (HBA)
# https://doi.org/10.1016/j.matcom.2021.08.013

def iterarHBA(maxIter, it, dim, population, bestSolution, fitness, function, typeProblem): 
  C = 2
  beta = 6
  epsilon = 0.00000000000000022204
  pi = math.pi
  
  N = population.__len__()
  Xnew = np.zeros([N,dim])
  
  alpha = C * math.exp(-it / maxIter)
  
  for i in range(N):    
    r6 = random.uniform(0,1)
    
    if r6 <= 0.5: F = 1
    
    else: F = -1

    r = random.uniform(0, 1)
    
    if r < 0.5:
      r2 = random.uniform(0, 1)
      r3 = random.uniform(0, 1)
      r4 = random.uniform(0, 1)
      r5 = random.uniform(0, 1)
      
    else:
      r7 = random.uniform(0, 1)
    
    for j in range(dim):   
      di = bestSolution[j] - population[i][j]
      
      if r < 0.5:
        if i != N - 1: S = np.power((population[i][j] - population[i + 1][j]), 2)
        
        else: S = np.power((population[i][j] - population[0][j]), 2)
        
        I = r2 * S / (4 * pi * np.power(di + epsilon, 2))

        Xnew[i][j] = (bestSolution[j] + 
                      F * beta * I * bestSolution[j] + 
                      F * r3 * alpha * di * 
                      np.abs( math.cos(2 * pi * r4) * (1 - math.cos(2 * pi * r5)))
                      )
        
      else:
        Xnew[i][j] = bestSolution[j] + F * r7 * alpha * di
    
    Xnew[i], newFitness = function(Xnew[i])
    
    if typeProblem == 'MIN': condition = newFitness < fitness[i]
    
    elif typeProblem == 'MAX': condition = newFitness > fitness[i]
    
    if condition: population[i] = Xnew[i]

  return np.array(population)