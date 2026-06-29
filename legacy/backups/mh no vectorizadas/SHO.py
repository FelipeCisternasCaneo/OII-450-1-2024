import numpy as np
import random
import math

# Sea-Horse Optimizer (SHO)
# https://doi.org/10.1007/s10489-022-03994-3

def levyFunction():
    lambd = 1.5
    s = 0.01
    
    w = random.uniform(0, 1)
    k = random.uniform(0, 1)

    gamma1 = math.gamma(lambd + 1)
    sin = math.sin(math.pi * lambd)
    gamma2 = math.gamma((lambd + 1) / 2)
    sigma = (gamma1 * sin) / (gamma2 * lambd * (2 ** ((lambd - 1) / 2)))

    return s * ((w * sigma) / abs(pow(k, (1 / lambd))))

def iterarSHO(maxIter, it, dim, population, bestSolution, function, typeProblem):
    population = np.array(population)
    bestSolution = np.array(bestSolution)
    N = population.shape[0]
    
    u = 0.05
    v = 0.05
    l = 0.05

    beta = np.random.randn(N, dim)
    r1 = np.random.randn(dim)

    for i in range(N):
        for j in range(dim):
            if r1[j] > 0:
                levy = levyFunction()
                theta = random.uniform(0, 2 * math.pi)   
                p = u * np.exp(theta * v)
                x = p * math.cos(theta)
                y = p * math.sin(theta)
                z = p * theta
                population[i, j] = (population[i, j] +
                                    levy * (bestSolution[j] - population[i, j]) *
                                    x * y * z +
                                    bestSolution[j])
            
            else:
                rand = random.uniform(0, 1)
                population[i, j] = (population[i, j] +
                                    rand * l * beta[i, j] *
                                    (population[i, j] - beta[i, j] * bestSolution[j]))
        
        population[i], _ = function(population[i])
        
    alpha = (1 - it / maxIter) ** ((2 * it) / maxIter)
    fitness = np.zeros(N)
    
    for i in range(N):
        for j in range(dim):
            r2 = random.uniform(0, 1)
            rand = random.uniform(0, 1)             

            if r2 > 0.1:
                population[i, j] = alpha * (bestSolution[j] - rand * population[i, j])
                
            else:
                population[i, j] = ((1 - alpha) * (population[i, j] -
                                    rand * bestSolution[j]) +
                                    alpha * population[i, j])
        
        population[i], fitness[i] = function(population[i])
        
    if typeProblem == 'MIN':
        sortIndex = np.argsort(fitness)
        
    elif typeProblem == 'MAX':
        sortIndex = np.argsort(fitness)[::-1]

    father = population[sortIndex[:N // 2]]
    mother = population[sortIndex[N // 2:]]

    offspring = np.zeros((N // 2, dim))
    fitnessOffspring = np.zeros(N // 2)
    
    for k in range(N // 2):
        r3 = np.random.rand()
        
        for j in range(dim):
            offspring[k, j] = r3 * father[k, j] + (1 - r3) * mother[k, j]
        
        offspring[k], fitnessOffspring[k] = function(offspring[k])

    newFitness = np.concatenate((fitness, fitnessOffspring))
    newPopulation = np.concatenate((population, offspring))
        
    if typeProblem == 'MIN':
        sortIndex = np.argsort(newFitness)
        
    elif typeProblem == 'MAX':
        sortIndex = np.argsort(newFitness)[::-1]

    population = newPopulation[sortIndex[:N]]
    
    return population