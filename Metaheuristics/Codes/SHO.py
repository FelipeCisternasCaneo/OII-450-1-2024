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

def iterarSHO(maxIter, iter, dim, population, best, fo, objective_type):
    population = np.array(population)
    best = np.array(best)
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
                                    levy * (best[j] - population[i, j]) *
                                    x * y * z +
                                    best[j])
            
            else:
                rand = random.uniform(0, 1)
                population[i, j] = (population[i, j] +
                                    rand * l * beta[i, j] *
                                    (population[i, j] - beta[i, j] * best[j]))
        
        population[i], _ = fo(population[i])
        
    alpha = (1 - iter / maxIter) ** ((2 * iter) / maxIter)
    fitness = np.zeros(N)
    
    for i in range(N):
        for j in range(dim):
            r2 = random.uniform(0, 1)
            rand = random.uniform(0, 1)             

            if r2 > 0.1:
                population[i, j] = alpha * (best[j] - rand * population[i, j])
                
            else:
                population[i, j] = ((1 - alpha) * (population[i, j] -
                                    rand * best[j]) +
                                    alpha * population[i, j])
        
        population[i], fitness[i] = fo(population[i])
        
    if objective_type == 'MIN':
        sortIndex = np.argsort(fitness)
        
    elif objective_type == 'MAX':
        sortIndex = np.argsort(fitness)[::-1]

    father = population[sortIndex[:N // 2]]
    mother = population[sortIndex[N // 2:]]

    offspring = np.zeros((N // 2, dim))
    fitnessOffspring = np.zeros(N // 2)
    
    for k in range(N // 2):
        r3 = np.random.rand()
        
        for j in range(dim):
            offspring[k, j] = r3 * father[k, j] + (1 - r3) * mother[k, j]
        
        offspring[k], fitnessOffspring[k] = fo(offspring[k])

    newFitness = np.concatenate((fitness, fitnessOffspring))
    newPopulation = np.concatenate((population, offspring))
        
    if objective_type == 'MIN':
        sortIndex = np.argsort(newFitness)
        
    elif objective_type == 'MAX':
        sortIndex = np.argsort(newFitness)[::-1]

    population = newPopulation[sortIndex[:N]]
    
    return population