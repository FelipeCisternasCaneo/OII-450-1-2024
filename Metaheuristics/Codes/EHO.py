import numpy as np
import random as rd

# Elk Herd Optimization (EHO)
# 
def iterarEHO(maxIter, it, dim, population, bestSolution, lb, ub, fitness):
   
    population = np.array(population)
    bestSolution = np.array(bestSolution)
    
    N = len(population)
    MalesRate = 0.6  # Porcentaje de toros en la manada
    nro_toros = int(N * MalesRate)

    # Ordena la pop en base al menor fitness
    sorted_indexes = np.argsort(fitness)
    sorted_fitness = fitness[sorted_indexes]

    copia_population = population
    
    # Update the best solution (BestBull)
    bestSolution  = population[sorted_indexes[0], :]
    bestFitness = sorted_fitness[0]

    transposeFitness = np.zeros(nro_toros)
    # Roulette Wheel Selection for families
    for i in range(nro_toros):
        transposeFitness[i] = 1 / sorted_fitness[i]
        
    total_fitness = np.sum(transposeFitness)
    Familes = np.zeros(N, dtype=int)

    for i in range(nro_toros + 1, N):
        female_index = sorted_indexes[i]
    
        randNumber = rd.uniform(0,1)
        sum_fitness = 0
        male_index = 0
        
        for j in range(nro_toros):
            sum_fitness = sum_fitness + (transposeFitness[j] / total_fitness)
            if sum_fitness > randNumber:
                male_index = j
                break
            
        Familes[female_index] = sorted_indexes[male_index]

    # Reproduction (Updating elk positions)
    for i in range(N):
        if Familes[i] == 0:  # Male(padre) elk
            h = rd.randint(0, N-1)
            for j in range(dim):
                copia_population[i, j] = population[i, j] + rd.random() * (population[h, j] - population[i, j])
                copia_population[i, j] = np.clip(copia_population[i, j], lb, ub)
                
        else:  # harem(madre) elk
            h = int(rd.uniform(0, N)) + 1
            MaleIndex = Familes[i] 
            indices = np.where(Familes == MaleIndex)[0]
            hh = np.random.permutation(len(indices))
            h = round(1 + (len(hh) - 1) * rd.random())
            
            for j in range(dim):
                #rand = -2 + 4 * rd.uniform(0,2)
                gama = 2 * rd.random() 
                beta = 2 - gama
                copia_population[i, j] = (population[i, j] + gama * (population[MaleIndex, j] - population[i, j]) + beta * (population[h - 1, j] - population[i, j]))
                #copia_population[i, j] = (population[i, j] + (population[MaleIndex, j] - population[i, j]) + rand * (population[h-1, j] - population[i, j]))
    
    return np.array(copia_population)
    