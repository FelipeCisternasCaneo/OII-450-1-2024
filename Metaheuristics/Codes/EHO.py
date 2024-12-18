import numpy as np

# Elk Herd Optimization (EHO)
def iterarEHO(maxIter, it, dim, population, bestSolution, lb, ub, fitness):
    N = len(population)
    MalesRate = 0.2  # Percentage of males in the population
    No_of_Males = round(N * MalesRate)
    
    population = np.array(population)
    bestSolution = np.array(bestSolution)

    # Sort the population based on fitness
    sorted_indexes = np.argsort(fitness)
    sorted_fitness = fitness[sorted_indexes]

    # Update the best solution (BestBull)
    bestSolution = population[sorted_indexes[0], :]
    bestFitness = sorted_fitness[0]

    # Roulette Wheel Selection for families
    TransposeFitness = 1 / sorted_fitness[:No_of_Males]
    total_fitness = np.sum(TransposeFitness)
    Families = np.zeros(N, dtype=int)

    for i in range(No_of_Males, N):
        randNumber = np.random.rand()
        sum_fitness = 0
        for j in range(No_of_Males):
            sum_fitness += TransposeFitness[j] / total_fitness
            if sum_fitness > randNumber:
                Families[sorted_indexes[i]] = sorted_indexes[j]
                break

    # Reproduction (Updating elk positions)
    for i in range(N):
        if Families[i] == 0:  # Male elk
            h = np.random.randint(N)
            for j in range(dim):
                population[i, j] = population[i, j] + np.random.rand() * (population[h, j] - population[i, j])
                population[i, j] = np.clip(population[i, j], lb[j], ub[j])
        else:  # Female elk
            h = np.random.randint(N)
            male_index = Families[i]
            for j in range(dim):
                population[i, j] = population[i, j] + np.random.rand() * (population[male_index, j] - population[h, j])
                population[i, j] = np.clip(population[i, j], lb[j], ub[j])

    # Return updated population and the best fitness found
    return population