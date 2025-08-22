import numpy as np

# Elk Herd Optimizer (EHO)
# https://doi.org/10.1007/s10462-023-10680-4

def iterarEHO(maxIter, iter, dim, population, best, lb, ub, fitness, fo):
    pop_size = len(population)
    bull_ratio = 0.2
    num_bulls = round(pop_size * bull_ratio)

    population = np.array(population)
    best = np.array(best)

    sorted_indexes = np.argsort(fitness)
    sorted_fitness = fitness[sorted_indexes]
    
    best = population[sorted_indexes[0], :]
        
    bull_indices = sorted_indexes[:num_bulls]
    
    # Selección Ruleta para las Familias (Rutting Season)
    epsilon = 1e-10
    bull_fitness = np.clip(sorted_fitness[:num_bulls], epsilon, None)
    inv_fitness = 1.0 / bull_fitness
    selection_probs = inv_fitness / np.sum(inv_fitness)

    Families = np.zeros(pop_size, dtype=int)

    for i in range(num_bulls, pop_size):
        FemaleIndex = sorted_indexes[i]
        selected_bull = np.random.choice(bull_indices, p=selection_probs)
        Families[FemaleIndex] = selected_bull

    # Reproducción (Calving Season)
    offspring_population = []

    for i in range(pop_size):
        individual = population[i].copy()
        # Bull (male) index
        if i in bull_indices:  
            h = np.random.randint(0, pop_size)
            
            new_individual = []

            alpha = np.random.rand()

            # dim = 10 
            # sol = (0, 0, 0, 0, 0, 0, 0, 0 )
            new_individual = individual + alpha * (population[h] - individual)
            new_individual = np.clip(new_individual, lb, ub)
            
            """
            for j in range(dim):
                new_val = individual[j] + alpha * (population[h, j] - individual[j])
                new_val = np.clip(new_val, lb[j], ub[j])
                new_individual.append(new_val)
            """
            
        # Harem (female) index
        else:  
            MaleIndex = Families[i]
            h = np.random.randint(0, num_bulls)
            random_bull = bull_indices[h]
            
            new_individual = []
            for j in range(dim):
                gamma = np.random.uniform(-2, 2)
                #gamma = 1
                #beta = np.random.uniform(0, 1)
                beta = 1
                new_val = individual[j] + beta * ((population[MaleIndex, j] - individual[j])) + gamma * ((population[random_bull, j] - individual[j]))
                new_val = np.clip(new_val, lb[j], ub[j])
                new_individual.append(new_val)
        
        offspring_population.append(new_individual)

    offspring_population = np.array(offspring_population)
    merged_population = np.concatenate([population, offspring_population])
    merged_fitness = np.zeros(len(merged_population))
    
    for i in range(len(merged_population)):
        _, merged_fitness[i] = fo(merged_population[i])
    
    # Seleccion poblacion
    sorted_indices = np.argsort(merged_fitness)
    selected_population = merged_population[sorted_indices[:pop_size]]
    
    return selected_population