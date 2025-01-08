import random
import numpy as np
from Util.util import selectionSort

# Genetic Algorithm

# selection of parent
def selectParent(population, fitness):
    position = selectionSort(fitness)

    parent1 = population[position[0]]
    parent2 = population[position[1]]
    
    return parent1, parent2

# crossover operator
def crossover(parent1, parent2, probCrossover):
    pivot = int(np.round((len(parent1) * probCrossover), 0))
    
    child1 = parent1[:pivot] + parent2[pivot:]
    child2 = parent2[:pivot] + parent1[pivot:]
    
    return child1, child2

# mutation operator
def mutate(chromosome, mutationRate):
    mutatedChromosome = []
    
    for gen in chromosome:
        if random.uniform(0, 1) < mutationRate: mutatedChromosome.append(1 - gen)
        
        else: mutatedChromosome.append(gen)
        
    return mutatedChromosome

def iterarGA(population, fitness, cross, muta):
    newPopulation = []
    
    for i in range(len(population) // 2):
        parent1, parent2 = selectParent(population, fitness)
        child1, child2 = crossover(parent1, parent2, cross)
        child1 = mutate(child1, muta)
        child2 = mutate(child1, muta)
        newPopulation.extend([child1, child2])    
    
    return np.array(newPopulation)