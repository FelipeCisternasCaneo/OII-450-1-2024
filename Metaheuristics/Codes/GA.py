import random
import numpy as np
from Util.util import selectionSort

# Genetic Algorithm (GA)

def selectParent(population, fitness):
    """
    Selecciona dos padres basados en el ranking de fitness.
    """
    position = selectionSort(fitness)
    parent1, parent2 = population[position[0]], population[position[1]]
    
    return parent1, parent2

def crossover(parent1, parent2, probCrossover):
    """
    Realiza el cruce de dos padres para generar dos hijos.
    """
    pivot = int(np.round(min(len(parent1), len(parent2)) * probCrossover, 0))
    
    child1 = np.concatenate([parent1[:pivot], parent2[pivot:]])
    child2 = np.concatenate([parent2[:pivot], parent1[pivot:]])
    
    return child1, child2

# mutation operator
def mutate(chromosome, mutationRate):
    mutatedChromosome = []
    
    for gen in chromosome:
        if random.uniform(0, 1) < mutationRate: mutatedChromosome.append(1 - gen)
        
        else: mutatedChromosome.append(gen)
        
    return mutatedChromosome

def iterarGA(population, fitness, cross, muta):
    """
    Genetic Algorithm main loop for generating the next population.
    
    Args:
        population (list): Current population of individuals.
        fitness (list): Fitness values of the current population.
        cross (float): Crossover probability.
        muta (float): Mutation rate.
    
    Returns:
        np.ndarray: New population.
    """
    newPopulation = []
    
    for _ in range(len(population) // 2):
        # Selección de padres
        parent1, parent2 = selectParent(population, fitness)
        
        # Cruce
        child1, child2 = crossover(parent1, parent2, cross)
        
        # Mutación
        child1 = mutate(child1, muta)
        child2 = mutate(child2, muta)
        
        # Añadir a la nueva población
        newPopulation.extend([child1, child2])
    
    return np.array(newPopulation)