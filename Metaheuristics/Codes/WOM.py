import random
import copy
import numpy as np

# Wombat Optimization Algorithm (WOM)
# https://doi.org/10.3390/math12071059

def eq4(population, fitness, i):
    CFP_i = []
    for k in range(len(fitness)):
        if fitness[k] < fitness[i] and i != k:
            CFP_i.append(population[k])
            
    if len(CFP_i) == 0: # Para que la lista no quede vacía, se agrega a si mismo
        CFP_i.append(population[i])
    
    return np.array(CFP_i)

def eq5(SFP, population, i, dim):
    # Actualizar cada dimensión de la nueva posición
    population_part1 = copy.deepcopy(population) # Copia profunda para lista de listas
    
    for j in range(dim):
        r_ij = np.random.rand()  # Generar un número aleatorio entre 0 y 1
        I_ij = np.random.choice([1, 2])  # Escoger entre 1 y 2
        
        # Actualizar la j-ésima dimensión de la nueva posición usando la fórmula
        population_part1[i][j] = population[i][j] + r_ij * (SFP[j] - I_ij * population[i][j])
        
    # Devolver la nueva posición sin modificar la población original
    
    return np.array(population_part1)

def eq6(new_population, population, fitness, function, i):
    new_fitness = fitness.copy()
    _, new_fitness[i] = function(new_population[i])
    
    if new_fitness[i] <= fitness[i]: # Si el fitness es mejor, se actualiza la población
        population[i] = new_population[i]
    
    return population, new_fitness

def eq7(population, i, t, dim, lb, ub):
    population_part2 = copy.deepcopy(population) # Copia profunda para lista de listas
    
    for j in range(dim):
        r_ij = np.random.rand()
        population_part2[i][j] = population[i][j] + (1 - 2*r_ij) * ((ub[j] - lb[j])/(t+1)) # Actualizar la j-ésima dimensión
    
    return np.array(population_part2)

def eq8(new_population, population, fitness, function, i): # Lo mismo que eq6
    new_fitness = fitness.copy()
    _, new_fitness[i] = function(new_population[i])
    
    if new_fitness[i] <= fitness[i]:
        population[i] = new_population[i]
        
    return population, new_fitness

def xplr(population, fitness, function, dim, i): # Fase de forrajeo / Exploracion
    CFP_i = eq4(population, fitness, i)
    pop_p1 = eq5(random.choice(CFP_i), population, i, dim)
    population, new_fitness = eq6(pop_p1, population, fitness, function, i)
    
    return population, new_fitness

def xplt(population, fitness, function, dim, lb, ub, i, t): # Fase de escape / Explotacion
    pop_p2 = eq7(population, i, t, dim, lb, ub)
    population, new_fitness = eq8(pop_p2, population, fitness, function, i)
    
    return population, new_fitness


def iterarWOM(maxIter, t, dim, population, fitness, lb, ub, f):
    for i in range(len(population)):
        population, fitness = xplr(population,fitness, f, dim, i)
        population, fitness = xplt(population, fitness, f, dim, lb, ub, i, t)
        
    return np.array(population)