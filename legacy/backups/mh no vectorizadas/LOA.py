import numpy as np
import random as rd

# Lyrebird Optimization Algorithm (LOA)
# http://doi.org/10.3390/biomimetics8060507

def escapar(lyrebird, mejores_zonas, dim):
    nueva_posicion = []
    
    for j in range(dim):
        r = rd.uniform(0, 1)
        safe_area = mejores_zonas[np.random.randint(0, len(mejores_zonas))]
        nueva_posicion_j = lyrebird[j] + r * (safe_area - lyrebird[j])
        nueva_posicion.append(nueva_posicion_j)
        
    return nueva_posicion

def esconderse(lyrebird, ub, lb , t, dim):
    nueva_posicion = []
    
    for j in range(dim):
        r = rd.uniform(0, 1)
        
        diff = ub - lb
        
        if t == 0:
            nueva_posicion_j =  lyrebird[j] + (1 - 2 * r) * (diff) / 1
            
        else:
            nueva_posicion_j =  lyrebird[j] + (1 - 2 * r) * (diff) / t
        
        nueva_posicion.append(nueva_posicion_j)
        
    return nueva_posicion

def iterarLOA(maxIter: int, population: list, mejores_fitness: list, lb: int, ub: int, t: int, dim):
    posibles_mejoras = []
    population = np.array(population)
    
    for i in range(len(population)):
        r = rd.uniform(0, 1)
        # Exploracion
        if r < 0.5:
            # Se seleccionan las zonas mejor evaluadas a la actual para la exploracion.
            mejores_zonas = []
            for j in range(len(mejores_fitness)):
                mejores_zonas.append(mejores_fitness[j])
                
            if len(mejores_zonas) > 0:
                nueva_posicion = escapar(population[i], mejores_zonas,dim)

        # Explotación
        else:
            nueva_posicion = esconderse(population[i], ub, lb, t, dim)
        
        # Se evalua el fitness del resultado obtenido, reemplazandolo si es mejor
        posibles_mejoras.append(nueva_posicion)

    posibles_mejoras = np.array(posibles_mejoras)
    
    return population, posibles_mejoras