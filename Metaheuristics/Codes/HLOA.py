import random
import numpy as np

# Horned Lizard Optimization Algorithm (HLOA)
# https://doi.org/10.1007/s10462-023-10653-7

def iterarHLOAScp(dim, population, best, lb0, ub0):
    """
    HLOA Binario para resolver problemas combinatoriales.

    maxIter: Máximo número de iteraciones
    it: Iteración actual
    dim: Dimensión de las soluciones
    population: Población actual (matriz binaria de tamaño N x dim)
    bestSolution: Mejor solución encontrada (binaria)
    lb: Límite inferior (usualmente 0)
    ub: Límite superior (usualmente 1)
    """
    # Asegurarse de que lb y ub sean listas del tamaño de las dimensiones
    if not isinstance(lb0, list):
        lb0 = [lb0] * dim
        
    if not isinstance(ub0, list):
        ub0 = [ub0] * dim

    # Convertir la población a numpy array si no lo es
    population = np.array(population)

    # Parámetros de HLOA
    alpha = 0.1  # Intensificación (oscurecer la piel)
    beta = 0.3   # Diversificación (aclarar la piel)
    epsilon = 1e-10  # Evitar divisiones por cero

    # Tamaño de la población
    N = len(population)

    # Función sigmoide para binarización
    def sigmoide(x):
        return 1 / (1 + np.exp(-x))

    for i in range(N):
        for j in range(dim):
            # Estrategia de Camuflaje (Crypsis)
            r1 = random.random()
            r2 = random.random()
            local_search = (1 - r1) * best[j] + r2 * (population[i][j] - best[j])
            
            # Aplicar sigmoide y binarizar
            if sigmoide(local_search) >= 0.5:
                population[i][j] = 1
                
            else:
                population[i][j] = 0

            # Estrategia de Oscurecimiento o Aclaramiento de la piel
            if random.random() < 0.5:
                # Aclarar la piel (exploración global)
                exploration = best[j] + beta * (population[i][j] - best[j])
                
            else:
                # Oscurecer la piel (exploración local)
                exploration = best[j] - alpha * (population[i][j] - lb0[j])
                
            # Aplicar sigmoide y binarizar
            if sigmoide(exploration) >= 0.5:
                population[i][j] = 1
            
            else:
                population[i][j] = 0

            # Estrategia de Expulsión de sangre (Blood-Squirting)
            if random.random() < 0.1:
                projectile_motion = best[j] + alpha * np.sin(random.random() * np.pi)
                # Aplicar sigmoide y binarizar
                if sigmoide(projectile_motion) >= 0.5:
                    population[i][j] = 1
                    
                else:
                    population[i][j] = 0

            # Estrategia de Movimiento para escapar (Move-to-escape)
            walk = random.uniform(-1, 1)
            escape_motion = best[j] + walk * (population[i][j] - best[j])
            # Aplicar sigmoide y binarizar
            
            if sigmoide(escape_motion) >= 0.5:
                population[i][j] = 1
                
            else:
                population[i][j] = 0

    return population

def iterarHLOABen(dim, population, best, lb, ub):
    '''
    maxIter: Máximo de iteraciones
    iter: Iteración actual
    dim: Dimensión de las soluciones
    population: Población actual (lista de listas)
    best: Mejor solución encontrada
    lb: Límite inferior
    ub: Límite superior
    '''
    
    # Asegurarse de que population sea un numpy array
    population = np.array(population)

    # Parámetros de HLOA
    alpha = 0.1  # Controla la intensificación
    beta = 0.3   # Controla la diversificación

    # Tamaño de la población
    N = len(population)
    
    for i in range(N):
        for j in range(dim):
            # Estrategia de Camuflaje (Crypsis)
            r1 = random.random()
            r2 = random.random()
            local_search = (1 - r1) * best[j] + r2 * (population[i][j] - best[j])
            population[i][j] = np.clip(local_search, lb[j], ub[j])

            # Estrategia de Oscurecimiento o Aclaramiento de la piel
            if random.random() < 0.5:
                # Aclarar la piel (exploración global)
                population[i][j] = best[j] + beta * (population[i][j] - best[j])
                
            else:
                # Oscurecer la piel (exploración local)
                population[i][j] = best[j] - alpha * (population[i][j] - lb[j])
            
            # Expulsión de sangre (Blood-Squirting)
            if random.random() < 0.1:
                projectile_motion = best[j] + alpha * np.sin(random.random() * np.pi)
                population[i][j] = np.clip(projectile_motion, lb[j], ub[j])
            
            # Movimiento para escapar (Move-to-escape)
            walk = random.uniform(-1, 1)
            escape_motion = best[j] + walk * (population[i][j] - best[j])
            population[i][j] = np.clip(escape_motion, lb[j], ub[j])

    # Devolver la población como numpy array (el solver espera un array, no una lista)
    return population
