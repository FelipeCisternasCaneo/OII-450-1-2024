"""
Cape Lynx Optimizer (CLO)

Reference:
Wang, X., & Yao, L. (2025). Cape lynx optimizer: A novel metaheuristic algorithm 
for enhancing wireless sensor network coverage. Measurement, 256, 118361.
https://doi.org/10.1016/j.measurement.2025.118361
"""

import numpy as np

# Constantes del algoritmo
FL = 0.2  # Factor de vuelo (flight)
FD = 0.1  # Factor de buceo (diving)

def split_array_permutation(dim):
    """
    Divide aleatoriamente un arreglo de dimensión 'dim' en tres sub-arreglos binarios
    que suman el arreglo original.
    """
    num_ones_1 = np.random.randint(1, dim - 1)
    num_ones_2 = np.random.randint(1, dim - num_ones_1)
    
    perm = np.random.permutation(dim)
    
    positions_1 = perm[:num_ones_1]
    positions_2 = perm[num_ones_1:num_ones_1 + num_ones_2]
    positions_3 = perm[num_ones_1 + num_ones_2:]
    
    array1 = np.zeros(dim)
    array2 = np.zeros(dim)
    array3 = np.zeros(dim)
    
    array1[positions_1] = 1
    array2[positions_2] = 1
    array3[positions_3] = 1
    
    return array1, array2, array3


def iterarCLO(maxIter, iter, population, dim, fitness, best, lb, ub, fo):
    """
    Crested Porcupine Optimizer (CLO) adaptado al sistema de iteraciones.
    
    IMPORTANTE: Este algoritmo evalúa DENTRO de la iteración (como el original).
    La evaluación individual por solución permite la selección greedy inmediata.
    """
    pop_size = population.shape[0]
    lb = np.array(lb)
    ub = np.array(ub)
    
    # Encontrar mejor solución actual (best_Sol en el original)
    best_idx = np.argmin(fitness)
    best_Sol = population[best_idx].copy()
    
    # Nueva población candidata
    new_population = population.copy()
    
    for i in range(pop_size):
        # Seleccionar 3 índices aleatorios diferentes de i
        candidates = [idx for idx in range(pop_size) if idx != i]
        if len(candidates) >= 3:
            ll = np.random.choice(candidates, size=3, replace=False)
        else:
            ll = np.random.choice(range(pop_size), size=3, replace=False)
        
        # ========== ERROR 1 CORREGIDO: Condición de iteración ==========
        # Original usa 't % 2 == 0 and (t < 300 or t > 700)'
        if iter % 2 == 0 and (iter < 300 or iter > 700):
            # Estrategia de exploración
            if dim == 2:
                # Caso especial para 2D
                angle = np.random.uniform(0, 2 * np.pi)
                distance = np.random.uniform(0, 2) * (1 - iter / maxIter)
                new_population[i, 0] = population[i, 0] + distance * np.cos(angle) * np.random.random() * np.random.uniform(0, 2)
                new_population[i, 1] = population[i, 1] + distance * np.sin(angle) * np.random.random() * np.random.uniform(0, 2)
            else:
                # Para dimensiones mayores
                HP = 1 if np.random.random() >= 0.2 else 0
                array1, array2, array3 = split_array_permutation(dim)
                
                # ========== ERROR 2 CORREGIDO: Usar x[i, :].copy() no population[i, :] ==========
                # El original usa x[i, :].copy() que es la población MODIFICADA en esta iteración
                new_population[i, :] = (
                    array1 * new_population[i, :].copy() + 
                    array2 * (new_population[ll[1], :].copy() + HP * np.random.uniform(0, 2, dim) * (new_population[ll[2], :] - new_population[ll[0], :])) + 
                    array3 * (new_population[ll[2], :].copy() + (1 - HP) * np.random.normal(0, 10, dim) * (new_population[ll[0], :] - new_population[ll[1], :]))
                )
        
        else:
            # Estrategia de explotación
            AP = 0.5
            
            if np.random.random() < 0.5:
                # Modo buceo (diving)
                A = 2 * AP * (1 - iter / maxIter)
                
                if np.random.random() < 0.5:
                    # ========== ERROR 3 CORREGIDO: shape de randn ==========
                    # Original: np.random.randn(1, dim) retorna shape (1, dim)
                    # Necesitamos shape (dim,) para sumar correctamente
                    new_population[i, :] = best_Sol + np.random.randn(dim) * A + np.random.random() * (new_population[ll[0], :] - new_population[ll[1], :])
                else:
                    random_idx = np.random.randint(pop_size)
                    new_population[i, :] = new_population[random_idx, :].copy() + np.random.randn(dim) * A + np.random.random() * (new_population[ll[0], :] - new_population[ll[1], :])
            else:
                # Modo vuelo (flight)
                random_idx = np.random.randint(pop_size)
                r1 = np.random.random()
                r2 = np.random.random()
                
                velocities = (FL * r1 * (best_Sol - new_population[i, :]) +
                             FD * r2 * (new_population[random_idx, :] - new_population[i, :]))
                
                new_population[i, :] = new_population[i, :] + velocities
        
        # Corrección de límites (igual que el original)
        for j in range(dim):
            if new_population[i, j] > ub[j]:
                new_population[i, j] = lb[j] + np.random.rand() * (ub[j] - lb[j])
            elif new_population[i, j] < lb[j]:
                new_population[i, j] = lb[j] + np.random.rand() * (ub[j] - lb[j])
        
        # ========== ERROR 4 CORREGIDO: EVALUACIÓN Y SELECCIÓN GREEDY ==========
        # El original evalúa AQUÍ, no en update_population()
        # Esto es crítico porque permite selección greedy inmediata
        new_population[i, :] = np.clip(new_population[i, :], lb, ub)
        _, new_fitness = fo(new_population[i, :])
        
        # Selección greedy: solo acepta si mejora
        if new_fitness < fitness[i]:
            population[i, :] = new_population[i, :].copy()
            fitness[i] = new_fitness
        else:
            # Si no mejora, revertir (mantener xp[i])
            new_population[i, :] = population[i, :].copy()
    
    # ========== RETORNAR LA POBLACIÓN YA EVALUADA ==========
    # Como ya evaluamos y seleccionamos, retornamos la población actualizada
    # update_population() NO debe re-evaluar
    return population