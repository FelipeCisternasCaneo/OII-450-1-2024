import numpy as np

# Traffic Jam Optimization (TJO)
# Basado en la implementación original en MATLAB

def iterarTJO(maxIter, iter, dim, population, best, pBest, lb, ub):
    """
    Traffic Jam Optimization (TJO) - Iteración Única
    
    Args:
        maxIter: Máximo de iteraciones
        iter: Iteración actual
        dim: Dimensión del problema
        population: Población actual (x)
        best: Mejor solución global actual (bestx)
        pBest: Memoria de la mejor posición de cada individuo (FlockMemoryX)
        lb: Límite inferior
        ub: Límite superior
    
    Returns:
        np.ndarray: Población actualizada (nuevas posiciones candidatas)
    """
    lb = np.array(lb)
    ub = np.array(ub)

    a_start, a_end = 0.2, 0.9 
    c_start, c_end = 0.1, 0.8
    
    a_t = a_start + (iter / maxIter) * (a_end - a_start)
    c_t = c_start + (iter / maxIter) * (c_end - c_start)
    
    N = population.shape[0]     
    r = iter / maxIter          
    
    best = np.array(best)
    
    # Calcular BestX (Posición objetivo híbrida)
    BestX = (1 - r) * pBest + r * best

    # Conductores conduciendo aleatoriamente (Traffic Jam)
    rand_sin = np.sin(2 * np.pi * np.random.rand(N, 1))
    rand_cos = np.cos(2 * np.pi * np.random.rand(N, 1))
    y = (1 - r) * np.exp(-r) * rand_sin * rand_cos * c_t
    
    rand_pos = (ub - lb) * np.random.rand(N, dim) + lb
    population = BestX + y * rand_pos
    
    # Auto-ajuste de los conductores (Drivers self-adjustment)
    indices_rand = np.random.randint(0, N, size=N)
    
    mask = np.random.rand(N, 1) > 0.5
    
    adjust_factor = c_t * np.sin(np.pi * np.random.rand(N, 1))
    
    diff_case_1 = population[indices_rand] - population
    
    diff_case_2 = BestX[indices_rand] - population
    
    population = population + adjust_factor * np.where(mask, diff_case_1, diff_case_2)
    
    # Policía de tráfico dirigiendo (Traffic police)
    police_factor = a_t * np.sin(2 * np.pi * np.random.rand(N, 1))
    population = BestX + police_factor * (BestX - population)
    
    # Manejo de límites (Cross-border processing)
    population = np.clip(population, lb, ub)
    
    return population