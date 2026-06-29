import numpy as np

# Lyrebird Optimization Algorithm (LOA)
# http://doi.org/10.3390/biomimetics8060507

def iterarLOA(iter, dim, population, best, lb0, ub0):
    """
    iter: Iteración actual
    dim: Dimensionalidad del problema
    population: Población actual (numpy array)
    best: Fitness de las mejores soluciones.
    lb0: Límite inferior del espacio de búsqueda (numpy array)
    ub0: Límite superior del espacio de búsqueda (numpy array)
    """

    population = np.array(population)
    N = population.shape[0]  # Tamaño de la población
    posibles_mejoras = np.zeros_like(population)
    
    # Generar valores aleatorios para decidir entre exploración y explotación
    r = np.random.uniform(0, 1, size=N)

    # Exploración: Escapar hacia mejores zonas
    exploracion_mask = r < 0.5
    if np.any(exploracion_mask):  # Si hay individuos que exploran
        mejores_zonas = np.array(best)
        safe_area_indices = np.random.randint(0, len(mejores_zonas), size=np.sum(exploracion_mask))
        safe_areas = mejores_zonas[safe_area_indices]
        r_exploracion = np.random.uniform(0, 1, size=(np.sum(exploracion_mask), dim))
        posibles_mejoras[exploracion_mask] = population[exploracion_mask] + r_exploracion * (safe_areas[:, np.newaxis] - population[exploracion_mask])

    # Explotación: Esconderse
    explotacion_mask = ~exploracion_mask
    if np.any(explotacion_mask):  # Si hay individuos que explotan
        r_explotacion = np.random.uniform(0, 1, size=(np.sum(explotacion_mask), dim))
        diff = ub0 - lb0
        adjustment = (diff / (iter if iter > 0 else 1)) * (1 - 2 * r_explotacion)
        posibles_mejoras[explotacion_mask] = population[explotacion_mask] + adjustment

    return population, posibles_mejoras