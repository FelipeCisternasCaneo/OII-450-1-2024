import numpy as np

# Lyrebird Optimization Algorithm (LOA)
# http://doi.org/10.3390/biomimetics8060507

def iterarLOA(maxIter, population, mejores_fitness, lb, ub, t, dim):
    """
    maxIter: Máximo de iteraciones.
    population: Población actual (numpy array).
    mejores_fitness: Fitness de las mejores soluciones.
    lb: Límite inferior de las variables.
    ub: Límite superior de las variables.
    t: Iteración actual.
    dim: Dimensiones del problema.
    """
    
    population = np.array(population)
    N = population.shape[0]  # Tamaño de la población
    posibles_mejoras = np.zeros_like(population)
    
    # Generar valores aleatorios para decidir entre exploración y explotación
    r = np.random.uniform(0, 1, size=N)

    # Exploración: Escapar hacia mejores zonas
    exploracion_mask = r < 0.5
    if np.any(exploracion_mask):  # Si hay individuos que exploran
        mejores_zonas = np.array(mejores_fitness)
        safe_area_indices = np.random.randint(0, len(mejores_zonas), size=np.sum(exploracion_mask))
        safe_areas = mejores_zonas[safe_area_indices]
        r_exploracion = np.random.uniform(0, 1, size=(np.sum(exploracion_mask), dim))
        posibles_mejoras[exploracion_mask] = population[exploracion_mask] + r_exploracion * (safe_areas[:, np.newaxis] - population[exploracion_mask])

    # Explotación: Esconderse
    explotacion_mask = ~exploracion_mask
    if np.any(explotacion_mask):  # Si hay individuos que explotan
        r_explotacion = np.random.uniform(0, 1, size=(np.sum(explotacion_mask), dim))
        diff = ub - lb
        adjustment = (diff / (t if t > 0 else 1)) * (1 - 2 * r_explotacion)
        posibles_mejoras[explotacion_mask] = population[explotacion_mask] + adjustment

    return population, posibles_mejoras