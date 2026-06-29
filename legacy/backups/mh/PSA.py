import numpy as np

# Pendulum Search Algorithm (PSA)
# https://doi.org/10.3390/a15060214

def iterarPSA(maxIter, t, dimension, population, bestSolution):
    """
    Implementación optimizada del algoritmo Pendulum Search Algorithm (PSA).
    """
    population = np.array(population)  # Asegurar que la población sea un array de numpy
    bestSolution = np.array(bestSolution)  # Asegurar que la mejor solución sea un array de numpy

    # Calcular parámetros constantes para esta iteración
    pend_factor = 2 * np.exp(-t / maxIter)  # Factor dependiente de la iteración
    rand = np.random.uniform(0, 1, size=population.shape)  # Generar matriz de números aleatorios
    pend = pend_factor * np.cos(2 * np.pi * rand)  # Movimiento pendular para cada dimensión

    # Actualizar las posiciones de toda la población
    population += pend * (bestSolution - population)

    return population
