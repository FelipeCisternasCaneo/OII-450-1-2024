import numpy as np
import random as rd

# Quokka Optimization Algorithm (QSO)
# https://doi.org/10.1515/jisys-2024-0051

def iterarQSO(population, best, lb, ub):
   
    # Parámetros iniciales de QSO
    T = rd.uniform(0.2, 0.44)  # Temperatura
    H = rd.uniform(0.3, 0.65)  # Humedad
    N = rd.uniform(0, 1)       # Nitrógeno

    # Convertimos la población en array numpy
    population = np.array(population)
    best = np.array(best)
    new_population = np.copy(population)

    # Actualización de la posición de cada quokka
    for i in range(len(population)):
        r1 = rd.random()
        r2 = rd.random()

        # Actualización de la posición usando la ecuación proporcionada
        drought = np.abs(population[i] - best)  # Parámetro de sequía
        new_position = population[i] + r1 * (best - drought) + r2 * (T * H - N)

        # Reparación de soluciones fuera de los límites
        new_population[i] = np.clip(new_position, lb, ub)

    return new_population
