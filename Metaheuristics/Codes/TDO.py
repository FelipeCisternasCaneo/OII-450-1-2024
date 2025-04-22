import numpy as np
import random

# Tasmanian Devil Optimization (TDO)
# https://doi.org/10.1109/ACCESS.2022.3151641

def iterarTDO(maxIter, iter, dim, population, fitness, fo, objective_type):
    """
    Implementación del algoritmo Tasmanian Devil Optimization (TDO).
    
    Args:
        maxIter (int): Número máximo de iteraciones.
        it (int): Iteración actual.
        dim (int): Dimensión del espacio de búsqueda.
        population (np.ndarray): Población actual.
        fitness (np.ndarray): Valores de fitness para la población actual.
        function (callable): Función objetivo que evalúa cada individuo.
        typeProblem (str): Tipo de problema ('MIN' o 'MAX').

    Returns:
        np.ndarray: Población actualizada.
    """
    N = len(population)
    population = np.array(population)

    for i in range(N):
        r = random.uniform(0.0, 1.0)

        # Selección de un demonio de Tasmania aleatorio de la población
        k = np.random.choice(np.delete(np.arange(N), i))
        CPi = population[k]
        xNew = np.copy(population[i])

        # Determinar si acercarse o alejarse en función del tipo de problema
        if objective_type == 'MIN':
            condition = fo(CPi)[1] < fitness[i]
        elif objective_type == 'MAX':
            condition = fo(CPi)[1] > fitness[i]
        else:
            raise ValueError("typeProblem debe ser 'MIN' o 'MAX'")

        for j in range(dim):
            if condition:
                # Si el demonio elegido es mejor, se acerca
                I = random.randint(1, 2)
                xNew[j] = population[i][j] + random.uniform(0.0, 1.0) * (CPi[j] - I * population[i][j])
            else:
                # Si el demonio elegido es peor, se aleja
                xNew[j] = population[i][j] + random.uniform(0.0, 1.0) * (population[i][j] - CPi[j])

        # Evaluar nueva posición
        xNew, fitnessNew = fo(xNew)
        if (objective_type == 'MIN' and fitnessNew < fitness[i]) or (objective_type == 'MAX' and fitnessNew > fitness[i]):
            population[i] = np.copy(xNew)

        # Explotación: búsqueda local si r >= 0.5
        if r >= 0.5:
            R = 0.01 * (1 - (iter / maxIter))  # Factor de explotación
            for j in range(dim):
                xNew[j] = population[i][j] + (2 * random.uniform(0.0, 1.0) - 1) * R * xNew[j]

            # Evaluar nueva posición tras explotación
            xNew, fitnessNew = fo(xNew)
            if (objective_type == 'MIN' and fitnessNew < fitness[i]) or (objective_type == 'MAX' and fitnessNew > fitness[i]):
                population[i] = np.copy(xNew)

    return population