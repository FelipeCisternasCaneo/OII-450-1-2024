import random
import numpy as np
import math

# Realiza una perturbación de movimiento mediante el vuelo de Lèvy
def levy_flight(size, alpha=1.5, beta=0.5):
    dl = (math.gamma(1 + alpha) * np.sin(np.pi * alpha / 2) / (
                math.gamma((1 + alpha) / 2) * alpha * 2 ** ((alpha - 1) / 2))) ** (1 / alpha)
    u = np.random.normal(0, dl, size)
    v = np.random.normal(0, 1, size)
    r = u / (abs(v) ** (1 / alpha))
    x = np.random.normal(0, (1 / beta), size)
    return x * r

# Realiza una iteración de la metaheurística Draco Lizard Optimizer (DLO) 
def iterarDLO(iter, maxIter, dim, population, fitness, best, lb, ub, fo):
    """
    Realiza una iteración del algoritmo Draco Lizard Optimizer (DLO).

    Args:
        iter (int): Iteración actual.
        maxIter (int): Máximo número de iteraciones.
        dim (int): Dimensión del problema.
        population (np.ndarray): Población actual (shape: population_size x dim).
        fitness (np.ndarray): Fitness de la población actual.
        best (np.ndarray): La mejor solución global encontrada hasta ahora.
        fo (callable): Función objetivo a minimizar. Debe devolver (solución_recortada, fitness).

    Returns:
        np.ndarray: Población actualizada después de la iteración.
    """

    # Se calcula la cantidad de individuos de la población
    population_size = len(population)

    # Se crean la nueva población de individudos y de fitness
    new_population = population.copy()
    new_fitness = fitness.copy()

    # Inicializar la población si es la primera iteración
    if iter == 0:
        f_best = np.inf
        population = np.zeros((population_size, dim))

        for individuo in range(population_size):
            A = np.random.rand(population)
            population[:, individuo] = A * (ub[individuo] - lb[individuo]) + lb[individuo]

        for i in range(population_size):
            L = population[i, :].copy()
            _, fitness[i] = fo(L)
            if fitness[i] < f_best:
                f_best = fitness[i].copy()
                best = population[i, :].copy()
    else:
        _, f_best = fo(best)
        
    pos_new = population.copy()

    # Se actualiza cada solución de la población
    for i in range(population_size):
        if iter < maxIter / 2:
            ## FASE DE EXPLORACIÓN
            pos_i = population[i, :].copy()
            # Definir valores randómicos
            I = round(1 + np.random.random())
            k = np.random.choice(range(population_size))
            j = np.random.choice(range(population_size))
            P = population[k, :].copy()
            F_P = fitness[k]

            # Decidir la dirección de planeo
            if iter < maxIter * 0.1:
                dl = np.array([random.choice([1, 1, 1, 1]) for _ in range(dim)])
            elif iter < maxIter * 0.2:
                dl = np.array([random.choice([0, 1, 1, 1]) for _ in range(dim)])
            elif iter < maxIter * 0.3:
                dl = np.array([random.choice([0, 0, 1, 1]) for _ in range(dim)])
            elif iter < maxIter * 0.4:
                dl = np.array([random.choice([0, 0, 0, 1]) for _ in range(dim)])
            else:
                dl = np.array([random.choice([0, 0, 0, 0]) for _ in range(dim)])

            # Calcular movimiento de exploración
            if fitness[i] > F_P:
                pos_i = pos_i + np.random.rand(dim) * (P - I * pos_i) + dl * np.random.rand(dim) * (P - population[j, :])
            else:
                pos_i = pos_i + np.random.rand(dim) * (pos_i - P) + dl * np.random.rand(dim) * (P - population[j, :])
        else:
            ## FASE DE EXPLOTACIÓN
            pos_i = population[i, :].copy()
            # Copiar la población original
            agentes = list(range(population_size))
            # agentes.remove(i)
            m = random.choice(agentes)
            selected_searchAgent = population[m, :].copy()
            p = np.random.random()

            # Calcular movimiento de explotación
            if p < 0.2:
                pos_i = best + levy_flight(dim) * (best - selected_searchAgent)
            else:
                pos_i = best + np.random.normal(loc=0.0, scale=10) * (1 - iter / maxIter) * (population[i, :] - selected_searchAgent)

        # Evaluar la nueva posición usando la función objetivo
        pos_i_clipped, f_new = fo(pos_i)

        # Actualizar la nueva solución en caso de ser mejor (Se asume minimización)
        if f_new <= fitness[i]:
            new_population[i, :] = pos_i_clipped
            new_fitness[i] = f_new

            if f_new <= f_best:
                best = pos_i_clipped.copy()
                f_best = f_new
    return new_population