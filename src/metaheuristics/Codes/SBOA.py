import numpy as np
import math

# Secretary Bird Optimization Algorithm (SBOA) - VERSIÓN ESTANDARIZADA
# https://doi.org/10.1007/s10462-024-10729-y

# La función levy no necesita cambios
def levy(dim, beta=1.5):
    sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    return u / np.abs(v) ** (1 / beta)

# --- Función iterarSBOA con nombres de parámetros estándar ---
def iterarSBOA(maxIter, iter, dim, population, fitness, best, fo):
    """
    Realiza una iteración del algoritmo SBOA.

    Args:
        maxIter (int): Máximo número de iteraciones.
        iter (int): Iteración actual.
        dim (int): Dimensión del problema.
        population (np.ndarray): Población actual.
        fitness (np.ndarray): Fitness de la población actual.
        best (np.ndarray): La mejor solución encontrada hasta ahora.
        fo (callable): Función objetivo a minimizar. Se espera que devuelva (solución_recortada, fitness).

    Returns:
        np.ndarray: La población actualizada después de la iteración.
    """
    # Asegurarse de que 'best' sea un array numpy para operaciones posteriores
    best_position = np.asarray(best) # Usamos una variable interna para claridad
    N = len(population) # Tamaño de la población

    # Calcular CF usando el nombre estándar 'iter'
    CF = (1 - iter / maxIter) ** (2 * iter / maxIter)

    # Etapas: búsqueda de presas, aproximación, ataque (usando 'iter')
    if iter < maxIter / 3:  # Etapa de búsqueda de presas
        R1 = np.random.rand(N, dim)
        indices = np.random.randint(N, size=(N, 2))
        X1 = population + (population[indices[:, 0]] - population[indices[:, 1]]) * R1

    elif maxIter / 3 <= iter < 2 * maxIter / 3:  # Etapa de aproximación
        RB = np.random.randn(N, dim)
        # Usar 'iter' en el cálculo
        exp_term = np.exp((iter / maxIter) ** 4)
        # Usar 'best_position' derivado del parámetro 'best'
        X1 = best_position + exp_term * (RB - 0.5) * (best_position - population)

    else:  # Etapa de ataque
        RL = 0.5 * levy(dim)
        # Usar 'best_position' derivado del parámetro 'best'
        X1 = best_position + CF * population * RL

    # Evaluación del fitness (usando 'fo' como nombre estándar de la función objetivo)
    for i in range(N):
        # Pasar la solución candidata X1[i] a la función 'fo'
        X1[i], f_newP1 = fo(X1[i]) # Usar 'fo'

        if f_newP1 <= fitness[i]:
            population[i] = X1[i]
            fitness[i] = f_newP1

    # Estrategia de escape (usando 'iter')
    r = np.random.rand()
    k = np.random.randint(N)
    Xrandom = population[k]

    RB = np.random.rand(N, dim)
    R2 = np.random.rand(N, dim)
    K = np.round(1 + np.random.rand(N))

    if r < 0.5:
        # Usar 'iter' en el cálculo
        escape_term = (1 - iter / maxIter) ** 2 * (2 * RB - 1)
        # Usar 'best_position' derivado del parámetro 'best'
        X2 = best_position + escape_term * population

    else:
        X2 = population + R2 * (Xrandom - K[:, None] * population)

    # Evaluación del fitness en la estrategia de escape (usando 'fo')
    for i in range(N):
         # Pasar la solución candidata X2[i] a la función 'fo'
        X2[i], f_newP2 = fo(X2[i]) # Usar 'fo'

        if f_newP2 <= fitness[i]:
            population[i] = X2[i]
            fitness[i] = f_newP2

    # Devolver la población actualizada
    return population