import numpy as np

# Wombat Optimization Algorithm (WOM)
# https://doi.org/10.3390/math12071059

def eq4(population, fitness):
    """
    Identifica los candidatos para la fase de exploración.
    """
    N = len(population)
    CFP_mask = np.zeros((N, N), dtype=bool)
    for i in range(N):
        CFP_mask[i] = (fitness < fitness[i]) & (np.arange(N) != i)

    # Asegurar al menos un candidato
    for i in range(N):
        if not CFP_mask[i].any():
            CFP_mask[i, i] = True  # Incluye a sí mismo como candidato

    return CFP_mask

def eq5(population, CFP_mask, dim):
    """
    Calcula nuevas posiciones en la fase de exploración.
    """
    N = len(population)
    r_ij = np.random.rand(N, dim)
    I_ij = np.random.choice([1, 2], size=(N, dim))
    SFP_indices = [np.random.choice(np.flatnonzero(CFP_mask[i])) for i in range(N)]
    SFP = population[SFP_indices]
    return population + r_ij * (SFP - I_ij * population)

def eq7(population, dim, lb, ub, t):
    """
    Calcula nuevas posiciones en la fase de explotación.
    """
    r_ij = np.random.rand(len(population), dim)
    return population + (1 - 2 * r_ij) * ((ub - lb) / (t + 1))

def iterarWOM(iter, dim, population, fitness, fo, lb, ub):
    """
    WOM optimizado para evitar errores de formas inhomogéneas.
    """
    population = np.array(population, dtype=np.float64)
    fitness = np.array(fitness, dtype=np.float64)
    lb = np.array(lb, dtype=np.float64)
    ub = np.array(ub, dtype=np.float64)

    # Fase de exploración
    CFP_mask = eq4(population, fitness)
    newPositionsP1 = eq5(population, CFP_mask, dim)
    resultsP1 = [fo(ind) for ind in newPositionsP1]
    solutionsP1 = np.array([res[0] for res in resultsP1])
    fitnessP1 = np.array([res[1] for res in resultsP1])

    # Actualizar población y fitness
    improvementP1 = fitnessP1 < fitness
    population[improvementP1] = solutionsP1[improvementP1]
    fitness[improvementP1] = fitnessP1[improvementP1]

    # Fase de explotación
    newPositionsP2 = eq7(population, dim, lb, ub, iter)
    resultsP2 = [fo(ind) for ind in newPositionsP2]
    solutionsP2 = np.array([res[0] for res in resultsP2])
    fitnessP2 = np.array([res[1] for res in resultsP2])

    # Actualizar población y fitness
    improvementP2 = fitnessP2 < fitness
    population[improvementP2] = solutionsP2[improvementP2]
    fitness[improvementP2] = fitnessP2[improvementP2]

    return population