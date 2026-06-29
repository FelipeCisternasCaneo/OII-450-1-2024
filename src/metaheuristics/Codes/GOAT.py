import numpy as np

def iterarGOAT(maxIter, iter, dim, population, best, fitness, fo, lb, ub,
               alpha=0.05, beta=0.5, jump_prob=0.1, filter_ratio=0.2, objective_type="min"):
    """
    Goat Optimization Algorithm (GOAT) - versión con mejoras suaves:
      - α, β, J adaptativos
      - Probabilidades dinámicas de explorar/explotar
      - Filtrado balanceado (global + local alrededor del best)
    """

    def evaluar(x):
        val = fo(x)
        if isinstance(val, tuple):
            return float(val[-1])
        return float(val)

    nPop = population.shape[0]

    # Normalizar límites
    lb = np.array([lb] * dim) if np.isscalar(lb) else np.array(lb)
    ub = np.array([ub] * dim) if np.isscalar(ub) else np.array(ub)

    # -------------------------------
    # Parámetros dinámicos (suaves)
    # -------------------------------
    alpha_t = alpha * (1 - iter / maxIter)       # menos ruido al final
    beta_t  = beta * (iter / maxIter)            # más explotación al final
    J_t     = jump_prob * (1 - iter / maxIter)   # menos saltos al final

    # Probabilidades dinámicas
    p_explore = 0.5 * (1 - iter / maxIter)       # explorar más al inicio
    p_exploit = 0.5 + 0.5 * (iter / maxIter)     # explotar más al final

    # -------------------------------
    # Actualización de cada cabra
    # -------------------------------
    for i in range(nPop):
        r1, r2, r3 = np.random.rand(3)

        # Exploración (probabilidad decreciente)
        if r1 < p_explore:
            population[i] = population[i] + alpha_t * np.random.randn(dim) * (ub - lb)

        # Explotación (probabilidad creciente)
        if r2 < p_exploit:
            population[i] = population[i] + beta_t * (best - population[i])

        # Saltos (probabilidad adaptativa)
        if r3 < J_t:
            rand_idx = np.random.randint(0, nPop)
            population[i] = population[i] + J_t * (population[rand_idx] - population[i])

        # Clamping
        population[i] = np.clip(population[i], lb, ub)

        # Evaluación y reemplazo si mejora
        fit_cand = evaluar(population[i])
        if objective_type.lower() == "min":
            if fit_cand < fitness[i]:
                fitness[i] = fit_cand
        else:
            if fit_cand > fitness[i]:
                fitness[i] = fit_cand

    # -------------------------------
    # Filtrado balanceado
    # -------------------------------
    n_filter = int(filter_ratio * nPop)
    if n_filter > 0:
        if objective_type.lower() == "min":
            worst_idx = np.argsort(fitness)[-n_filter:]
        else:
            worst_idx = np.argsort(fitness)[:n_filter]

        half = len(worst_idx) // 2
        for idx in worst_idx[:half]:
            # Reemplazo global (exploración)
            population[idx] = lb + (ub - lb) * np.random.rand(dim)
            fitness[idx] = evaluar(population[idx])
        for idx in worst_idx[half:]:
            # Reemplazo local (cerca del best)
            population[idx] = best + 0.1 * (ub - lb) * np.random.randn(dim)
            population[idx] = np.clip(population[idx], lb, ub)
            fitness[idx] = evaluar(population[idx])

    # -------------------------------
    # Actualizar mejor global
    # -------------------------------
    if objective_type.lower() == "min":
        idx = np.argmin(fitness)
        if fitness[idx] < evaluar(best):
            best = np.copy(population[idx])
    else:
        idx = np.argmax(fitness)
        if fitness[idx] > evaluar(best):
            best = np.copy(population[idx])

    return population, fitness, best
