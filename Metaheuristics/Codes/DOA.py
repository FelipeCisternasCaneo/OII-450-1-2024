import numpy as np

def iterarDOA(maxIter, iter, dim, population, best, fo, lb, ub):
    """
    Dream Optimization Algorithm (DOA) adaptado desde código MATLAB original.
    
    LINK ARTICULO: https://doi.org/10.1016/j.cma.2024.117718
    CODIGO MATLAB: https://ww2.mathworks.cn/matlabcentral/fileexchange/178419-dream-optimization-algorithm-doa

    Parameters:
    - maxIter (int): Número total de iteraciones.
    - iter (int): Iteración actual.
    - dim (int): Dimensión del problema.
    - population (np.ndarray): Población actual (soluciones).
    - best (np.ndarray): Mejor solución global conocida.
    - fo (callable): Función objetivo. Debe retornar (solución, fitness).
    - lb (float o np.ndarray): Límite inferior.
    - ub (float o np.ndarray): Límite superior.

    Returns:
    - new_population (np.ndarray): Nueva población.
    - new_best (np.ndarray): Mejor solución global actualizada.
    """

    pop_size = population.shape[0]
    new_population = np.copy(population)

    if np.isscalar(lb):
        lb = np.full(dim, lb)
    if np.isscalar(ub):
        ub = np.full(dim, ub)

    fbest_global = fo(best)[1]  # Valor fitness de la mejor global
    new_best = best.copy()

    if iter < int(0.9 * maxIter):  # Exploración
        group_size = pop_size // 5
        for m in range(5):
            start_idx = m * group_size
            end_idx = (m + 1) * group_size if m < 4 else pop_size
            group_pop = population[start_idx:end_idx]

            group_fitness = [fo(ind)[1] for ind in group_pop]
            best_idx = np.argmin(group_fitness)
            best_in_group = group_pop[best_idx].copy()
            fbest_group = group_fitness[best_idx]

            for j in range(start_idx, end_idx):
                individual = population[j].copy()
                k = np.random.randint(max(1, dim // (8 * (m+1))), max(1, dim // (3 * (m+1))) + 1)
                indices = np.random.permutation(dim)[:k]

                if np.random.rand() < 0.9:
                    for h in indices:
                        delta = np.random.rand() * (ub[h] - lb[h]) + lb[h]
                        factor = (np.cos((iter + maxIter / 10) * np.pi / maxIter) + 1) / 2
                        individual[h] += delta * factor

                        # Manejo de límites
                        if individual[h] > ub[h] or individual[h] < lb[h]:
                            if dim > 15:
                                others = np.delete(np.arange(pop_size), j)
                                sel = np.random.choice(others)
                                individual[h] = population[sel, h]
                            else:
                                individual[h] = np.random.rand() * (ub[h] - lb[h]) + lb[h]
                else:
                    for h in indices:
                        sel = np.random.randint(pop_size)
                        individual[h] = population[sel, h]

                ind_fit = fo(individual)[1]
                if ind_fit < fbest_group:
                    new_population[j] = individual
                    fbest_group = ind_fit
                    best_in_group = individual.copy()

                # Actualizar mejor global
                if ind_fit < fbest_global:
                    fbest_global = ind_fit
                    new_best = individual.copy()

    else:  # Explotación
        for j in range(pop_size):
            individual = new_best.copy()
            k = np.random.randint(2, max(2, dim // 3) + 1)
            indices = np.random.permutation(dim)[:k]

            for h in indices:
                delta = np.random.rand() * (ub[h] - lb[h]) + lb[h]
                factor = (np.cos(iter * np.pi / maxIter) + 1) / 2
                individual[h] += delta * factor

                # Manejo de límites
                if individual[h] > ub[h] or individual[h] < lb[h]:
                    if dim > 15:
                        others = np.delete(np.arange(pop_size), j)
                        sel = np.random.choice(others)
                        individual[h] = population[sel, h]
                    else:
                        individual[h] = np.random.rand() * (ub[h] - lb[h]) + lb[h]

            ind_fit = fo(individual)[1]
            if ind_fit < fbest_global:
                new_population[j] = individual
                fbest_global = ind_fit
                new_best = individual.copy()

    return new_population