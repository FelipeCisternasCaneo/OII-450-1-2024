def iterarAPO(maxIter, population, dim, fitness, fo):
    import numpy as np
    import math

    """
    Parámetros:
    - maxIter (int): número máximo de iteraciones (equivalente a max_FE/ps).
    - population (np.ndarray): población inicial [ps, dim].
    - dim (int): dimensión del problema.
    - fitness (np.ndarray): fitness inicial correspondiente a cada individuo.
    - fo (callable): función objetivo que retorna (solución, fitness).

    Retorna:
    - pob_protozoa (np.ndarray): población final.
    - fitness_values (np.ndarray): fitness final.
    - curva (np.ndarray): evolución del mejor fitness.
    """

    # --- Parámetros APO del paper ---
    ps = population.shape[0]
    pf_max = 0.1
    np_pairs = 2
    epsilon = 1e-10
    xmin, xmax = -100, 100

    # --- Inicialización ---
    pob_protozoa = population.copy()
    fitness_values = fitness.copy()

    best_idx = np.argmin(fitness_values)
    best_fitness = fitness_values[best_idx]

    curva = []
    FE = 0          # Evaluaciones de la función objetivo

    # --- Bucle principal (basado en evaluaciones) ---
    while FE < maxIter:
        #if FE % (ps * 10) == 0:
        #    print(f"Iteración: {FE//ps} | Mejor fitness: {best_fitness}")

        idx_sorted = np.argsort(fitness_values)
        protozoa_fitness = fitness_values[idx_sorted]
        protozoa_sorted = pob_protozoa[idx_sorted, :]

        prop = pf_max * np.random.rand()
        size_dr = int(np.ceil(ps * prop))
        dr_set = set(np.random.choice(ps, size_dr, replace=False))

        protozoa_nuevo = np.zeros_like(pob_protozoa)

        # --- Etapas de dormancia, reproducción y forrajeo ---
        for i in range(ps):
            if i in dr_set:
                # Dormancia o reproducción
                p_dr = 0.5 * (1 + math.cos((1 - (i + 1) / ps) * math.pi))
                if np.random.rand() < p_dr:
                    # Ecuación (11): Dormancia
                    protozoa_nuevo[i, :] = xmin + np.random.rand(1, dim) * (xmax - xmin)
                else:
                    # Ecuación (13): Reproducción
                    flag = 1 if np.random.rand() < 0.5 else -1
                    mapeo_rep = np.zeros(dim)
                    idx_rand = np.random.permutation(dim)[:int(np.ceil(np.random.rand() * dim))]
                    mapeo_rep[idx_rand] = 1

                    perturbacion = xmin + np.random.rand(1, dim) * (xmax - xmin)
                    protozoa_nuevo[i, :] = (
                        protozoa_sorted[i, :] + flag * np.random.rand() * perturbacion * mapeo_rep
                    )
            else:
                # Forrajeo (autotrófico o heterotrófico)
                ratio = FE / (maxIter * ps)
                factor = np.random.rand() * (1 + math.cos(ratio * math.pi))
                mapeo_f = np.zeros(dim)
                idx_rand = np.random.permutation(dim)[:int(np.ceil(dim * (i + 1) / ps))]
                mapeo_f[idx_rand] = 1

                prob_ah = 0.5 * (1 + math.cos(ratio * math.pi))
                vecinos = np.zeros((np_pairs, dim))

                if np.random.rand() < prob_ah:
                    # Autotrófico (Ecuación 1)
                    j = np.random.randint(0, ps)
                    for k in range(np_pairs):
                        k_minus = np.random.randint(0, i + 1) if i > 0 else 0
                        k_plus = np.random.randint(i, ps)
                        peso_a = math.exp(
                            -abs(protozoa_fitness[k_minus] / (protozoa_fitness[k_plus] + epsilon))
                        )
                        vecinos[k, :] = peso_a * (
                            protozoa_sorted[k_minus, :] - protozoa_sorted[k_plus, :]
                        )
                    vecinos_sum = np.sum(vecinos, axis=0) / np_pairs
                    protozoa_nuevo[i, :] = (
                        protozoa_sorted[i, :]
                        + factor * (pob_protozoa[j, :] - protozoa_sorted[i, :] + vecinos_sum) * mapeo_f
                    )
                else:
                    # Heterotrófico (Ecuación 7)
                    for k in range(np_pairs):
                        i_kminus = max(0, i - k - 1)
                        i_kplus = min(ps - 1, i + k + 1)
                        peso_h = math.exp(
                            -abs(protozoa_fitness[i_kminus] / (protozoa_fitness[i_kplus] + epsilon))
                        )
                        vecinos[k, :] = peso_h * (
                            protozoa_sorted[i_kminus, :] - protozoa_sorted[i_kplus, :]
                        )
                    flag = 1 if np.random.rand() < 0.5 else -1
                    X_near = (1 + flag * np.random.rand(1, dim) * (1 - ratio)) * protozoa_sorted[i, :]
                    vecinos_sum = np.sum(vecinos, axis=0) / np_pairs
                    protozoa_nuevo[i, :] = (
                        protozoa_sorted[i, :]
                        + factor * (X_near - protozoa_sorted[i, :] + vecinos_sum) * mapeo_f
                    )

        # --- Corrección de límites ---
        protozoa_nuevo = np.clip(protozoa_nuevo, xmin, xmax)

        # --- Evaluación de nueva población ---
        fitness_nuevo = np.zeros(ps)
        for i in range(ps):
            sol, fit = fo(protozoa_nuevo[i, :])
            protozoa_nuevo[i, :] = sol
            fitness_nuevo[i] = fit
        FE += ps  # Contador de evaluaciones

        # --- Actualización (Ecuación 14) ---
        mask = fitness_nuevo < protozoa_fitness
        protozoa_sorted[mask, :] = protozoa_nuevo[mask, :]
        protozoa_fitness[mask] = fitness_nuevo[mask]

        pob_protozoa = protozoa_sorted
        fitness_values = protozoa_fitness

        # --- Actualizar mejor global ---
        best_idx = np.argmin(fitness_values)
        best_fitness = fitness_values[best_idx]
        curva.append(best_fitness)

        #print("Después de incrementar FE:", FE)

    return pob_protozoa, fitness_values