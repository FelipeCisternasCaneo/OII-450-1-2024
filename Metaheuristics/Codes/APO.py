def iterarAPO(maxIter, iter, population, dim, fitness, fo):
    """
    APO adaptado para sistema basado en iteraciones.
    
    Parámetros:
    - maxIter (int): número máximo de iteraciones del sistema
    - iter (int): iteración actual
    - population (np.ndarray): población actual [ps, dim]
    - dim (int): dimensión del problema
    - fitness (np.ndarray): fitness actual
    - fo (callable): función objetivo que retorna (solución, fitness)
    
    Retorna:
    - population (np.ndarray): población actualizada
    """
    import numpy as np
    import math

    # --- Parámetros APO ---
    ps = population.shape[0]
    pf_max = 0.1
    np_pairs = 2
    epsilon = 1e-10
    xmin, xmax = -100, 100  # Estos deberían venir de lb/ub

    # --- Sortear población por fitness ---
    idx_sorted = np.argsort(fitness)
    protozoa_fitness = fitness[idx_sorted]
    protozoa_sorted = population[idx_sorted, :]

    # --- Determinar conjunto de dormancia/reproducción ---
    prop = pf_max * np.random.rand()
    size_dr = int(np.ceil(ps * prop))
    dr_set = set(np.random.choice(ps, size_dr, replace=False))

    protozoa_nuevo = np.zeros_like(population)
    
    # --- Ratio de progreso (basado en iteración actual) ---
    ratio = iter / maxIter

    # --- Generar nueva población ---
    for i in range(ps):
        if i in dr_set:
            # Dormancia o reproducción
            p_dr = 0.5 * (1 + math.cos((1 - (i + 1) / ps) * math.pi))
            
            if np.random.rand() < p_dr:
                # Dormancia (Eq. 11)
                protozoa_nuevo[i, :] = xmin + np.random.rand(dim) * (xmax - xmin)
            else:
                # Reproducción (Eq. 13)
                flag = 1 if np.random.rand() < 0.5 else -1
                mapeo_rep = np.zeros(dim)
                idx_rand = np.random.permutation(dim)[:int(np.ceil(np.random.rand() * dim))]
                mapeo_rep[idx_rand] = 1

                perturbacion = xmin + np.random.rand(dim) * (xmax - xmin)
                protozoa_nuevo[i, :] = (
                    protozoa_sorted[i, :] + flag * np.random.rand() * perturbacion * mapeo_rep
                )
        else:
            # Forrajeo (autotrófico o heterotrófico)
            factor = np.random.rand() * (1 + math.cos(ratio * math.pi))
            mapeo_f = np.zeros(dim)
            idx_rand = np.random.permutation(dim)[:int(np.ceil(dim * (i + 1) / ps))]
            mapeo_f[idx_rand] = 1

            prob_ah = 0.5 * (1 + math.cos(ratio * math.pi))
            vecinos = np.zeros((np_pairs, dim))

            if np.random.rand() < prob_ah:
                # Autotrófico (Eq. 1)
                j = np.random.randint(0, ps)
                
                for k in range(np_pairs):
                    if i == 0:
                        k_minus = 0
                        k_plus = np.random.randint(1, ps)
                    elif i == ps - 1:
                        k_minus = np.random.randint(0, ps - 1)
                        k_plus = ps - 1
                    else:
                        k_minus = np.random.randint(0, i)
                        k_plus = np.random.randint(i + 1, ps)
                    
                    peso_a = math.exp(
                        -abs(protozoa_fitness[k_minus] / (protozoa_fitness[k_plus] + epsilon))
                    )
                    vecinos[k, :] = peso_a * (
                        protozoa_sorted[k_minus, :] - protozoa_sorted[k_plus, :]
                    )
                
                vecinos_sum = np.sum(vecinos, axis=0) / np_pairs
                protozoa_nuevo[i, :] = (
                    protozoa_sorted[i, :]
                    + factor * (population[j, :] - protozoa_sorted[i, :] + vecinos_sum) * mapeo_f
                )
            else:
                # Heterotrófico (Eq. 7)
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
                X_near = (1 + flag * np.random.rand(dim) * (1 - ratio)) * protozoa_sorted[i, :]
                vecinos_sum = np.sum(vecinos, axis=0) / np_pairs
                protozoa_nuevo[i, :] = (
                    protozoa_sorted[i, :]
                    + factor * (X_near - protozoa_sorted[i, :] + vecinos_sum) * mapeo_f
                )

    # --- Corrección de límites ---
    protozoa_nuevo = np.clip(protozoa_nuevo, xmin, xmax)

    # --- Evaluación de nueva población ---
    # NOTA: La evaluación se hace en update_population(), no aquí
    # Solo retornamos las posiciones candidatas
    
    return protozoa_nuevo