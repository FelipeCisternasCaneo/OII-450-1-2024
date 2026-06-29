import numpy as np

# Fox-inspired Optimization Algorithm (FOX)
# https://doi.org/10.21203/rs.3.rs-1939478/v1

def iterarFOX(maxIter, it, dim, population, bestSolution):
    population = np.array(population)
    bestSolution = np.array(bestSolution)
    
    c1, c2 = 0.18, 0.82  # Constantes de ajuste
    a = 2 * (1 - (it / maxIter))  # Parámetro adaptativo

    # Generar valores aleatorios necesarios
    r = np.random.rand(population.shape[0])  # Aleatorio para individuos
    p = np.random.rand(population.shape[0])  # Aleatorio para individuos
    Time_S_T = np.random.rand(population.shape[0], dim)  # Tiempo aleatorio para todos los individuos

    # Máscaras para condiciones
    mask_r_gte_05 = r >= 0.5
    mask_r_lt_05 = ~mask_r_gte_05  # Complemento de `r >= 0.5`
    mask_p_gt_c1 = p > c1

    # Exploración para individuos que cumplen `r >= 0.5`
    if np.any(mask_r_gte_05):
        # Subconjunto de población y valores relevantes
        Time_S_T_selected = Time_S_T[mask_r_gte_05]
        Dist_S_T = (bestSolution / Time_S_T_selected) * Time_S_T_selected
        Dist_Fox_Prey = 0.5 * Dist_S_T
        tt = np.sum(Time_S_T_selected, axis=1) / dim
        t = tt / 2
        Jump = 0.5 * 9.81 * t ** 2

        # Aplicar condiciones `p > c1` y `p <= c1`
        selected_population = population[mask_r_gte_05]
        selected_population[mask_p_gt_c1[mask_r_gte_05]] = Dist_Fox_Prey[mask_p_gt_c1[mask_r_gte_05]] * Jump[mask_p_gt_c1[mask_r_gte_05]][:, None] * c1
        selected_population[~mask_p_gt_c1[mask_r_gte_05]] = Dist_Fox_Prey[~mask_p_gt_c1[mask_r_gte_05]] * Jump[~mask_p_gt_c1[mask_r_gte_05]][:, None] * c2
        population[mask_r_gte_05] = selected_population

    # Explotación para individuos que cumplen `r < 0.5`
    if np.any(mask_r_lt_05):
        population[mask_r_lt_05] = bestSolution + np.random.randn(np.sum(mask_r_lt_05), dim) * (t.min() * a)

    return population