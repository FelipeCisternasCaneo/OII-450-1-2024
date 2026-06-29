import numpy as np
from scipy.special import gamma

# Enhanced Beluga Whale Optimization Algorithm (EBWOA)
# https://doi.org/10.1007/s11518-024-5608-x
# Vectorizado: elimina loops dim-level, mantiene lógica idéntica.

def iterarEBWOA(maxIter, iter, dim, population, best, lb0, ub0):
    population = np.array(population, dtype=float)
    N = population.shape[0]

    WF = 0.1 - 0.05 * (iter / maxIter)
    kk = (1 - 0.5 * iter / maxIter) * np.random.rand(N)

    # Pre-computar constantes de Lévy (iguales para todos los individuos)
    alpha = 3 / 2
    sigma = (gamma(1 + alpha) * np.sin(np.pi * alpha / 2) /
             (gamma((1 + alpha) / 2) * alpha * 2**((alpha - 1) / 2)))**(1 / alpha)

    for i in range(N):
        if kk[i] > 0.5:
            # === EXPLORACIÓN ===
            r1 = np.random.rand()
            r2 = np.random.rand()
            RJ = np.random.randint(N)
            while RJ == i:
                RJ = np.random.randint(N)

            sin_val = np.sin(np.radians(r2 * 360))
            cos_val = np.cos(np.radians(r2 * 360))

            if dim <= N / 5:
                params = np.random.permutation(dim)[:2]
                p0, p1 = params[0], params[1]
                population[i, p0] += (population[RJ, p0] - population[i, p1]) * (r1 + 1) * sin_val
                population[i, p1] += (population[RJ, p0] - population[i, p1]) * (r1 + 1) * cos_val
            else:
                params = np.random.permutation(dim)
                half = dim // 2
                # Vectorizar los pares
                even_idx = np.arange(half) * 2
                odd_idx = even_idx + 1
                p_even = params[even_idx]
                p_odd = params[odd_idx]

                diff_even = (population[RJ, params[0]] - population[i, p_even])
                diff_odd = (population[RJ, params[0]] - population[i, p_odd])

                population[i, even_idx] = population[i, p_even] + diff_even * (r1 + 1) * sin_val
                population[i, odd_idx] = population[i, p_odd] + diff_odd * (r1 + 1) * cos_val

        else:
            # === EXPLOTACIÓN ===
            r3 = np.random.rand() - (2 * np.sin((np.pi / 2) * (iter / maxIter) ** 4))
            r4 = np.random.rand()
            C1 = 2 * r4 * (1 - iter / maxIter)

            RJ = np.random.randint(N)
            while RJ == i:
                RJ = np.random.randint(N)

            # Lévy flight vectorizado (una vez por individuo, no por dimensión)
            u = np.random.randn(dim) * sigma
            v = np.random.randn(dim)
            S = u / np.abs(v) ** (1 / alpha)
            KD = 0.05
            LevyFlight = KD * S

            # Actualizar TODAS las dimensiones de golpe
            population[i] = (r3 * best - r4 * population[i] +
                             C1 * LevyFlight * (population[RJ] - population[i]))

    # === WHALE FALL: vectorizado por individuo ===
    for i in range(N):
        if kk[i] <= WF:
            RJ = np.random.randint(N)
            r5 = np.random.rand()
            r6 = np.random.rand()
            r7 = np.random.rand()
            C2 = 2 * N * WF
            stepsize2 = r7 * (ub0 - lb0) * np.exp(-C2 * iter / maxIter)
            # Vectorizado: todas las dimensiones de una vez
            population[i] = r5 * population[i] - r6 * population[RJ] + stepsize2

    return population