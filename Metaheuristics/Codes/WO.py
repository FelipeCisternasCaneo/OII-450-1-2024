import numpy as np
import math


# -------------------------------
# Utilidades estocásticas (externas en tu proyecto, pero implementadas aquí)
# -------------------------------
def levyFlight(dim, beta=1.5):
    """
    Genera un vector de vuelo de Lévy (Mantegna).
    Retorna un vector de longitud 'dim'.
    """
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2.0) /
             (math.gamma((1 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0))) ** (1.0 / beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    return u / (np.abs(v) ** (1.0 / beta))


def hal(index, base):
    """
    Radical inverse (Halton) para un índice y base dados.
    index: entero >= 1 (usa i+1 típicamente)
    base: entero primo pequeño (p.ej., 2,3,5,7,...)
    """
    result = 0.0
    f = 1.0 / base
    i = int(index)
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result


# -------------------------------
# Iteración del Walrus Optimizer
# -------------------------------
def iterarWO(maxIter, iter, dim, population, fitness, best, fo, lb, ub):
    """
    Iteración del Walrus Optimizer (WO).

    Parámetros:
      - maxIter (int)
      - iter (int)
      - dim (int)
      - population (np.ndarray) shape (N, dim)
      - fitness (np.ndarray) shape (N,)
      - best (np.ndarray) shape (dim,)
      - fo (callable): recibe x y retorna (x_recortado, f(x)) ya dentro de límites
      - lb (float | list | np.ndarray): límite inferior (escalar o vector de dim)
      - ub (float | list | np.ndarray): límite superior (escalar o vector de dim)

    Retorna:
      - np.ndarray: población actualizada (N, dim)
    """
    # Normalizar tipos y formas para evitar TypeError con listas
    population = np.asarray(population, dtype=float)
    best_position = np.asarray(best, dtype=float)

    # Asegurar lb/ub como arrays 1D de tamaño dim
    if np.isscalar(lb):
        lb = np.full(dim, float(lb), dtype=float)
    else:
        lb = np.asarray(lb, dtype=float).reshape(-1)
        if lb.size != dim:
            lb = np.full(dim, float(lb[0]), dtype=float)

    if np.isscalar(ub):
        ub = np.full(dim, float(ub), dtype=float)
    else:
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if ub.size != dim:
            ub = np.full(dim, float(ub[0]), dtype=float)

    # Clip inicial de seguridad
    population = np.clip(population, lb, ub)

    N = population.shape[0]
    GBestX = np.tile(best_position, (N, 1))

    # Parámetros WO
    P = 0.4
    F_number = int(round(N * P))         # hembras
    M_number = F_number                  # machos
    C_number = max(0, N - F_number - M_number)  # crías

    # Parámetros dinámicos
    Alpha = 1.0 - (iter / maxIter)
    Beta = 1.0 - 1.0 / (1.0 + np.exp((0.5 * maxIter - iter) / maxIter * 10.0))
    A = 2.0 * Alpha
    R = 2.0 * np.random.rand() - 1.0
    Danger_signal = A * R
    Safety_signal = np.random.rand()

    if abs(Danger_signal) >= 1.0:
        # Migración
        r3 = np.random.rand()
        idx1 = np.random.permutation(N)
        idx2 = np.random.permutation(N)
        migration_step = (Beta * (r3 ** 2)) * (population[idx1] - population[idx2])
        population = population + migration_step

    else:
        if Safety_signal >= 0.5:
            # Machos (secuencia de Halton)
            if M_number > 0:
                base = 7
                M = np.zeros((M_number, dim), dtype=float)
                for i_m in range(M_number):
                    for j in range(dim):
                        M[i_m, j] = hal(i_m + 1, base)
                # Escalar Halton [0,1] a [lb, ub]
                population[:M_number] = lb + M * (ub - lb)

            # Hembras
            if F_number > 0:
                for j_idx in range(M_number, M_number + F_number):
                    if M_number > 0:
                        i_ref = (j_idx - M_number) % M_number
                        ref = population[i_ref]
                    else:
                        ref = GBestX[j_idx]
                    population[j_idx] = population[j_idx] \
                        + Alpha * (ref - population[j_idx]) \
                        + (1.0 - Alpha) * (GBestX[j_idx] - population[j_idx])

            # Crías (Levy)
            if C_number > 0:
                for i_c in range(N - C_number, N):
                    Pr = np.random.rand()
                    o = GBestX[i_c] + population[i_c] * levyFlight(dim)
                    population[i_c] = Pr * (o - population[i_c])

        if Safety_signal < 0.5 and abs(Danger_signal) >= 0.5:
            # Escape rápido
            r4 = np.random.rand(N, 1)
            population = population * R - np.abs(GBestX - population) * (r4 ** 2)

        if Safety_signal < 0.5 and abs(Danger_signal) < 0.5:
            # Alimentación (búsqueda local intensiva)
            for i in range(N):
                for j in range(dim):
                    theta1 = np.random.rand()
                    a1 = Beta * np.random.rand() - Beta
                    b1 = np.tan(theta1 * np.pi)
                    X1 = best_position[j] - a1 * b1 * abs(best_position[j] - population[i, j])

                    theta2 = np.random.rand()
                    a2 = Beta * np.random.rand() - Beta
                    b2 = np.tan(theta2 * np.pi)
                    X2 = best_position[j] - a2 * b2 * abs(best_position[j] - population[i, j])

                    population[i, j] = 0.5 * (X1 + X2)

    # Reaplicar límites
    population = np.clip(population, lb, ub)

    # Evaluación del fitness (siguiendo tu interfaz fo)
    for i in range(N):
        population[i], f_new = fo(population[i])
        if f_new < fitness[i]:
            fitness[i] = f_new

    return population
