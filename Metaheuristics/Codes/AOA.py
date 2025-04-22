import numpy as np

# Arithmetic Optimization Algorithm (AOA)
# https://doi.org/10.1016/j.cma.2020.113609

def iterarAOA(maxIter, iter, dim, population, best, lb0, ub0):
    population = np.array(population)
    best = np.array(best)
    new_solutions = np.copy(population)

    max_mop = 1
    min_mop = 0.2
    alpha = 5
    mu = 0.499
    upper_bound = ub0
    lower_bound = lb0

    # Aseguramos un límite inferior para mop
    mop = max(1 - ((iter) ** (1 / alpha) / (maxIter) ** (1 / alpha)), 1e-6)
    moa = min_mop + iter * ((max_mop - min_mop) / maxIter)

    num_solutions = population.shape[0]
    dimension = dim

    for i in range(num_solutions):
        for j in range(dimension):
            r1 = np.random.rand()
            if r1 < moa:
                r2 = np.random.rand()
                if r2 > 0.5:
                    new_solutions[i, j] = best[j] / (mop + np.finfo(float).eps) * ((upper_bound - lower_bound) * mu + lower_bound)
                else:
                    new_solutions[i, j] = best[j] * mop * ((upper_bound - lower_bound) * mu + lower_bound)
            else:
                r3 = np.random.rand()
                if r3 > 0.5:
                    new_solutions[i, j] = best[j] - mop * ((upper_bound - lower_bound) * mu + lower_bound)
                else:
                    new_solutions[i, j] = best[j] + mop * ((upper_bound - lower_bound) * mu + lower_bound)

        # Verificación de límites
        Flag_UB = new_solutions[i, :] > upper_bound
        Flag_LB = new_solutions[i, :] < lower_bound
        new_solutions[i, :] = (new_solutions[i, :] * (~(Flag_UB | Flag_LB))) + upper_bound * Flag_UB + lower_bound * Flag_LB

    return new_solutions
