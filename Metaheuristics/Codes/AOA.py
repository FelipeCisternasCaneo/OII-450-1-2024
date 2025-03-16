import numpy as np

# Arithmetic Optimization Algorithm (AOA)
# https://doi.org/10.1016/j.cma.2020.113609

def iterarAOA(max_iterations, current_iteration, dim, solutions, bestSolution):
    solutions = np.array(solutions)
    bestSolution = np.array(bestSolution)
    new_solutions = np.copy(solutions)

    max_mop = 1
    min_mop = 0.2
    alpha = 5
    mu = 0.499
    upper_bound = 1
    lower_bound = 0

    # Aseguramos un límite inferior para mop
    mop = max(1 - ((current_iteration) ** (1 / alpha) / (max_iterations) ** (1 / alpha)), 1e-6)
    moa = min_mop + current_iteration * ((max_mop - min_mop) / max_iterations)

    num_solutions = solutions.shape[0]
    dimension = dim

    for i in range(num_solutions):
        for j in range(dimension):
            r1 = np.random.rand()
            if r1 < moa:
                r2 = np.random.rand()
                if r2 > 0.5:
                    new_solutions[i, j] = bestSolution[j] / (mop + np.finfo(float).eps) * ((upper_bound - lower_bound) * mu + lower_bound)
                else:
                    new_solutions[i, j] = bestSolution[j] * mop * ((upper_bound - lower_bound) * mu + lower_bound)
            else:
                r3 = np.random.rand()
                if r3 > 0.5:
                    new_solutions[i, j] = bestSolution[j] - mop * ((upper_bound - lower_bound) * mu + lower_bound)
                else:
                    new_solutions[i, j] = bestSolution[j] + mop * ((upper_bound - lower_bound) * mu + lower_bound)

        # Verificación de límites
        Flag_UB = new_solutions[i, :] > upper_bound
        Flag_LB = new_solutions[i, :] < lower_bound
        new_solutions[i, :] = (new_solutions[i, :] * (~(Flag_UB | Flag_LB))) + upper_bound * Flag_UB + lower_bound * Flag_LB

    return new_solutions
