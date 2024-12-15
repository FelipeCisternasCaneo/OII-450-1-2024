import numpy as np

# Quokka Swarm Optimization (QSO)
# https://doi.org/10.1515/jisys-2024-0051

def iterarQSO(maxIter, iter, dim, pop, best, lb, ub):
    temp = np.random.uniform(0.2, 0.44) * np.exp(-3 * iter / maxIter)
    hidro = np.random.uniform(0.3, 0.65) * np.exp(-3 * iter / maxIter)
    nitro = np.random.uniform(0, 1)
    drought = [np.random.uniform(0.5, 1.5) for _ in range(len(pop))]

    for i in range(len(pop)):
        for j in range(dim):
            delta_w = abs(best[j] - pop[i][j])
            delta_x = best[j] - pop[i][j]

            rand = np.random.uniform(0, 1)

            d_new = (temp+hidro)/(0.8 * drought[i]) + delta_w * rand * delta_x
            nueva_pos = pop[i][j] + d_new * nitro

            pop[i][j] = nueva_pos

    return np.array(pop)