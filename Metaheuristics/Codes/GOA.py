import numpy as np
import math

# Gannet Optimization Algorithm (GOA)
# https://doi.org/10.1016/j.matcom.2022.06.007

def V(x):
    return np.where(x <= np.pi, -(1 / np.pi) * x + 1, (1 / np.pi) * x - 1)

def levy(size, beta=1.5):
    u = np.random.uniform(0, 1, size)
    v = np.random.uniform(0, 1, size)

    gamma1 = math.gamma(1 + beta)
    gamma2 = math.gamma((1 + beta) / 2)
    sigma = ((gamma1 * np.sin(np.pi * beta / 2)) / (gamma2 * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)

    return 0.01 * ((u * sigma) / (np.abs(v) ** (1 / beta)))

def iterarGOA(maxIter, iter, dim, population, best, fitness, fo, objective_type):
    population = np.array(population)
    
    m, vel, c = 2.5, 1.5, 0.2
    t = 1 - (iter / maxIter)
    t2 = 1 + (iter / maxIter)

    Xr = population[np.random.randint(len(population))]
    Xm = np.mean(population, axis=0)

    MX = np.copy(population)
    r = np.random.uniform(0, 1, len(population))

    for i in range(len(population)):
        if r[i] > 0.5:  # Exploración
            q = np.random.uniform(0, 1)
            r_vals = np.random.uniform(0, 1, size=(dim, 4))

            if q >= 0.5:
                a = 2 * np.cos(2 * np.pi * r_vals[:, 0]) * t
                A = (2 * r_vals[:, 1] - 1) * a
                u1 = np.random.uniform(-a, a)
                u2 = A * (population[i] - Xr)
                MX[i] = population[i] + u1 + u2
            else:
                b = 2 * V(2 * np.pi * r_vals[:, 2]) * t
                B = (2 * r_vals[:, 3] - 1) * b
                v1 = np.random.uniform(-b, b)
                v2 = B * (population[i] - Xm)
                MX[i] = population[i] + v1 + v2
        else:  # Explotación
            r6 = np.random.uniform(0, 1)
            L = 0.2 + (2 - 0.2) * r6
            R = (m * vel ** 2) / L
            capturability = 1 / (R * t2)

            if capturability >= c:
                delta = capturability * np.abs(population[i] - best)
                MX[i] = t * delta * (population[i] - best) + population[i]
            else:
                p = levy(dim)
                MX[i] = best - (population[i] - best) * p * t

        MX[i], mxFitness = fo(MX[i])
        condition = mxFitness < fitness[i] if objective_type == 'MIN' else mxFitness > fitness[i]

        if condition:
            population[i] = MX[i]

    return population