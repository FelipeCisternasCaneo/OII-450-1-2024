import numpy as np
from scipy.special import gamma
import math

def equation_9(X_g_i, xBest_g, V_g_i, D_g_i, q9_c1, q9_c2, c, F):
    angle_g_i = np.arccos(np.dot(X_g_i, xBest_g) / (np.linalg.norm(X_g_i) * np.linalg.norm(xBest_g)))

    D_g_i_new = D_g_i + c * F * angle_g_i
    
    D_g_i_new = np.clip(D_g_i_new, -np.pi, np.pi)

    V_g_i_new = V_g_i + (q9_c1 * V_g_i * np.cos(D_g_i) - q9_c2 * (xBest_g - X_g_i))

    X_g_i_new = xBest_g + q9_c1 * V_g_i_new * np.cos(D_g_i) - q9_c2 * (xBest_g - X_g_i)
    
    return X_g_i_new, V_g_i_new, D_g_i_new

def equation_12(X_g_i, xBest_g, XBest, t, maxIter):
    m = np.random.uniform(-1, 1)
    
    n_t = -t / (2 * np.pi * maxIter)
    
    distA_t = XBest - X_g_i
    distB_t = xBest_g - X_g_i
    
    S1_t = m * n_t * np.cos(X_g_i) * distA_t
    
    S2_t = m * n_t * np.sin(X_g_i) * distB_t
    
    X_g_i_new = X_g_i + (S1_t + S2_t)
    
    return X_g_i_new

def equation_17(X_g_i, XBest, dim):
    # equation 19
    beta=1.5
    num = gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (num / den) ** (1 / beta)

    # equation 18
    u = np.random.normal(0, 1, size=dim)
    v = np.random.normal(0, 1, size=dim)
    stepsize = 0.01 * u * sigma / (np.abs(v) ** (1 / beta))

    distB_t = XBest - X_g_i

    return X_g_i + (np.pi / 3) * np.random.uniform(-1, 1) * distB_t * stepsize

def equation_20(X_g_i, xBest_g, XBest):
    m = np.random.uniform(-1, 1)
    
    distA_t = xBest_g - X_g_i
    distB_t = XBest - X_g_i
    
    Q_t = distA_t - distB_t
    
    M_t = Q_t * X_g_i
    
    X_g_i_new = m * Q_t * np.cos(X_g_i) + distA_t * np.cos(M_t)
    
    return X_g_i_new

def equation_23(X_g_i, xBest_g, XBest, q23_C1, q23_C2):
    return X_g_i + q23_C1 * (xBest_g - X_g_i) + q23_C2 * (XBest - X_g_i)

def iterarSRO(maxIter, iter, dim, population, best, fo, vel, userData):
    N = len(population)
    groupCount = 4
    groups = []

    for g in range(groupCount):
        groupSize = math.trunc(N / groupCount)
        groups.append(np.arange(g*groupSize, g*groupSize + ( (N % groupCount if g == groupCount - 1 else groupSize) )))

    if not userData:
        userData["no_change_counter"] = 0
    
    # equation 23 constants
    q23_C1, q23_C2 = np.random.uniform(0, 2, 2)

    # equation 9 constants
    q9_c1 = np.random.uniform(0, 1)
    q9_c2 = np.random.uniform(-1, 1)
    c = np.random.uniform(-2, 2)
    F = np.random.choice([1, -1])

    V_g_i = np.zeros_like(population[0])
    D_g_i = np.zeros_like(population[0])
    
    if iter % 20 == 0:
        for g in range(groupCount):
            np.arange(g*groupCount, g*groupCount + ( (N % groupCount if g == groupCount - 1 else groupCount) ))
            worst_ships = groups[g][np.argsort([fo(ship)[1] for ship in population[groups[g]]])[-3:]]
            for i in worst_ships:
                population[i] = equation_23(population[i], min(population[groups[g]], key=lambda x: fo(x)[1]), best, q23_C1, q23_C2)
    
    if userData["no_change_counter"] >= 10:
        for i in range(N):
            population[i], V_g_i, D_g_i = equation_9(population[i], min(population[groups[g]], key=lambda x: fo(x)[1]), V_g_i, D_g_i, q9_c1, q9_c2, c, F)
    else:
        for i in range(N):
            g = math.trunc(i / groupCount)
            if 1 <= i <= N / 2:
                population[i] = equation_12(population[i], min(population[groups[g]], key=lambda x: fo(x)[1]), best, iter, maxIter)
            elif N / 2 < i <= 0.9 * N:
                population[i] = equation_17(population[i], best, dim)
            else:
                population[i] = equation_20(population[i], min(population[groups[g]], key=lambda x: fo(x)[1]), best)

    newBest = best

    # see if best fitness change
    if fo(newBest)[1] <= fo(best)[1]:
        userData["no_change_counter"] += 1
    else:
        userData["no_change_counter"] = 0

    return population