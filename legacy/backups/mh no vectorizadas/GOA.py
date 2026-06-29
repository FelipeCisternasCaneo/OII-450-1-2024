import numpy as np
import random
import math

# Gannet optimization algorithm
# https://doi.org/10.1016/j.matcom.2022.06.007

def V(x):
    if x <= math.pi: return (-(1 / math.pi) * x + 1) # (0,π)
    elif x > math.pi: return ((1 / math.pi) * x - 1) # (π,2π)

def levy():
    beta = 1.5
    
    mu = random.uniform(0, 1)
    v = random.uniform(0, 1)

    gamma1 = math.gamma(1 + beta)
    gamma2 = math.gamma((1+beta)/2)
    seno = math.sin(math.pi * beta / 2)
    expo = 1 / beta
    
    sigma = ((gamma1 * seno) / (gamma2 * beta * 2 ** ((beta - 1) / 2))) ** expo
        
    return 0.01 * ((mu * sigma) / (abs(v) ** expo))

def iterarGOA(maxIter, it, dim, population, bestSolution, fitness, function, typeProblem):
    population = np.array(population)

    m = 2.5  # Peso en Kg del Gannet
    vel = 1.5  # Velocidad en el agua en m/s del Gannet
    c = 0.2  # Determina si se ejejuta movimiento levy o ajustes de trayectoria
    
    MX = population.copy()
    t = 1 - (it / maxIter)
    t2 = 1 + (it / maxIter)
    Xr = population[random.randint(0, len(population) - 1)]
    Xm = [np.mean(population[:, j]) for j in range(dim)]

    # ========= Exploracion =========
    for i in range(len(population)):
        r = random.uniform(0, 1)
        
        if r > 0.5:
                q = random.uniform(0, 1)
                
                if q >= 0.5:
                    for j in range(dim):
                        r2 = random.uniform(0, 1)
                        r4 = random.uniform(0, 1)

                        a = 2 * math.cos(2 * math.pi * r2) * t
                        A = (2 * r4 - 1) * a

                        u1 = random.uniform(-a, a)
                        u2 = A * (population[i,j] - Xr[j])

                        # Ecuacion 7a
                        MX[i][j] = population[i,j] + u1 + u2
                
                else:
                    for j in range(dim):
                        r3 = random.uniform(0, 1)
                        r5 = random.uniform(0, 1)

                        b = 2 * V(2 * math.pi * r3) * t
                        B = (2 * r5 - 1) * b

                        v1 = random.uniform(-b, b)
                        v2 = B * (population[i,j] - Xm[j])
                        # Ecuacion 7b
                        MX[i][j] = population[i,j] + v1 + v2

        # ========= Explotacion =========
        else:
                r6 = random.uniform(0, 1)
                L = 0.2 + (2 - 0.2) * r6
                R = (m * vel**2) / L
                capturability = 1 / (R * t2)
                
                # Caso ajustes exitosos
                if capturability >= c:
                    for j in range(dim):
                        delta = capturability * abs(population[i,j] - bestSolution[j])
                        # Ecuacion 17a
                        MX[i][j] = t * delta * (population[i,j] - bestSolution[j]) + population[i,j]

                # Caso movimiento Levy
                else:
                    for j in range(dim):
                        p = levy()
                        # Ecuacion 17b
                        MX[i][j] = bestSolution[j] - (population[i,j] - bestSolution[j]) * p * t
        
        MX[i],mxFitness = function(MX[i])
        
        if typeProblem == 'MIN': condition = mxFitness < fitness[i]
        
        elif typeProblem == 'MAX': condition = mxFitness > fitness[i]
        
        if condition: population[i] = MX[i]

    return population