import numpy as np

# Honey Badger Algorithm (HBA)
# https://doi.org/10.1016/j.matcom.2021.08.013

def iterarHBA(maxIter, iter, population, best, fitness, fo, objective_type):
    C = 2
    beta = 6
    epsilon = 1e-16  # Valor pequeño para evitar divisiones por cero
    pi = np.pi
    
    N = len(population)
    alpha = C * np.exp(-iter / maxIter)
    Xnew = np.zeros_like(population)

    r6 = np.random.uniform(0, 1, N)
    F = np.where(r6 <= 0.5, 1, -1)

    r = np.random.uniform(0, 1, N)

    for i in range(N):
        di = best - population[i]
        
        if r[i] < 0.5:
            r_vals = np.random.uniform(0, 1, 5)
            
            if i != N - 1:
                S = np.power(population[i] - population[i + 1], 2)
            else:
                S = np.power(population[i] - population[0], 2)
                
            I = r_vals[0] * S / (4 * pi * np.power(np.abs(di) + epsilon, 2))

            Xnew[i] = (
                best
                + F[i] * beta * I * best
                + F[i] * r_vals[1] * alpha * di
                * np.abs(np.cos(2 * pi * r_vals[2]) * (1 - np.cos(2 * pi * r_vals[3])))
            )
        else:
            r7 = np.random.uniform(0, 1)
            Xnew[i] = best + F[i] * r7 * alpha * di

        # Evaluación del fitness de la nueva solución
        Xnew[i], newFitness = fo(Xnew[i])
        condition = newFitness < fitness[i] if objective_type == 'MIN' else newFitness > fitness[i]

        if condition:
            population[i] = Xnew[i]

    return np.array(population)
