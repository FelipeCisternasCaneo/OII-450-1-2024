import numpy as np
from scipy.special import gamma

# Enhanced Beluga Whale Optimization Algorithm (EBWOA)
# https://doi.org/10.1007/s11518-024-5608-x

def iterarEBWOA(maxIter, iter, dim, population, best, lb0, ub0):
    population = np.array(population)
    WF = 0.1 - 0.05 * (iter / maxIter)  # Whale fall probability
    kk = (1 - 0.5 * iter / maxIter) * np.random.rand(population.shape[0])
    
    for i in range(population.shape[0]):
        for j in range(dim):
            #print("population: ", population)
            if kk[i] > 0.5:  # Exploration phase
                r1 = np.random.rand()  # Número aleatorio
                r2 = np.random.rand()  # Número aleatorio
                RJ = np.random.randint(population.shape[0])  # Selección por ruleta
                # Asegurarse de que RJ sea diferente a i
                while RJ == i:
                    RJ = np.random.randint(population.shape[0])
                    
                if dim <= population.shape[0] / 5:
                    params = np.random.permutation(dim)[:2]  # Selección de 2 parámetros aleatorios
                    
                    # Actualización de las posiciones de los parámetros seleccionados
                    population[i, params[0]] = population[i, params[0]] + (population[RJ, params[0]] - population[i, params[1]]) * (r1 + 1) * np.sin(np.radians(r2 * 360))
                    population[i, params[1]] = population[i, params[1]] + (population[RJ, params[0]] - population[i, params[1]]) * (r1 + 1) * np.cos(np.radians(r2 * 360))
                    
                else:
                    params = np.random.permutation(dim)  # Selección de todos los parámetros de forma aleatoria
                    
                    for j in range(dim // 2):
                        # Actualización de las posiciones de los pares de parámetros
                        population[i, 2 * j] = population[i, params[2*j]] + (population[RJ, params[0]] - population[i, params[2*j]]) * (r1 + 1) * np.sin(np.radians(r2 * 360))
                        population[i, 2 * j + 1] = population[i, params[2 * j + 1]] + (population[RJ, params[0]] - population[i, params[2 * j + 1]]) * (r1 + 1) * np.cos(np.radians(r2 * 360)) 
                               
            else: # Explotation phase
                # Variables aleatorias
                r3 = np.random.rand() - (2 * np.sin((np.pi / 2) * (iter / maxIter) ** 4))
                r4 = np.random.rand()
                # Cálculo de C1
                C1 = 2 * r4 * (1 - iter / maxIter)
                # Selección por ruleta (aseguramos que RJ sea diferente de i)
                RJ = np.random.randint(0, population.shape[0])
                
                while RJ == i:
                    RJ = np.random.randint(0, population.shape[0])
                    
                alpha = 3 / 2
                sigma = (gamma(1 + alpha) * np.sin(np.pi * alpha / 2) / (gamma((1 + alpha) / 2) * alpha * 2**((alpha - 1) / 2)))**(1 / alpha)
                u = np.random.randn(1, dim) * sigma
                v = np.random.randn(1, dim)
                S = u / np.abs(v) ** (1 / alpha)
                # Parámetro de escala KD
                KD = 0.05
                LevyFlight = KD * S
                # Actualizar la posición del beluga i-ésimo
                population[i, j] = r3 * best[j] - r4 * population[i, j] + C1 * LevyFlight[0, j] * (population[RJ, j] - population[i, j])
                
    for i in range(population.shape[0]):
        for j in range(dim):
            if kk[i] <= WF:
                RJ = np.random.randint(population.shape[0])
                r5 = np.random.rand()
                r6 = np.random.rand()
                r7 = np.random.rand()
                C2 = 2 * population.shape[0] * WF
                stepsize2 = r7 * (ub0 - lb0) * np.exp(-C2 * iter / maxIter)
                population[i, j] = r5 * population[i, j] - r6 * population[RJ, j] + stepsize2
                
    return population       