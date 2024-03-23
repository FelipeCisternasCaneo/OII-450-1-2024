import numpy as np
import random

def iterarTDO(maxIter, it, dim, population, fitness, function, typeProblem):
    N = len(population)
    population = np.array(population)

    for i in range(N):
        r = random.uniform(0.0, 1.0)

        # Se escoge un demonio de Tasmania aleatorio de entre la población
        k = np.random.choice(np.delete(np.arange(N), i))
        CPi = population[k]
        xNew = np.copy(population[i])

        if typeProblem == 'MIN': condition = function(CPi)[1] < fitness[i]
        elif typeProblem == 'MAX': condition = function(CPi)[1] > fitness[i]
        for j in range(dim):
            # Definimos si este se debe alejar o acercar, en función de su evaluación según la FO
            # Si el demonio escogido evaluado resulta ser mejor, se acerca
            if condition:
                I = random.randint(1, 2)
                xNew[j] = population[i][j] + random.uniform(0.0, 1.0) * (CPi[j] - I * population[i][j])
            # Si el demonio escogido evaluado resulta ser peor, se aleja
            else:
                xNew[j] = population[i][j] + random.uniform(0.0, 1.0) * (population[i][j] - CPi[j])

        xNew,fitnessNew = function(xNew)
        if typeProblem == 'MIN': condition = fitnessNew < fitness[i]
        elif typeProblem == 'MAX': condition = fitnessNew > fitness[i]
        if condition: population[i] = np.copy(xNew)

        if r >= 0.5:
            # EXPLOTACIÓN
            R = 0.01 * (1 - (it / maxIter))

            # Se realiza la búsqueda local (Nueva posición)
            for j in range(dim):
                xNew[j] = population[i][j] + (2 * random.uniform(0.0, 1.0) - 1) * R * xNew[j]

            xNew,fitnessNew = function(xNew)
            if typeProblem == 'MIN': condition = fitnessNew < fitness[i]
            elif typeProblem == 'MAX': condition = fitnessNew > fitness[i]
            if condition: population[i] = np.copy(xNew)

    return population