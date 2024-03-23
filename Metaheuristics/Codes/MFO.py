import random
import numpy as np

'''
ALGORITMO MFO

IN:
    maxIter:   máximo de iteraciones del algoritmo
    t:          iteracion actual
    dimension:  dimension del problema (cantidad de restricciones)
    nsa:        número de agentes de búsqueda (cantidad de polillas)
    population:  posiciones de los agentes de búsqueda
    fitness:    arreglo de fitness de cada agente de busqueda
    bestSolution: mejor fitness encontrado

OUT:
    población perturbada.

* las ecuaciones corresponden al paper original del algoritmo
'''

def iterarMFO(maxIter, iter, dimension, nsa, population, bestSolutions, fitness, bestSolutionsFitness):
    #print(f"population:\n{population}")
    #print(f"fitness:\n{fitness}")

    # Recalcular cantidad de llamas, Eq. 3.14
    flameNum = int(np.ceil(nsa - (iter + 1) * ((nsa - 1) / maxIter)))

    mothPos = population.copy() # arreglo de posiciones de las polillas
    mothFit = fitness.copy() # arreglo de los fitness de cada polilla

    if iter == 0:
        # Ordenar la primera generación de polillas segun sus fitness
        order = mothFit.argsort(axis=0)
        mothFit = mothFit[order]
        mothPos = mothPos[order, :]

        # En la primera generacion, las llamas se corresponden con las posiciones de las polillas
        flames = np.copy(mothPos)
        flamesFit = np.copy(mothFit)

    else:
        # inicializae flamas, según cantidad permitida
        flames = bestSolutions[:flameNum, :]
        flamesFit = bestSolutionsFitness[:flameNum]

        # si no es la primera generación, se ordenan solo las llamas
        auxPop = np.vstack((flames, mothPos))       # se juntan las posiciones de llamas y polillas
        auxFit = np.hstack((flamesFit, mothFit))    # se juntan los fitnes de llamas y polillas
        order = auxFit.argsort(axis=0)              # se obtiene el orden de mejor fitness al peor
        auxFit = auxFit[order]                      # se ordenan los fitness
        auxPop = auxPop[order, :]                   # se ordenan las posiciones según su fitness

        # Actualizar las llamas
        flames = auxPop[:flameNum, :]
        flamesFit = auxFit[:flameNum]

    b = 1 # constante que define la forma de la espiral

    # actualizar r (constante de convergencia)
    # r decrece linealmente de -1 a -2 segun el avance en las iteraciones
    r = -1 + (iter + 1) * ((-1) / maxIter) 
    
    # preparar las llamas para mover las polillas
    temp1 = flames[:flameNum, :]                                                 # tomo las llamas activas
    temp2 = flames[flameNum - 1, :] * np.ones(shape=(nsa - flameNum, dimension))  # tomo y repito la ultima llama
    temp2 = np.vstack((temp1, temp2))                                           # lo junto para obtener un array del tamaño de la población
    
    # calcular la distancia entre las polillas y las llamas a las que siguen (Eq. 3.13)
    distanceToFlames = np.abs(temp2 - mothPos)

    # actualizar posición de las polillas
    for i in range(nsa):
        for j in range(dimension):
            
            t = (r - 1) * np.random.rand() + 1   # t es un número aleatorio entre r y 1

            # calcular nueva distancia de la polilla según Eq. 3.12
            mothPos[i,j] = distanceToFlames[i,j] * np.exp(b * t) * np.cos(t * 2 * np.pi) + temp2[i,j]

        
    return mothPos, flames