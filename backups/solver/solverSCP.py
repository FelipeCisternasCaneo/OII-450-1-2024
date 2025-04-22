import numpy as np
import os
import time

from Problem.SCP.problem import SCP

from Metaheuristics.imports import iterarGWO, iterarSCA, iterarWOA, iterarPSA, iterarGA
from Metaheuristics.imports import iterarPSO, iterarFOX, iterarEOO, iterarRSA, iterarGOA
from Metaheuristics.imports import iterarHBA, iterarTDO, iterarSHO, iterarSBOA, iterarEHO
from Metaheuristics.imports import iterarEBWOA, iterarFLO, iterarHLOAScp, iterarLOA, iterarNO
from Metaheuristics.imports import iterarPOA, IterarPO, iterarWOM, iterarQSO

from Diversity.imports import diversidadHussain, porcentajesXLPXPT
from Discretization import discretization as b
from Util import util
from BD.sqlite import BD

def solverSCP(id, mh, maxIter, pop, instances, DS, repairType, param):
    dirResult = './Resultados/'
    
    instance = SCP(instances)
    
    # tomo el tiempo inicial de la ejecucion
    initialTime = time.time()
    initializationTime1 = time.time()
    
    results = open(dirResult + mh + "_" + instances.split(".")[0] + "_" + str(id) + ".csv", "w")
    results.write(f'iter,fitness,time,XPL,XPT,DIV\n')
    
    vel = None
    pBestScore = None
    pBest = None
    
    print("AAAAAAAAAAAAAAAA")
    
    if mh == 'PSO':
        vel = np.zeros((pop, instance.getColumns()))
        pBestScore = np.zeros(pop)
        pBestScore.fill(float("inf"))
        pBest = np.zeros((pop, instance.getColumns()))
    
    # Genero una población inicial binaria, esto ya que nuestro problema es binario
    population = np.random.randint(low = 0, high = 2, size = (pop, instance.getColumns()))

    maxDiversity = diversidadHussain(population)
    XPL, XPT, state = porcentajesXLPXPT(maxDiversity, maxDiversity)
    
    # Genero un vector donde almacenaré los fitness de cada individuo
    fitness = np.zeros(pop)

    # Genero un vector dedonde tendré mis soluciones rankeadas
    solutionsRanking = np.zeros(pop)
    
    # Calculo de factibilidad de cada individuo y calculo del fitness inicial
    for i in range(population.__len__()):
        flag, aux = instance.factibilityTest(population[i])
        
        if not flag: #solucion infactible
            population[i] = instance.repair(population[i], repairType)
            
        fitness[i] = instance.fitness(population[i])
        
        if mh == 'PSO':
            if pBestScore[i] > fitness[i]:
                pBestScore[i] = fitness[i]
                pBest[i, :] = population[i, :].copy()
                
    print("i despues del for = " + str(i))
        
    solutionsRanking = np.argsort(fitness) # rankings de los mejores fitnes
    bestRowAux = solutionsRanking[0] # DETERMINO MI MEJOR SOLUCION Y LA GUARDO 
    best = population[bestRowAux].copy()
    bestFitness = fitness[bestRowAux]
    
    matrixBin = population.copy()
    
    initializationTime2 = time.time()
    
    # mostramos nuestro fitness iniciales
    print("------------------------------------------------------------------------------------------------------")
    print(f"{instances} - {DS} - {instance.getBlockSizes()} - best fitness inicial: {str(bestFitness)}")
    print("------------------------------------------------------------------------------------------------------")
    print("iteracion: " +
            str(0) +
            ", best: " + str(bestFitness) +
            ", optimo: " + str(instance.getOptimum()) +
            ", time (s): " + str(round(initializationTime2-initializationTime1,3)) +
            ", XPT: " + str(XPT) +
            ", XPL: " + str(XPL) +
            ", DIV: " + str(maxDiversity))
    
    results.write(f'0,{str(bestFitness)},{str(round(initializationTime2-initializationTime1, 3))},{str(XPL)},{str(XPT)},{maxDiversity}\n')
    
    # Función objetivo para GOA, HBA, TDO, SHO y SBOA
    def fo(x):
        print("i en el fo = " + str(i))
        x = b.aplicarBinarizacion(x, DS, best, matrixBin[i])
        x = instance.repair(x, repairType) # Reparación de soluciones
        
        return x, instance.fitness(x) # Return de la solución reparada y valor de función objetivo
    
    if mh == 'PO':
        iterarPO = IterarPO(fo, instance.getColumns(), pop, maxIter, 0, 1)
    
    for iter in range(0, maxIter):
        # obtengo mi tiempo inicial
        timerStart = time.time()

        # perturbo la poblacion con la metaheuristica, pueden usar SCA y GWO
        # en las funciones internas tenemos los otros dos for, for de individuos y for de dimensiones
        # print(population)
        
        if mh == "SCA":
            population = iterarSCA(maxIter, iter, instance.getColumns(), population, best)
            
        if mh == "GWO":
            population = iterarGWO(maxIter, iter, instance.getColumns(), population, fitness, 'MIN')
            
        if mh == 'WOA':
            population = iterarWOA(maxIter, iter, instance.getColumns(), population.tolist(), best.tolist())
            
        if mh == 'PSA':
            population = iterarPSA(maxIter, iter, instance.getColumns(), population.tolist(), best.tolist())
            
        if mh == "GA":
            cross = float(param.split(";")[0].split(":")[1])
            muta = float(param.split(";")[1].split(":")[1])
            population = iterarGA(population.tolist(), fitness, cross, muta)
            
        if mh == 'PSO':
            population, vel = iterarPSO(maxIter, iter, instance.getColumns(), population, best, pBest, vel, 1)
            
        if mh == 'FOX':
            population = iterarFOX(maxIter, iter, instance.getColumns(), population.tolist(), best.tolist())
            
        if mh == 'EOO':
            population = iterarEOO(maxIter, iter, population.tolist(), best.tolist())
            
        if mh == 'RSA':
            population = iterarRSA(maxIter, iter, instance.getColumns(), population.tolist(), best.tolist(), 0, 1)
            
        if mh == 'GOA':
            population = iterarGOA(maxIter, iter, instance.getColumns(), population, best.tolist(), fitness.tolist(), fo, 'MIN')
            
        if mh == 'HBA':
            population = iterarHBA(maxIter, iter, instance.getColumns(), population, best, fitness, fo, 'MIN')
            
        if mh == 'TDO':
            population = iterarTDO(maxIter, iter, instance.getColumns(), population.tolist(), fitness.tolist(), fo, 'MIN')
            
        if mh == 'SHO':
            population = iterarSHO(maxIter, iter, instance.getColumns(), population.tolist(), best.tolist(), fo, 'MIN')
            
        if mh == 'SBOA': 
            population = iterarSBOA(maxIter, iter, instance.getColumns(), population.tolist(), fitness.tolist(), best.tolist(), fo)
            
        if mh == 'EHO':
            lb = [0] * instance.getColumns()
            ub = [1] * instance.getColumns()
            
            population = iterarEHO(maxIter, iter, instance.getColumns(), population.tolist(), best.tolist(), lb, ub, fitness)
            
        if mh == 'EBWOA':
            population = iterarEBWOA(maxIter, iter, instance.getColumns(), population.tolist(), best.tolist(), 0, 1)
            
        if mh == 'FLO': 
            population = iterarFLO(maxIter, iter, instance.getColumns(), population, fitness, best, fo, 'MIN', 0, 1)
            
        if mh == 'HLOA':
            population = iterarHLOAScp(maxIter, iter, instance.getColumns(), population.tolist(), best.tolist(), 0, 1)
            
        if mh == "LOA":
            population, posibles_mejoras = iterarLOA(maxIter, population, best.tolist(), 0, 1, iter, instance.getColumns())
            
        if mh == 'NO':
            population = iterarNO(maxIter, iter, instance.getColumns(), population.tolist(), best.tolist())
            
        if mh == 'POA':
            population = iterarPOA(maxIter, iter, instance.getColumns(), population.tolist(), fitness.tolist(), fo, 0, 1, 'MIN')
            
        if mh == 'PO':
            iterarPO.pob(population.tolist(), iter)
            population = iterarPO.optimizer(iter)
            
        if mh == 'WOM':
            lb = [0] * instance.getColumns()
            ub = [1] * instance.getColumns()
            
            population = iterarWOM(maxIter, iter, instance.getColumns(), population.tolist(), fitness.tolist(), lb, ub, fo)
        
        if mh == 'QSO':
            population = iterarQSO(maxIter, iter, instance.getColumns(), population.tolist(), best.tolist(), 0, 1)
        
        # Binarizo, calculo de factibilidad de cada individuo y calculo del fitness
        for i in range(population.__len__()):

            if mh != "GA":
                population[i] = b.aplicarBinarizacion(population[i].tolist(), DS, best, matrixBin[i])

            flag, aux = instance.factibilityTest(population[i])
            # print(aux)
            if not flag: #solucion infactible
                population[i] = instance.repair(population[i], repairType)
                
            fitness[i] = instance.fitness(population[i])

            if mh == 'PSO':
                if fitness[i] < pBestScore[i]:
                    pBest[i] = np.copy(population[i])
                    
            if mh == 'LOA':
                func, fitn = fo(posibles_mejoras[i])
                if fitn < fitness[i]:
                    population[i] = posibles_mejoras[i]

        solutionsRanking = np.argsort(fitness) # rankings de los mejores fitness
        
        # conservo el best
        if fitness[solutionsRanking[0]] < bestFitness:
            bestFitness = fitness[solutionsRanking[0]]
            best = population[solutionsRanking[0]]
        
        matrixBin = population.copy()

        div_t = diversidadHussain(population)

        if maxDiversity < div_t: maxDiversity = div_t
            
        XPL, XPT, state = porcentajesXLPXPT(div_t, maxDiversity)

        timerFinal = time.time()
        # calculo mi tiempo para la iteracion t
        timeEjecuted = timerFinal - timerStart
        if (iter + 1) % (maxIter // 4) == 0:
        # if (iter+1) % 10 == 0:
            print("iteracion: " +
                str(iter + 1) +
                ", best: " + str(bestFitness) +
                ", optimo: " + str(instance.getOptimum()) +
                ", time (s): " + str(round(timeEjecuted, 3)) +
                ", XPT: " + str(XPT) +
                ", XPL: " + str(XPL) +
                ", DIV: " + str(div_t))
        
        results.write(f'{iter + 1},{str(bestFitness)},{str(round(timeEjecuted, 3))},{str(XPL)},{str(XPT)},{str(div_t)}\n')
        
    print("------------------------------------------------------------------------------------------------------")
    print("best fitness: " + str(bestFitness))
    print("Cantidad de columnas seleccionadas: " + str(sum(best)))
    print("------------------------------------------------------------------------------------------------------")
    
    finalTime = time.time()
    timeExecution = finalTime - initialTime
    
    print("Tiempo de ejecucion (s): " + str(timeExecution))
    
    results.close()
    
    binary = util.convert_into_binary(dirResult + mh + "_" + instances.split(".")[0] + "_" + str(id) + ".csv")
    
    fileName = mh + "_" + instances.split(".")[0]

    bd = BD()
    bd.insertarIteraciones(fileName, binary, id)
    bd.insertarResultados(bestFitness, timeExecution, best, id)
    bd.actualizarExperimento(id, 'terminado')
    
    os.remove(dirResult + mh + "_" + instances.split(".")[0] + "_" + str(id) + ".csv")