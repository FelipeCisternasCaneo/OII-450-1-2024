import time
import numpy as np
import os

from Diversity.imports import diversidadHussain,porcentajesXLPXPT
from Metaheuristics.imports import iterarGWO,iterarPSA,iterarSCA,iterarWOA
from Metaheuristics.imports import iterarPSO,iterarFOX,iterarEOO,iterarRSA,iterarGOA,iterarHBA,iterarTDO,iterarSHO
from Problem.Benchmark.Problem import fitness as f
from util import util
from BD.sqlite import BD


def solverB(id, mh, maxIter, pop, function, lb, ub, dim):
    
    dirResult = './Resultados/'

    # tomo el tiempo inicial de la ejecucion
    initialTime = time.time()
    
    initializationTime1 = time.time()

    print("------------------------------------------------------------------------------------------------------")
    print("Funcion benchmark a resolver: "+function)
    
    results = open(dirResult+mh+"_"+function+"_"+str(id)+".csv", "w")
    results.write(
        f'iter,fitness,time,XPL,XPT\n'
    )
    
    # Genero una población inicial binaria, esto ya que nuestro problema es binario
    population = np.random.uniform(low=lb, high=ub, size = (pop, dim))
    
    maxDiversity = diversidadHussain(population)
    XPL , XPT, state = porcentajesXLPXPT(maxDiversity, maxDiversity)
    
    # Genero un vector donde almacenaré los fitness de cada individuo
    fitness = np.zeros(pop)

    # Genero un vetor dedonde tendré mis soluciones rankeadas
    solutionsRanking = np.zeros(pop)
    
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    
    # calculo de factibilidad de cada individuo y calculo del fitness inicial
    for i in range(population.__len__()):
        for j in range(dim):
            population[i, j] = np.clip(population[i, j], lb[j], ub[j])            

        fitness[i] = f(function, population[i])
        
    solutionsRanking = np.argsort(fitness) # rankings de los mejores fitnes
    bestRowAux = solutionsRanking[0]
    # DETERMINO MI MEJOR SOLUCION Y LA GUARDO 
    best = population[bestRowAux].copy()
    bestFitness = fitness[bestRowAux]
    
    initializationTime2 = time.time()
    
    # mostramos nuestro fitness iniciales
    print("------------------------------------------------------------------------------------------------------")
    print("fitness incial: "+str(fitness))
    print("best fitness inicial: "+str(bestFitness))
    print("------------------------------------------------------------------------------------------------------")
    print("COMIENZA A TRABAJAR LA METAHEURISTICA "+mh)
    print("------------------------------------------------------------------------------------------------------")
    print("iteracion: "+
            str(0)+
            ", best: "+str(bestFitness)+
            ", mejor iter: "+str(fitness[solutionsRanking[0]])+
            ", peor iter: "+str(fitness[solutionsRanking[pop-1]])+
            ", time (s): "+str(round(initializationTime2-initializationTime1,3))+
            ", XPT: "+str(XPT)+
            ", XPL: "+str(XPL))
    results.write(
        f'0,{str(bestFitness)},{str(round(initializationTime2-initializationTime1,3))},{str(XPL)},{str(XPT)}\n'
    )

    bestPop = np.copy(population)

    # Función objetivo para GOA, HBA, TDO y SHO
    def fo(x):
        for j in range(x.__len__()): x[j] = np.clip(x[j], lb[j], ub[j]) # Reparación de soluciones
        return x,f(function,x) # Return de la solución reparada y valor de función objetivo
    
    for iter in range(0, maxIter):
        # print(f"bestPop = {bestPop[0,:]}")
        # print(f"pop = {population[0,:]}")
        # obtengo mi tiempo inicial
        timerStart = time.time()
        
        # perturbo la population con la metaheuristica, pueden usar SCA y GWO
        # en las funciones internas tenemos los otros dos for, for de individuos y for de dimensiones
        # print(population)
        if mh == "SCA":
            population = iterarSCA(maxIter, iter, dim, population.tolist(), best.tolist())
        if mh == "GWO":
            population = iterarGWO(maxIter, iter, dim, population.tolist(), fitness.tolist(), 'MIN')
        if mh == 'WOA':
            population = iterarWOA(maxIter, iter, dim, population.tolist(), best.tolist())
        if mh == 'PSA':
            population = iterarPSA(maxIter, iter, dim, population.tolist(), best.tolist())
        if mh == 'PSO':
            population = iterarPSO(maxIter, iter, dim, population.tolist(), best.tolist(),bestPop.tolist())
        if mh == 'FOX':
            population = iterarFOX(maxIter, iter, dim, population.tolist(), best.tolist())
        if mh == 'EOO':
            population = iterarEOO(maxIter, iter, population.tolist(), best.tolist())
        if mh == 'RSA':
            population = iterarRSA(maxIter, iter, dim, population.tolist(), best.tolist(),lb[0],ub[0])
        if mh == 'GOA':
            population = iterarGOA(maxIter, iter, dim, population, best.tolist(), fitness.tolist(),fo, 'MIN')
        if mh == 'HBA':
            population = iterarHBA(maxIter, iter, dim, population.tolist(), best.tolist(), fitness.tolist(),fo, 'MIN')
        if mh == 'TDO':
            population = iterarTDO(maxIter, iter, dim, population.tolist(), fitness.tolist(),fo, 'MIN')
        if mh == 'SHO':
            population = iterarSHO(maxIter, iter, dim, population.tolist(), best.tolist(),fo, 'MIN')

        # calculo de factibilidad de cada individuo y calculo del fitness inicial
        for i in range(population.__len__()):
            for j in range(dim):
                population[i, j] = np.clip(population[i, j], lb[j], ub[j])            

            fitness[i] = f(function, population[i])

            if mh == 'PSO':
                if fitness[i] < f(function, bestPop[i]):
                    bestPop[i] = np.copy(population[i])
            
        solutionsRanking = np.argsort(fitness) # rankings de los mejores fitness
        
        #Conservo el best
        if fitness[solutionsRanking[0]] < bestFitness:
            bestFitness = fitness[solutionsRanking[0]]
            best = population[solutionsRanking[0]]

        div_t = diversidadHussain(population)

        if maxDiversity < div_t:
            maxDiversity = div_t
            
        XPL , XPT, state = porcentajesXLPXPT(div_t, maxDiversity)

        timerFinal = time.time()
        # calculo mi tiempo para la iteracion t
        timeEjecuted = timerFinal - timerStart
        
        print("iteracion: "+
            str(iter+1)+
            ", best: "+str(bestFitness)+
            ", mejor iter: "+str(fitness[solutionsRanking[0]])+
            ", peor iter: "+str(fitness[solutionsRanking[pop-1]])+
            ", time (s): "+str(round(timeEjecuted,3))+
            ", XPT: "+str(XPT)+
            ", XPL: "+str(XPL))
        
        results.write(
            f'{iter+1},{str(bestFitness)},{str(round(timeEjecuted,3))},{str(XPL)},{str(XPT)}\n'
        )
    print("------------------------------------------------------------------------------------------------------")
    print("best fitness: "+str(bestFitness))
    print("------------------------------------------------------------------------------------------------------")
    finalTime = time.time()
    timeExecution = finalTime - initialTime
    print("Tiempo de ejecucion (s): "+str(timeExecution))
    results.close()
    
    binary = util.convert_into_binary(dirResult+mh+"_"+function+"_"+str(id)+".csv")

    fileName = mh+"_"+function

    bd = BD()
    bd.insertarIteraciones(fileName, binary, id)
    bd.insertarResultados(bestFitness, timeExecution, best, id)
    bd.actualizarExperimento(id, 'terminado')
    
    os.remove(dirResult+mh+"_"+function+"_"+str(id)+".csv")