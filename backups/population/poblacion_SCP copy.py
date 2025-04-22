import numpy as np
from Discretization import discretization as b

def initialize_population(mh, pop, instance):
    vel, pBestScore, pBest = None, None, None
    
    if mh == 'PSO':
        vel = np.zeros((pop, instance.getColumns()))
        pBestScore = np.full(pop, float("inf"))  # Más directo
        pBest = np.zeros((pop, instance.getColumns()))
    
    # Genero una población inicial binaria, esto ya que nuestro problema es binario
    population = np.random.randint(low = 0, high = 2, size = (pop, instance.getColumns()))
    
    return population, vel, pBestScore, pBest

def evaluate_population(mh, population, fitness, instance, pBest, pBestScore, repairType):
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
        
    solutionsRanking = np.argsort(fitness) # rankings de los mejores fitnes
    bestRowAux = solutionsRanking[0] # DETERMINO MI MEJOR SOLUCION Y LA GUARDO 
    best = population[bestRowAux].copy()
    bestFitness = fitness[bestRowAux]
    
    return fitness, best, bestFitness, pBest, pBestScore

def binarize_and_evaluate(mh, population, fitness, DS, best, matrixBin, instance, repairType, pBest, pBestScore, posibles_mejoras, fo):
    # Binarizo, calculo de factibilidad de cada individuo y calculo del fitness
    for i in range(population.__len__()):

        if mh != "GA":
            population[i] = b.aplicarBinarizacion(population[i], DS, best, matrixBin[i])

        flag, _ = instance.factibilityTest(population[i])
        
        if not flag: #solucion infactible
            population[i] = instance.repair(population[i], repairType)
            
        fitness[i] = instance.fitness(population[i])

        if mh == 'PSO':
            if fitness[i] < pBestScore[i]:
                pBest[i] = np.copy(population[i])
                
        if mh == 'LOA':
            _, fitn = fo(posibles_mejoras[i])
            
            if fitn < fitness[i]:
                population[i] = posibles_mejoras[i]
    
    return population, fitness, pBest

def update_best_solution(population, fitness, best, bestFitness):
    # Genero un vector de donde tendré mis soluciones rankeadas
    solutionsRanking = np.argsort(fitness) # rankings de los mejores fitness
    
    # conservo el best
    if fitness[solutionsRanking[0]] < bestFitness:
        bestFitness = fitness[solutionsRanking[0]]
        best = population[solutionsRanking[0]]
    
    return best, bestFitness