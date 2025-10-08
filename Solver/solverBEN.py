import time
import numpy as np
import os
import json

from Diversity.Codes.diversity import initialize_diversity, calculate_diversity
from Diversity.imports import compute_gap_rdp, diversity_per_dimension, population_entropy
from Metaheuristics.imports import IterarPO
from Problem.Benchmark.Problem import fitness as f

from Solver.population.population_BEN import initialize_population, evaluate_population, update_population, iterate_population
from Util.console_logging import print_initial, print_iteration, print_final
from Util.csv_writer import open_csv, write_csv_row, close_csv
from Util.util import convert_into_binary

from BD.sqlite import BD

def solverBEN(id, mh, maxIter, pop, function, lb, ub, dim):
    dirResult = './Resultados/Transitorio/'
    
    os.makedirs(dirResult, exist_ok = True)
    
    bd = BD()
    
    if not isinstance(lb, list):
        lb = [lb] * dim
        
    if not isinstance(ub, list):
        ub = [ub] * dim
    
    initialTime = time.time()
    optimo = bd.obtenerOptimoInstancia(function)[0][0]
    
    if function == 'F8':
        optimo = optimo * dim
    
    initializationTime1 = time.time()
    
    def fo_vectorized(x):
        x = np.clip(x, lb, ub)
        return x, float(f(function, x))
    
    population, vel, pBestScore, pBest = initialize_population(mh, pop, dim, lb, ub)
    
    fitness = np.zeros(pop)
    
    fitness, best, bestFitness, pBest, pBestScore = evaluate_population(
        mh, population, fitness, dim, lb, ub, function)
    
    # Inicializamos el archivo de resultados
    results = open(dirResult + f"{mh}_{function}_{id}.csv", "w")
    results.write("iter,best_fitness,mean_fitness,std_fitness,time,XPL,XPT,DIV,GAP,RDP,ENT,Divj_mean,Divj_min,Divj_max\n")
    
    results_divj = open(dirResult + f"{mh}_{function}_{id}_divj.csv", "w")
    divj_header = ",".join([f"Divj_{j+1}" for j in range(dim)])
    results_divj.write(f"iter,{divj_header}\n")
    
    maxDiversity, XPL, XPT = initialize_diversity(population)

    initializationTime2 = time.time()
    
        # --- Write full iteration 0 row (all columns) ---
    meanFitness0 = float(np.mean(fitness))
    stdFitness0  = float(np.std(fitness))
    gap0, rdp0   = compute_gap_rdp(bestFitness, optimo)
    
    divj_vec0, divj_mean0, divj_min0, divj_max0 = diversity_per_dimension(population)
    ent_avg0, ent_dim0 = population_entropy(population, bins=20, lb=lb, ub=ub)
    
    time0 = initializationTime2 - initializationTime1
    
    results.write(
    f"0,{bestFitness:.6e},{meanFitness0:.6f},{stdFitness0:.6f},"
    f"{time0:.3f},{XPL:.6f},{XPT:.6f},{maxDiversity:.6f},"
    f"{gap0:.6f},{rdp0:.6f},{ent_avg0:.6f},{divj_mean0:.6f},{divj_min0:.6f},{divj_max0:.6f}\n"
    )
    
    results_divj.write("0," + ",".join([f"{v:.6f}" for v in divj_vec0]) + "\n")
    
    print_initial(f"{function} {dim} {mh}", bestFitness)
    
    if mh == 'PO':
        iterarPO = IterarPO(fo_vectorized, dim, pop, maxIter, lb[0], ub[0])
        
    userData = {}
        
    # Bucle de iteraciones
    for iter in range(1, maxIter + 1):
        timerStart = time.time()
        
        population, vel, posibles_mejoras = iterate_population(
            mh,               # 1
            population,       # 2
            iter,             # 3
            maxIter,          # 4
            dim,              # 5
            fitness,          # 6
            best,             # 7
            vel=vel,          # 8
            pBest=pBest,      # 9
            ub=ub,            # 10
            lb=lb,            # 11
            fo=fo_vectorized,  # 12
            userData=userData # 13
        )
        
        if mh == 'PO':
            iterarPO.pob(population, iter)
            population = iterarPO.optimizer(iter)
            
        population, fitness, best, bestFitness, div_t = update_population(
        population, fitness, dim, lb, ub, function, best, bestFitness, pBest, pBestScore, mh, posibles_mejoras)
        
        div_t, maxDiversity, XPL, XPT = calculate_diversity(population, maxDiversity)
        
        # --- Additional metrics ---
        meanFitness = float(np.mean(fitness))
        stdFitness = float(np.std(fitness))
        gap, rdp = compute_gap_rdp(bestFitness, optimo)

        divj_vec, divj_mean, divj_min, divj_max = diversity_per_dimension(population)
        ent_avg, ent_dim = population_entropy(population, bins=20, lb=lb, ub=ub)
        
        timerFinal = time.time()
        
        results.write(
            f"{iter},{bestFitness:.6e},{meanFitness:.6f},{stdFitness:.6f},"
            f"{round(timerFinal - timerStart,3)},{XPL:.6f},{XPT:.6f},{div_t:.6f},"
            f"{gap:.6f},{rdp:.6f},{ent_avg:.6f},{divj_mean:.6f},{divj_min:.6f},{divj_max:.6f}\n"
        )
        
        results_divj.write(f"{iter}," + ",".join([f"{v:.6f}" for v in divj_vec]) + "\n")
        
        print_iteration(iter, maxIter, bestFitness, optimo, timerFinal - timerStart, XPT, XPL, div_t)

    finalTime = time.time()
    
    print_final(bestFitness, initialTime, finalTime)
    
    results.close()
    results_divj.close()

    binary = convert_into_binary(dirResult + f"{mh}_{function}_{id}.csv")
    
    bd.insertarIteraciones(f"{mh}_{function}", binary, id)
    bd.insertarResultados(bestFitness, finalTime - initialTime, best, id)
    bd.actualizarExperimento(id, 'terminado')
    
    os.remove(dirResult + f"{mh}_{function}_{id}.csv")