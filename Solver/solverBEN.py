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

def solverBEN(id, mh, maxIter, pop, function, lb, ub, dim, extra_params=None):
    dirResult = './Resultados/Transitorio/'

    os.makedirs(dirResult, exist_ok=True)

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
    
    # ========== SOLO AGREGAR ESTO ==========
    nfe_counter = [0]  # Contador simple
    # ========================================
    
    def fo_vectorized(x):
        x = np.clip(x, lb, ub)
        nfe_counter[0] += 1  # ← Solo incrementar aquí
        return x, float(f(function, x))
    
    population, vel, pBestScore, pBest = initialize_population(mh, pop, dim, lb, ub)
    fitness = np.zeros(pop)
    
    # NO cambiar esta función, solo pasar el contador
    fitness, best, bestFitness, pBest, pBestScore = evaluate_population(
        mh, population, fitness, dim, lb, ub, function, nfe_counter)
    
    # ========== SOLO AGREGAR nfe EN EL HEADER ==========
    results = open(dirResult + f"{mh}_{function}_{id}.csv", "w")
    results.write("iter,nfe,best_fitness,mean_fitness,std_fitness,time,XPL,XPT,DIV,GAP,RDP,ENT,Divj_mean,Divj_min,Divj_max\n")
    # ====================================================
    
    results_divj = open(dirResult + f"{mh}_{function}_{id}_divj.csv", "w")
    divj_header = ",".join([f"Divj_{j+1}" for j in range(dim)])
    results_divj.write(f"iter,{divj_header}\n")
    
    maxDiversity, XPL, XPT = initialize_diversity(population)
    initializationTime2 = time.time()
    
    # Iteración 0
    meanFitness0 = float(np.mean(fitness))
    stdFitness0  = float(np.std(fitness))
    gap0, rdp0   = compute_gap_rdp(bestFitness, optimo)
    divj_vec0, divj_mean0, divj_min0, divj_max0 = diversity_per_dimension(population)
    ent_avg0, ent_dim0 = population_entropy(population, bins=20, lb=lb, ub=ub)
    time0 = initializationTime2 - initializationTime1
    
    # ========== SOLO AGREGAR nfe_counter[0] ==========
    results.write(
        f"0,{nfe_counter[0]},{bestFitness:.6e},{meanFitness0:.6f},{stdFitness0:.6f},"
        f"{time0:.3f},{XPL:.6f},{XPT:.6f},{maxDiversity:.6f},"
        f"{gap0:.6f},{rdp0:.6f},{ent_avg0:.6f},{divj_mean0:.6f},{divj_min0:.6f},{divj_max0:.6f}\n"
    )
    # =================================================
    
    results_divj.write("0," + ",".join([f"{v:.6f}" for v in divj_vec0]) + "\n")
    print_initial(f"{function} {dim} {mh}", bestFitness)
    
    if mh == 'PO':
        iterarPO = IterarPO(fo_vectorized, dim, pop, maxIter, lb[0], ub[0])
    
    userData = {}
    if mh == "GOAT":
        if extra_params is not None:
            userData["jump_prob"] = float(extra_params.get("jump_prob", 0.3))
            userData["filter_ratio"] = float(extra_params.get("filter_ratio", 0.5))
        else:
            userData["jump_prob"] = 0.3
            userData["filter_ratio"] = 0.5
    
    # ========== EL BUCLE SIGUE SIENDO POR ITERACIONES ==========
    for iter in range(1, maxIter + 1):  # ← NO cambiar esto
        timerStart = time.time()
        
        population, vel, posibles_mejoras = iterate_population(
            mh, population, iter, maxIter, dim, fitness, best,
            vel=vel, pBest=pBest, ub=ub, lb=lb, fo=fo_vectorized, userData=userData
        )
        
        if mh == 'PO':
            iterarPO.pob(population, iter)
            population = iterarPO.optimizer(iter)
        
        # NO cambiar esta función, solo pasar el contador
        population, fitness, best, bestFitness, div_t = update_population(
            population, fitness, dim, lb, ub, function, best, bestFitness, 
            pBest, pBestScore, mh, posibles_mejoras, nfe_counter)
        
        div_t, maxDiversity, XPL, XPT = calculate_diversity(population, maxDiversity)
        
        meanFitness = float(np.mean(fitness))
        stdFitness = float(np.std(fitness))
        gap, rdp = compute_gap_rdp(bestFitness, optimo)
        divj_vec, divj_mean, divj_min, divj_max = diversity_per_dimension(population)
        ent_avg, ent_dim = population_entropy(population, bins=20, lb=lb, ub=ub)
        
        timerFinal = time.time()
        
        # ========== SOLO AGREGAR nfe_counter[0] ==========
        results.write(
            f"{iter},{nfe_counter[0]},{bestFitness:.6e},{meanFitness:.6f},{stdFitness:.6f},"
            f"{round(timerFinal - timerStart,3)},{XPL:.6f},{XPT:.6f},{div_t:.6f},"
            f"{gap:.6f},{rdp:.6f},{ent_avg:.6f},{divj_mean:.6f},{divj_min:.6f},{divj_max0:.6f}\n"
        )
        # =================================================
        
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