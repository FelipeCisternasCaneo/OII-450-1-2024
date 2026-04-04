import sys, os
import numpy as np
import random
import time
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Solver.domain_managers.scp_domain import ScpDomainManager
from Metaheuristics.Codes.EBWOA import iterarEBWOA as vec_ebwoa
from Metaheuristics.Codes.HLOA import iterarHLOAScp as vec_hloa
from Metaheuristics.Codes.EHO import iterarEHO as vec_eho

from tests.legacy_mh.EBWOA_legacy import iterarEBWOA as leg_ebwoa
from tests.legacy_mh.HLOA_legacy import iterarHLOAScp as leg_hloa
from tests.legacy_mh.EHO_legacy import iterarEHO as leg_eho

def run_experiment(mh_name, version_func, domain, n_runs=5, max_iter=30, is_eho=False):
    all_best_fitness = []
    
    for run in range(n_runs):
        # Cada corrida tiene su propia semilla para varianza
        seed = 42 + run
        np.random.seed(seed)
        random.seed(seed)
        
        pop = domain.initialize_population()
        fitness = np.array([domain.evaluate(p) for p in pop])
        best, best_fit = domain.find_best(pop, fitness)
        
        # Envoltorio para simular el solver
        for it in range(1, max_iter + 1):
            if is_eho:
                pop = version_func(max_iter, it, domain.dim, pop, best, domain.lb, domain.ub, fitness, domain.fo)
            elif mh_name == 'HLOA' and 'leg' in version_func.__module__:
                pop = version_func(domain.dim, pop, best, 0, 1)
            elif mh_name == 'HLOA': # Vectorized
                pop = version_func(domain.dim, pop, best, domain.lb, domain.ub)
            else: # EBWOA leg/vec
                # Legacy EBWOA expects scalars for lb/ub in my previous fix, but let's be safe
                lb_val = 0 if 'leg' in version_func.__module__ else domain.lb
                ub_val = 1 if 'leg' in version_func.__module__ else domain.ub
                pop = version_func(max_iter, it, domain.dim, pop, best, lb_val, ub_val)
            
            # Binarizar y Reparar (Simulando Universal Solver)
            for i in range(len(pop)):
                # Nota: para binarizar pasamos la pop anterior como proxy si no la tenemos trackeada
                pop[i], fitness[i] = domain.binarize_and_evaluate(pop[i], best, pop[i])
            
            best, best_fit = domain.update_best(pop, fitness, best, best_fit)
            
        all_best_fitness.append(best_fit)
        
    return np.mean(all_best_fitness), np.min(all_best_fitness), np.std(all_best_fitness)

if __name__ == '__main__':
    domain = ScpDomainManager('scp41', pop_size=10, repair_type='complex', ds='V1-STD', unicost=False)
    
    mhs = [
        ('EBWOA', vec_ebwoa, leg_ebwoa, False),
        ('HLOA', vec_hloa, leg_hloa, False),
        ('EHO', vec_eho, leg_eho, True)
    ]
    
    results = []
    print(f"Comparando convergencia (Fitness) en SCP41 (Runs=2, Iter=10)...\n")
    
    for name, v_func, l_func, is_eho in mhs:
        print(f"Probando {name}...")
        l_mean, l_min, l_std = run_experiment(name, l_func, domain, n_runs=2, max_iter=10, is_eho=is_eho)
        v_mean, v_min, v_std = run_experiment(name, v_func, domain, n_runs=2, max_iter=10, is_eho=is_eho)
        
        results.append({
            'MH': name,
            'Legacy_Mean': l_mean, 'Vec_Mean': v_mean,
            'Legacy_Best': l_min, 'Vec_Best': v_min,
            'Diff_Mean': v_mean - l_mean
        })

    df = pd.DataFrame(results)
    print("\n--- RESULTADOS DE CONVERGENCIA (FITNESS) ---")
    print(df.to_string(index=False))
    df.to_csv('tests/fitness_comparison.csv', index=False)
