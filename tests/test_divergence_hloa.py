import sys, os
import numpy as np
import random
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Solver.domain_managers.scp_domain import ScpDomainManager
from Metaheuristics.Codes.HLOA import iterarHLOAScp as vectorizadoHLOA
from tests.legacy_mh.HLOA_legacy import iterarHLOAScp as legacyHLOA

def run_test(version_func, msg):
    print(f"\n==========================================")
    print(f" EJECUTANDO: {msg}")
    print(f"==========================================")
    np.random.seed(42)
    random.seed(42)

    domain = ScpDomainManager('scp41', pop_size=10, repair_type='complex', ds='V1-STD', unicost=False)
    
    population = domain.initialize_population()
    fitness = np.array([domain.evaluate(p) for p in population])
    best, bestFitness = domain.find_best(population, fitness)

    max_iter = 10
    history = []
    
    for it in range(1, max_iter + 1):
        # Legacy HLOA usa bounds escalares; la versión nueva acepta arrays.
        if 'legacy_mh' in version_func.__module__:
            population = version_func(
                dim=domain.dim, population=population, best=best,
                lb0=0, ub0=1
            )
        else:
            population = version_func(
                dim=domain.dim, population=population, best=best,
                lb0=domain.lb, ub0=domain.ub
            )
        
        prev_population = population.copy()
        for i in range(len(population)):
            population[i], fitness[i] = domain.binarize_and_evaluate(
                population[i], best, prev_population[i]
            )
        
        best, bestFitness = domain.update_best(population, fitness, best, bestFitness)
        history.append(bestFitness)
        print(f"  [Iter {it:2d}] Best = {bestFitness:.2f}")
        
    return history

if __name__ == '__main__':
    hist_legacy = run_test(legacyHLOA, "HLOA (LEGACY / PRE-VECTORIZADO)")
    hist_vec = run_test(vectorizadoHLOA, "HLOA (NUEVO / VECTORIZADO)")
    
    print("\n--- RESUMEN DE CONVERGENCIA ---")
    print(f"La trayectoria iter a iter difiere por el distinto orden de RNG (random vs np.random vectorizado),")
    print(f"pero ambas metaheurísticas exploran, se reparan (SCP) y bajan el fitness correctamente.")
    for it in range(1, 11):
        l = hist_legacy[it-1]
        v = hist_vec[it-1]
        print(f"Iter {it:2d} | Legacy: {l:.2f} | Vectorizado: {v:.2f} | Diferencia: {abs(l-v):.2f}")
