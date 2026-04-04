"""
Profiling rápido de metaheurísticas para identificar cuellos de botella.
Mide el tiempo promedio por iteración de cada MH con dimensión alta.
"""
import numpy as np
import random
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Solver.domain_managers.ben_domain import BenDomainManager
from Solver.metaheuristic_adapter import MetaheuristicAdapter
from Solver.termination_manager import TerminationCriteria


def profile_mh(mh_name, dim=100, pop=50, iters=30):
    """Perfila una MH y retorna el tiempo promedio por iteración (ms)."""
    np.random.seed(42)
    random.seed(42)

    domain = BenDomainManager('F1', dim, pop, -100, 100)
    adapter = MetaheuristicAdapter(mh_name, pop, dim, domain.lb, domain.ub)
    adapter.resolve_mh_name('BEN')

    population = domain.initialize_population()
    mh_state = adapter.initialize_state(population)

    fitness = np.zeros(pop)
    for i in range(pop):
        fitness[i] = domain.evaluate(population[i])

    if mh_name == 'PSO' and mh_state['pBestScore'] is not None:
        for i in range(pop):
            if fitness[i] < mh_state['pBestScore'][i]:
                mh_state['pBestScore'][i] = fitness[i]
                mh_state['pBest'][i] = population[i].copy()
    if mh_name == 'TJO' and mh_state['pBest'] is not None:
        mh_state['pBest'] = population.copy()

    best, bestFitness = domain.find_best(population, fitness)

    # Warmup
    for it in range(1, 4):
        population, fitness, mh_state = adapter.run_iteration(
            population, fitness, mh_state, it, iters, best, fo=domain.fo
        )
        population, fitness = domain.process_new_population(
            population, fitness, mh_name, mh_state
        )
        mh_state = adapter.update_pbest(population, fitness, mh_state)
        best, bestFitness = domain.update_best(population, fitness, best, bestFitness)

    # Medir
    times = []
    for it in range(4, iters + 1):
        t0 = time.perf_counter()
        population, fitness, mh_state = adapter.run_iteration(
            population, fitness, mh_state, it, iters, best, fo=domain.fo
        )
        population, fitness = domain.process_new_population(
            population, fitness, mh_name, mh_state
        )
        mh_state = adapter.update_pbest(population, fitness, mh_state)
        best, bestFitness = domain.update_best(population, fitness, best, bestFitness)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    return np.mean(times), np.std(times)


if __name__ == '__main__':
    ALL_MHS = [
        'GWO', 'PSO', 'WOA', 'SCA', 'FOX', 'NO', 'PSA', 'AOA', 'EOO',
        'EBWOA', 'RSA', 'LOA', 'QSO', 'WOM',
        'HBA', 'GOA', 'FLO', 'POA', 'TDO', 'SBOA', 'SHO', 'PGA',
        'SSO', 'DLO', 'DOA', 'EHO', 'DRA', 'WO', 'DHOA', 'ALA', 'CCMGO',
        'APO', 'CLO', 'TJO', 'GOAT', 'HLOA',
    ]

    print(f"{'MH':>8s}  {'Avg (ms)':>10s}  {'Std (ms)':>10s}  {'Status':>8s}")
    print("-" * 45)

    results = []
    for mh in ALL_MHS:
        try:
            avg, std = profile_mh(mh, dim=100, pop=50, iters=30)
            status = " SLOW" if avg > 50 else (" OK" if avg < 10 else " MED")
            results.append((mh, avg, std, status))
            print(f"{mh:>8s}  {avg:10.2f}  {std:10.2f}  {status}")
        except Exception as e:
            print(f"{mh:>8s}  {'ERROR':>10s}  {str(e)[:20]}")

    # Ranking
    results.sort(key=lambda x: x[1], reverse=True)
    print(f"\n{'='*45}")
    print("  TOP 10 MÁS LENTAS (candidatas a vectorizar)")
    print(f"{'='*45}")
    for mh, avg, std, status in results[:10]:
        print(f"  {mh:>8s}: {avg:.2f} ms/iter")

