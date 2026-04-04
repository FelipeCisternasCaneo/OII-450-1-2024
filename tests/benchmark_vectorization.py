import sys, os
import numpy as np
import random
import time
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Metaheuristics.Codes.EBWOA import iterarEBWOA as vec_ebwoa
from Metaheuristics.Codes.EHO import iterarEHO as vec_eho
from Metaheuristics.Codes.HLOA import iterarHLOAScp as vec_hloa

from tests.legacy_mh.EBWOA_legacy import iterarEBWOA as leg_ebwoa
from tests.legacy_mh.EHO_legacy import iterarEHO as leg_eho
from tests.legacy_mh.HLOA_legacy import iterarHLOAScp as leg_hloa

def benchmark_mh(name, vec_func, leg_func, dim, pop_size, is_eho=False):
    # Setup dummy data
    np.random.seed(42)
    random.seed(42)
    population = np.random.rand(pop_size, dim)
    best = np.random.rand(dim)
    lb = np.zeros(dim)
    ub = np.ones(dim)
    fitness = np.random.rand(pop_size)
    def fo(x): return x, np.sum(x**2)
    
    max_iter = 100
    it = 1
    
    # Warmup
    if is_eho:
        _ = vec_func(max_iter, it, dim, population.copy(), best, lb, ub, fitness, fo)
        _ = leg_func(max_iter, it, dim, population.copy(), best, lb, ub, fitness, fo)
    elif name == 'HLOA':
        _ = vec_func(dim, population.copy(), best, lb, ub)
        _ = leg_func(dim, population.copy(), best, 0, 1)
    else:
        _ = vec_func(max_iter, it, dim, population.copy(), best, lb, ub)
        _ = leg_func(max_iter, it, dim, population.copy(), best, 0, 1)

    # Benchmark Legacy
    start = time.perf_counter()
    reps = 10
    for _ in range(reps):
        if is_eho:
            _ = leg_func(max_iter, it, dim, population.copy(), best, 0, 1, fitness, fo)
        elif name == 'HLOA':
            _ = leg_func(dim, population.copy(), best, 0, 1)
        else:
            _ = leg_func(max_iter, it, dim, population.copy(), best, 0, 1)
    t_leg = (time.perf_counter() - start) / reps
    
    # Benchmark Vectorized
    start = time.perf_counter()
    for _ in range(reps):
        if is_eho:
            _ = vec_func(max_iter, it, dim, population.copy(), best, lb, ub, fitness, fo)
        elif name == 'HLOA':
            _ = vec_func(dim, population.copy(), best, lb, ub)
        else:
            _ = vec_func(max_iter, it, dim, population.copy(), best, lb, ub)
    t_vec = (time.perf_counter() - start) / reps
    
    speedup = t_leg / t_vec
    return t_leg, t_vec, speedup

if __name__ == '__main__':
    # DIM=200 es suficiente para ver la diferencia masiva sin que el legacy tarde minutos
    DIM = 200
    POP = 30
    
    results = []
    print(f"Benchmarking (DIM={DIM}, POP={POP})...\n")
    
    for mh in [('EBWOA', vec_ebwoa, leg_ebwoa, False), 
               ('EHO', vec_eho, leg_eho, True),
               ('HLOA', vec_hloa, leg_hloa, False)]:
        name, v, l, eho = mh
        t_l, t_v, s = benchmark_mh(name, v, l, DIM, POP, is_eho=eho)
        results.append({'MH': name, 'Legacy (ms)': t_l*1000, 'Vectorized (ms)': t_v*1000, 'Speedup': s})
        print(f"{name:8}: Legacy={t_l*1000:7.2f}ms | Vec={t_v*1000:7.2f}ms | Speedup={s:7.2f}x")

    df = pd.DataFrame(results)
    df.to_csv('tests/benchmark_results.csv', index=False)
