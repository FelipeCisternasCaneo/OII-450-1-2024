"""
Test A/B BEN: Legacy vs Universal Solver (Protocolo de Integridad)
=================================================================
Ejecuta la MISMA metaheurística con la MISMA semilla en ambos solvers
y compara que el fitness sea idéntico iteración por iteración.

Ejecutar desde la raíz del proyecto:
    python -m tests.test_ab_ben
"""

import numpy as np
import random as _random
import sys
import os

# Asegurar que el path del proyecto esté en sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _reset_ccmgo_globals():
    """CCMGO usa variables globales que contaminan entre ejecuciones."""
    try:
        import Metaheuristics.Codes.CCMGO as ccmgo_mod
        ccmgo_mod._initialized = False
        ccmgo_mod._FEs = 0
        ccmgo_mod._rec = 1
        ccmgo_mod._rM = None
        ccmgo_mod._rM_cost = None
        ccmgo_mod._best_cost = np.inf
        ccmgo_mod._best_M = None
    except (ImportError, AttributeError):
        pass


def _build_userData(mh_name):
    """Construye userData idéntico al que solverBEN.py construye."""
    userData = {}
    if mh_name == 'GOAT':
        userData['jump_prob'] = 0.3
        userData['filter_ratio'] = 0.5
    return userData


def run_legacy_ben(seed, mh_name, function, dim, lb, ub, pop_size, max_iter):
    """Ejecuta el flujo legacy de population_BEN exactamente como solverBEN."""
    _reset_ccmgo_globals()
    np.random.seed(seed)
    _random.seed(seed)

    from Problem.Benchmark.Problem import fitness as f
    from Solver.population.population_BEN import (
        initialize_population, evaluate_population,
        update_population, iterate_population
    )

    lb_list = [lb] * dim
    ub_list = [ub] * dim
    nfe_counter = [0]
    userData = _build_userData(mh_name)

    def fo(x):
        x = np.clip(x, lb_list, ub_list)
        nfe_counter[0] += 1
        return x, float(f(function, x))

    population, vel, pBestScore, pBest = initialize_population(mh_name, pop_size, dim, lb_list, ub_list)
    fitness = np.zeros(pop_size)
    fitness, best, bestFitness, pBest, pBestScore = evaluate_population(
        mh_name, population, fitness, dim, lb_list, ub_list, function, nfe_counter
    )

    convergence = [bestFitness]
    nfe_history = [nfe_counter[0]]

    for iter_num in range(1, max_iter + 1):
        population, vel, posibles_mejoras = iterate_population(
            mh_name, population, iter_num, max_iter, dim, fitness, best,
            vel=vel, pBest=pBest, ub=ub_list, lb=lb_list, fo=fo,
            userData=userData
        )

        population, fitness, best, bestFitness, div_t = update_population(
            population, fitness, dim, lb_list, ub_list, function, best, bestFitness,
            pBest, pBestScore, mh_name, posibles_mejoras, nfe_counter
        )

        convergence.append(bestFitness)
        nfe_history.append(nfe_counter[0])

    return np.array(convergence), np.array(nfe_history)


def run_universal_ben(seed, mh_name, function, dim, lb, ub, pop_size, max_iter):
    """Ejecuta el flujo del Universal Solver para BEN."""
    _reset_ccmgo_globals()
    np.random.seed(seed)
    _random.seed(seed)

    from Solver.domain_managers.ben_domain import BenDomainManager
    from Solver.metaheuristic_adapter import MetaheuristicAdapter
    from Solver.termination_manager import TerminationCriteria

    domain = BenDomainManager(function, dim, pop_size, lb, ub)
    adapter = MetaheuristicAdapter(mh_name, pop_size, dim, domain.lb, domain.ub)
    adapter.resolve_mh_name('BEN')
    termination = TerminationCriteria(max_iter=max_iter)

    population = domain.initialize_population()
    mh_state = adapter.initialize_state(population)

    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        fitness[i] = domain.evaluate(population[i])

    if mh_name == 'PSO' and mh_state['pBestScore'] is not None:
        for i in range(pop_size):
            if fitness[i] < mh_state['pBestScore'][i]:
                mh_state['pBestScore'][i] = fitness[i]
                mh_state['pBest'][i] = population[i].copy()
    if mh_name == 'TJO' and mh_state['pBest'] is not None:
        mh_state['pBest'] = population.copy()

    best, bestFitness = domain.find_best(population, fitness)
    userData = _build_userData(mh_name)

    convergence = [bestFitness]
    nfe_history = [domain.nfe]

    while not termination.is_met():
        termination.increment_iter()
        iter_num = termination.current_iter

        population, fitness, mh_state = adapter.run_iteration(
            population, fitness, mh_state, iter_num,
            max_iter, best, fo=domain.fo,
            userData=userData if userData else None
        )

        population, fitness = domain.process_new_population(
            population, fitness, mh_name, mh_state
        )

        mh_state = adapter.update_pbest(population, fitness, mh_state)
        best, bestFitness = domain.update_best(population, fitness, best, bestFitness)

        convergence.append(bestFitness)
        nfe_history.append(domain.nfe)
        termination.current_fe = domain.nfe

    return np.array(convergence), np.array(nfe_history)


def test_mh(mh_name, function='F1', dim=10, lb=-100, ub=100, pop=10, iters=30, seed=42):
    """Compara legacy vs universal para una MH específica."""
    print(f"\n{'='*60}")
    print(f"  TEST A/B: {mh_name} | {function} | dim={dim} | seed={seed}")
    print(f"{'='*60}")

    print(f"  [A] Legacy...", end=" ", flush=True)
    conv_legacy, nfe_legacy = run_legacy_ben(seed, mh_name, function, dim, lb, ub, pop, iters)
    print(f"Best={conv_legacy[-1]:.10e}")

    print(f"  [B] Universal...", end=" ", flush=True)
    conv_universal, nfe_universal = run_universal_ben(seed, mh_name, function, dim, lb, ub, pop, iters)
    print(f"Best={conv_universal[-1]:.10e}")

    fitness_match = np.allclose(conv_legacy, conv_universal, atol=1e-12)
    nfe_match = np.array_equal(nfe_legacy, nfe_universal)

    if fitness_match and nfe_match:
        print(f"   PASS — Convergencia idéntica ({len(conv_legacy)} puntos)")
    else:
        print(f"   FAIL")
        if not fitness_match:
            diffs = np.abs(conv_legacy - conv_universal)
            idx = np.argmax(diffs)
            print(f"     Diverge en iter {idx}: legacy={conv_legacy[idx]:.10e} vs universal={conv_universal[idx]:.10e}")
        if not nfe_match:
            print(f"     NFE: legacy={nfe_legacy[-1]} vs universal={nfe_universal[-1]}")

    return fitness_match and nfe_match


if __name__ == '__main__':
    ALL_MHS = [
        'GWO', 'PSO', 'WOA', 'SCA', 'FOX', 'NO', 'PSA', 'AOA', 'EOO',
        'EBWOA', 'RSA', 'LOA', 'QSO', 'WOM',
        'HBA', 'GOA', 'FLO', 'POA', 'TDO', 'SBOA', 'SHO', 'PGA',
        'SSO', 'DLO', 'DOA', 'EHO', 'DRA', 'WO', 'DHOA', 'ALA', 'CCMGO',
        'APO', 'CLO', 'TJO', 'GOAT', 'HLOA',
    ]

    results = {}
    for mh in ALL_MHS:
        try:
            results[mh] = test_mh(mh, function='F1', dim=10, pop=10, iters=30, seed=42)
        except Exception as e:
            print(f"    ERROR en {mh}: {type(e).__name__}: {e}")
            results[mh] = False

    print(f"\n{'='*60}")
    print(f"  RESUMEN FINAL A/B BEN")
    print(f"{'='*60}")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for mh, ok in sorted(results.items()):
        print(f"  {mh:>8s}: {'' if ok else ''}")
    print(f"\n  Total: {passed}/{total}")
    sys.exit(0 if passed == total else 1)

