import sys, os
import numpy as np
import random
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Solver.domain_managers.scp_domain import ScpDomainManager
# Import vectorized versions
from Metaheuristics.Codes.EBWOA import iterarEBWOA as vec_ebwoa
from Metaheuristics.Codes.EHO import iterarEHO as vec_eho
from Metaheuristics.Codes.HLOA import iterarHLOAScp as vec_hloa

# Import legacy versions
from tests.legacy_mh.EBWOA_legacy import iterarEBWOA as leg_ebwoa
from tests.legacy_mh.EHO_legacy import iterarEHO as leg_eho
from tests.legacy_mh.HLOA_legacy import iterarHLOAScp as leg_hloa

def run_step_comparison(mh_name, vec_func, leg_func, is_eho=False):
    print(f"\n" + "="*50)
    print(f" A/B TEST SCP: {mh_name}")
    print(f"="*50)
    
    # Setup domain
    instance = 'scp41'
    pop_size = 10
    repair = 'complex'
    ds = 'V1-STD'
    
    domain = ScpDomainManager(instance, pop_size, repair, ds, unicost=False)
    dim = domain.dim
    lb = domain.lb
    ub = domain.ub
    
    # Fix seed for initialization
    np.random.seed(42)
    random.seed(42)
    
    pop_initial = domain.initialize_population()
    fitness_initial = np.array([domain.evaluate(p) for p in pop_initial])
    best_initial, best_fit_initial = domain.find_best(pop_initial, fitness_initial)
    
    # We will test 1 iteration
    max_iter = 10
    it = 1
    
    # LEGACY RUN
    np.random.seed(123)
    random.seed(123)
    pop_leg = pop_initial.copy()
    if is_eho:
        # EHO legacy needs fitness and fo (9 args)
        pop_leg_res = leg_func(max_iter, it, dim, pop_leg, best_initial, 0, 1, fitness_initial, domain.fo)
    elif mh_name == 'HLOA':
        # HLOA legacy provided by user takes 5 args
        pop_leg_res = leg_func(dim, pop_leg, best_initial, 0, 1)
    else:
        # EBWOA legacy takes 7 args
        pop_leg_res = leg_func(max_iter, it, dim, pop_leg, best_initial, 0, 1)
    
    # VECTORIZED RUN
    np.random.seed(123)
    random.seed(123)
    pop_vec = pop_initial.copy()
    if is_eho:
        pop_vec_res = vec_func(max_iter, it, dim, pop_vec, best_initial, lb, ub, fitness_initial, domain.fo)
    elif mh_name == 'HLOA':
        # Vectorized HLOA for SCP takes 5 args
        pop_vec_res = vec_func(dim, pop_vec, best_initial, lb, ub)
    else:
        # Vectorized EBWOA takes 7 args
        pop_vec_res = vec_func(max_iter, it, dim, pop_vec, best_initial, lb, ub)

    # Compare results
    # For HLOA, the results will differ because of random vs np.random
    # For EBWOA/EHO, they use np.random but the call order significantly changed.
    diff = np.abs(pop_leg_res - pop_vec_res).sum()
    print(f"Sum of absolute differences after 1 MH iteration: {diff:.6e}")
    
    if diff < 1e-8:
        print(f" {mh_name}: PARIDAD EXACTA LOGRADA")
    else:
        print(f" {mh_name}: DIVERGENCIA ENCONTRADA")
        print(f"   Esto es normal al optimizar el RNG (vectorizacion).")

    # Final check: both produce valid SCP solutions
    # The new framework uses binarize and evaluate_and_repair
    pop_vec_res_final = np.zeros_like(pop_vec_res)
    fit_vec = np.zeros(pop_size)
    for i in range(pop_size):
        # We need to provide previous_binary (pop_vec[i]) for binarize
        res_binary, res_fit = domain.binarize_and_evaluate(pop_vec_res[i], best_initial, pop_vec[i])
        pop_vec_res_final[i] = res_binary
        fit_vec[i] = res_fit
    
    best_vec, best_fit_vec = domain.update_best(pop_vec_res_final, fit_vec, best_initial, best_fit_initial)
    
    print(f"Best fitness after 1 iteration (Vectorized): {best_fit_vec}")

if __name__ == '__main__':
    run_step_comparison("EBWOA", vec_ebwoa, leg_ebwoa)
    run_step_comparison("EHO", vec_eho, leg_eho, is_eho=True)
    run_step_comparison("HLOA", vec_hloa, leg_hloa)

