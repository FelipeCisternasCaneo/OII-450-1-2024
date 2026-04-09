"""
Universal Solver
================
Solver unificado que orquesta CUALQUIER metaheurística sobre CUALQUIER dominio
de problema usando Inyección de Dependencias:

    domain      → Cualquier subclase de BaseDomainManager
    adapter     → MetaheuristicAdapter (envuelve la MH elegida)
    termination → TerminationCriteria (iter, FE, o ambos)

El solver interactúa EXCLUSIVAMENTE con la interfaz BaseDomainManager,
sin conocer subclases concretas (ScpDomainManager, BenDomainManager, etc.).

Flujo del bucle principal:
    1. adapter.run_iteration()         → Mueve la población
    2. domain.process_new_population() → Evalúa según el dominio
    3. adapter.update_pbest()          → Actualiza memoria personal (PSO/TJO)
    4. domain.update_best()            → Actualiza mejor global
    5. Métricas + logging              → Diversidad, Gap, Entropía, CSV

Uso:
    from Solver.universal_solver import universal_solver
    from Solver.domain_managers.ben_domain import BenDomainManager
    from Solver.termination_manager import TerminationCriteria

    domain = BenDomainManager('F1', dim=30, pop_size=25, lb=-100, ub=100)
    tc = TerminationCriteria(max_iter=500)
    universal_solver(id=1, mh_name='PSO', domain=domain, termination=tc)
"""

import os
import time
import random
import numpy as np

from Solver.termination_manager import TerminationCriteria, resolve_effective_max_iter
from Solver.metaheuristic_adapter import MetaheuristicAdapter, MH_NEEDS_PBEST
from Solver.domain_managers.base_domain import BaseDomainManager

from Diversity.Codes.diversity import initialize_diversity, calculate_diversity
from Diversity.imports import (
    compute_gap_rdp,
    diversity_per_dimension,
    population_entropy,
)
from Metaheuristics.imports import IterarPO
from Util.console_logging import print_initial, print_iteration, print_final
from Util.util import convert_into_binary
from BD.sqlite import BD


def universal_solver(id, mh_name, domain, termination, extra_params=None):
    """
    Solver universal para cualquier combinación de MH + dominio.

    Args:
        id:           ID del experimento en la base de datos.
        mh_name:      Nombre de la metaheurística ('PSO', 'GWO', 'TJO', etc.).
        domain:       Instancia de BaseDomainManager (BenDomainManager o ScpDomainManager).
        termination:  Instancia de TerminationCriteria.
        extra_params: Parámetros extra de la MH (dict). Ejemplos:
                      - GOAT: {'jump_prob': 0.3, 'filter_ratio': 0.5}
                      - GA (SCP): {'cross': 0.8, 'muta': 0.5}
    """
    # ==================== SETUP ====================
    bd = BD()
    dirResult = "./Resultados/Transitorio/"
    os.makedirs(dirResult, exist_ok=True)

    # Nombre base para archivos CSV y BD (polimórfico)
    file_base = f"{mh_name}_{domain.label}"

    # Crear adaptador
    adapter = MetaheuristicAdapter(
        mh_name, domain.pop_size, domain.dim, domain.lb, domain.ub
    )
    adapter.resolve_mh_name(domain.domain_type)

    # Iteraciones efectivas para MH que requieren maxIter internamente
    # (para coeficientes como w en PSO, a en GWO, etc.)
    effective_max_iter = resolve_effective_max_iter(
        termination, domain.pop_size, mh_name
    )

    # ==================== RNG SEEDING ====================
    # Semilla determinista derivada del ID del experimento.
    # Cada corrida (ID distinto) produce resultados diferentes pero reproducibles.
    # Para replicar: usar el mismo ID con los mismos parámetros.
    seed = id  # El ID es único por corrida en la BD
    np.random.seed(seed)
    random.seed(seed)

    # ==================== INICIALIZACIÓN ====================
    initialTime = time.time()
    optimum = domain.get_optimum()
    initializationTime1 = time.time()

    # Resetear NFE del dominio
    domain.reset_nfe()

    # Generar población inicial
    population = domain.initialize_population()

    # Inicializar estado de la MH (vel, pBest, etc.)
    mh_state = adapter.initialize_state(population)

    # Inicializar estado de iteración del dominio
    domain.set_iteration_state(np.zeros(domain.dim), population.copy())

    # Evaluar población inicial (polimórfico: SCP repara, BEN evalúa directo)
    fitness = np.zeros(domain.pop_size)
    for i in range(domain.pop_size):
        population[i], fitness[i] = domain.evaluate_and_repair(population[i])

    # Inicializar pBest con evaluación inicial
    if mh_name == "PSO" and mh_state["pBestScore"] is not None:
        for i in range(domain.pop_size):
            if fitness[i] < mh_state["pBestScore"][i]:
                mh_state["pBestScore"][i] = fitness[i]
                mh_state["pBest"][i] = population[i].copy()

    if mh_name == "TJO" and mh_state["pBest"] is not None:
        mh_state["pBest"] = population.copy()

    # Encontrar mejor solución inicial
    best, bestFitness = domain.find_best(population, fitness)

    # Actualizar estado del dominio con el mejor encontrado
    domain.set_iteration_state(best, population.copy())

    initializationTime2 = time.time()

    # ==================== CSV SETUP (Buffered I/O) ====================
    results_path = os.path.join(dirResult, f"{file_base}_{id}.csv")
    results_divj_path = os.path.join(dirResult, f"{file_base}_{id}_divj.csv")

    results = open(results_path, "w")
    results_divj = open(results_divj_path, "w")

    lines_results = [
        "iter,nfe,best_fitness,mean_fitness,std_fitness,"
        "time,XPL,XPT,DIV,GAP,RDP,ENT,"
        "Divj_mean,Divj_min,Divj_max\n"
    ]

    divj_header = ",".join([f"Divj_{j + 1}" for j in range(domain.dim)])
    lines_divj = [f"iter,{divj_header}\n"]

    try:
        # ==================== ITERACIÓN 0 (Métricas iniciales) ====================
        maxDiversity, XPL, XPT = initialize_diversity(population)
        lb_list = domain.lb.tolist()
        ub_list = domain.ub.tolist()

        meanFitness0 = float(np.mean(fitness))
        stdFitness0 = float(np.std(fitness))
        gap0, rdp0 = compute_gap_rdp(bestFitness, optimum)
        divj_vec0, divj_mean0, divj_min0, divj_max0 = diversity_per_dimension(
            population
        )
        ent_avg0, _ = population_entropy(population, bins=20, lb=lb_list, ub=ub_list)
        time0 = initializationTime2 - initializationTime1

        lines_results.append(
            f"0,{domain.nfe},{bestFitness:.6e},{meanFitness0:.6f},{stdFitness0:.6f},"
            f"{time0:.3f},{XPL:.6f},{XPT:.6f},{maxDiversity:.6f},"
            f"{gap0:.6f},{rdp0:.6f},{ent_avg0:.6f},"
            f"{divj_mean0:.6f},{divj_min0:.6f},{divj_max0:.6f}\n"
        )
        lines_divj.append("0," + ",".join([f"{v:.6f}" for v in divj_vec0]) + "\n")

        # Logging de consola
        label = domain.get_console_label(mh_name)
        print_initial(label, bestFitness)

        # ==================== SETUP ESPECIAL POR MH ====================
        # PO: Crear instancia de clase
        iterarPO = None
        if mh_name == "PO":
            iterarPO = IterarPO(
                domain.fo,
                domain.dim,
                domain.pop_size,
                effective_max_iter,
                domain.lb[0],
                domain.ub[0],
            )

        # userData para MH que lo requieren (polimórfico: dominio parsea si necesita)
        userData = domain.prepare_extra_params(mh_name, extra_params)
        if mh_name == "GOAT":
            userData.setdefault("jump_prob", 0.3)
            userData.setdefault("filter_ratio", 0.5)

        # ==================== BUCLE PRINCIPAL ====================
        while not termination.is_met():
            termination.increment_iter()
            iter_num = termination.current_iter
            timerStart = time.time()

            # --- 1. Iteración de la MH: mover la población ---
            population, fitness, mh_state = adapter.run_iteration(
                population,
                fitness,
                mh_state,
                iter_num,
                effective_max_iter,
                best,
                fo=domain.fo,
                userData=userData if userData else None,
            )

            # --- 2. PO: manejo especial (clase propia) ---
            if mh_name == "PO":
                iterarPO.pob(population, iter_num)
                population = iterarPO.optimizer(iter_num)
                if not isinstance(population, np.ndarray):
                    population = np.array(population)

            # --- 3. Procesamiento del dominio (clip/binarizar/reparar/evaluar) ---
            population, fitness = domain.process_new_population(
                population, fitness, mh_name, mh_state
            )

            # --- 4. Actualizar pBest (PSO, TJO) ---
            mh_state = adapter.update_pbest(population, fitness, mh_state)

            # --- 5. Actualizar mejor solución global ---
            best, bestFitness = domain.update_best(
                population, fitness, best, bestFitness
            )

            # --- 6. Actualizar estado del dominio para la siguiente iteración ---
            domain.set_iteration_state(best, population.copy())

            # --- 7. Diversidad ---
            div_t, maxDiversity, XPL, XPT = calculate_diversity(
                population, maxDiversity
            )

            # --- 8. Métricas ---
            meanFitness = float(np.mean(fitness))
            stdFitness = float(np.std(fitness))
            gap, rdp = compute_gap_rdp(bestFitness, optimum)
            divj_vec, divj_mean, divj_min, divj_max = diversity_per_dimension(
                population
            )
            ent_avg, _ = population_entropy(population, bins=20, lb=lb_list, ub=ub_list)
            timerFinal = time.time()
            dt = timerFinal - timerStart

            # --- 9. Buffer CSV ---
            lines_results.append(
                f"{iter_num},{domain.nfe},{bestFitness:.6e},"
                f"{meanFitness:.6f},{stdFitness:.6f},"
                f"{dt:.3f},{XPL:.6f},{XPT:.6f},{div_t:.6f},"
                f"{gap:.6f},{rdp:.6f},{ent_avg:.6f},"
                f"{divj_mean:.6f},{divj_min:.6f},{divj_max:.6f}\n"
            )
            lines_divj.append(
                f"{iter_num}," + ",".join([f"{v:.6f}" for v in divj_vec]) + "\n"
            )

            # --- 10. Consola ---
            is_fe_mode = termination.max_iter is None and termination.max_fe is not None
            print_iteration(
                iter_num,
                effective_max_iter,
                bestFitness,
                optimum,
                dt,
                XPT,
                XPL,
                div_t,
                progress_mode="fe" if is_fe_mode else "iter",
                fe=domain.nfe,
                max_fe=termination.max_fe,
            )

            # --- 11. Sincronizar NFE con el criterio de parada ---
            termination.current_fe = domain.nfe

        # ==================== FINALIZACIÓN ====================
        finalTime = time.time()
        print_final(bestFitness, initialTime, finalTime)

    finally:
        # Flush de buffers a disco (una sola operación de escritura)
        results.write("".join(lines_results))
        results.flush()
        results.close()
        results_divj.write("".join(lines_divj))
        results_divj.flush()
        results_divj.close()

    # ==================== PERSISTENCIA EN BD ====================
    binary = convert_into_binary(results_path)
    bd.insertarIteraciones(file_base, binary, id)
    bd.insertarResultados(bestFitness, finalTime - initialTime, best, id)
    bd.actualizarExperimento(id, "terminado")

    # Limpieza de archivos temporales
    os.remove(results_path)
    os.remove(results_divj_path)
