import numpy as np

def iterarSSO(maxIter, iter, dim, population, fitness, best, fo=None, lb=None, ub=None, **kwargs):
    """
    Algoritmo Salmon Salar Optimization (SSO) con memoria de líderes históricos.
    Compatible con el sistema bio-inspirado del curso.
    """

    # Obtener el estado persistente desde kwargs
    state = kwargs.get("state", {})

    # Inicializar la lista de líderes si no existe o si cambia la dimensión
    if (
        "LeaderSalmonList" not in state or
        len(state["LeaderSalmonList"]) == 0 or
        state.get("dim") != dim
    ):
        state["LeaderSalmonList"] = [population[np.argmin(fitness)]]
        state["dim"] = dim

    LeaderSalmonList = state["LeaderSalmonList"]

    def find_best(salmon_list):
        return min(salmon_list, key=lambda x: fo(x)[1])

    def update_leader_list(leader_list, candidate, max_leaders=3):
        leader_list.append(candidate)
        # Eliminar duplicados (opcional)
        unique = [list(x) for x in set(tuple(np.round(ind, decimals=8)) for ind in leader_list)]
        unique = [np.array(u) for u in unique]
        unique.sort(key=lambda x: fo(x)[1])
        return unique[:max_leaders]

    def generate_candidates(salmon, best_leader):
        alpha = (salmon + best_leader) / 2
        beta = np.abs(salmon - best_leader)
        candidates = []
        for _ in range(3):
            candidate = np.random.normal(loc=alpha, scale=beta)
            candidate = np.clip(candidate, lb, ub)
            candidates.append(candidate)
        return candidates

    new_population = []
    for i in range(population.shape[0]):
        salmon = population[i]
        best_leader = find_best(LeaderSalmonList)

        # Generar 3 candidatos
        candidates = generate_candidates(salmon, best_leader)
        all_candidates = [salmon] + candidates

        # Evaluar y seleccionar el mejor
        fitness_values = [fo(c)[1] for c in all_candidates]
        best_candidate = all_candidates[np.argmin(fitness_values)]

        new_population.append(best_candidate)

    # Actualizar lista de líderes con el mejor de esta generación
    new_population = np.array(new_population)
    fitness_new = [fo(ind)[1] for ind in new_population]
    best_idx = np.argmin(fitness_new)
    leader_candidate = new_population[best_idx]

    LeaderSalmonList = update_leader_list(LeaderSalmonList, leader_candidate)
    state["LeaderSalmonList"] = LeaderSalmonList

    return np.array(new_population)
