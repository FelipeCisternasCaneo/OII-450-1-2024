import numpy as np

def iterarDRA(
    maxIter, iter, dim, population, fitness, best, fo=None,
    lb=None, ub=None, lb0=None, ub0=None, vel=None, pBest=None,
    objective_type='MIN'
):
   
    N = population.shape[0]
    new_population = np.copy(population)
    new_fitness = np.copy(fitness)

    n_communities = 3  # número de comunidades religiosas
    group_size = N // n_communities

    BPSP = 0.3
    MP = (1 - np.random.rand()) * (1 - iter / (maxIter * 2)) * np.random.rand()
    RP = 0.05

    def is_better(f1, f2):
        return (f1 < f2) if objective_type == 'MIN' else (f1 > f2)

    for cid in range(n_communities):
        start = cid * group_size
        end = (cid + 1) * group_size if cid < n_communities - 1 else N

        community = new_population[start:end]
        community_fitness = new_fitness[start:end]

        best_idx = np.argmin(community_fitness) if objective_type == 'MIN' else np.argmax(community_fitness)
        worst_idx = np.argmax(community_fitness) if objective_type == 'MIN' else np.argmin(community_fitness)
        missionary = np.copy(community[best_idx])
        missionary_fitness = community_fitness[best_idx]

        for i in range(community.shape[0]):
            if i == best_idx:
                continue  # El misionero no cambia

            follower = np.copy(community[i])
            r = np.random.rand()

            # Selección
            if r <= BPSP:
                d = int(np.floor(np.random.uniform(0, dim)))
                follower[d] = missionary[d] * (np.random.rand() * np.cos(np.random.rand()))

            # Proselitismo
            elif r > MP:
                mean_belief = np.mean(missionary)
                randn = np.random.randn()
                if r > (1 - MP):
                    follower = (follower * 0.01) + (
                        mean_belief * (1 - MP)
                        + (1 - mean_belief)
                        - (np.random.rand() - 4 * np.sin(np.pi * np.random.rand()))
                    )
                else:
                    follower = (
                        (np.random.rand() - 2 * np.random.rand() - mean_belief)
                        * (2 * np.random.rand() - (1 - MP) * randn)
                    )
            # Milagro
            else:
                if np.random.rand() <= 0.5:
                    follower = follower * np.cos(np.pi / 2) * (np.random.rand() - np.cos(np.random.rand()))
                else:
                    follower = follower + np.random.rand() * (follower - np.round((1 / np.random.rand()) * follower))

            # Recompensa o Castigo
            if np.random.rand() < RP:
                randn = np.random.randn()
                if np.random.rand() < 0.5:
                    follower = follower - randn  # recompensa
                else:
                    follower = follower * (1 + randn)  # castigo

            follower = np.clip(follower, lb, ub)
            follower, fit_follower = fo(follower)

            # Si el nuevo es mejor que el peor, lo reemplaza
            if is_better(fit_follower, community_fitness[worst_idx]):
                community[worst_idx] = follower
                community_fitness[worst_idx] = fit_follower
                # Si además es mejor que el misionero, lo reemplaza
                if is_better(fit_follower, missionary_fitness):
                    missionary = follower
                    missionary_fitness = fit_follower
                    best_idx = worst_idx  # el nuevo misionero es quien reemplazó al peor

        # Guardar comunidad actualizada
        new_population[start:end] = community
        new_fitness[start:end] = community_fitness

    return new_population, new_fitness
