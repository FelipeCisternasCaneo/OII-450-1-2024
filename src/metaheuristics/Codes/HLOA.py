import random
import numpy as np

# Horned Lizard Optimization Algorithm (HLOA)
# https://doi.org/10.1007/s10462-023-10653-7
# Vectorizado: elimina loops dim-level y samplea de forma vectorial.

def iterarHLOAScp(dim, population, best, lb0, ub0):
    # Asegurar que lb0 y ub0 sean arrays de las dimensiones correctas
    if np.isscalar(lb0): lb0 = np.full(dim, lb0)
    if np.isscalar(ub0): ub0 = np.full(dim, ub0)
    
    population = np.array(population, dtype=float)
    lb0 = np.array(lb0, dtype=float)
    ub0 = np.array(ub0, dtype=float)
    best = np.array(best, dtype=float)

    alpha = 0.1
    beta = 0.3
    N = len(population)
    
    def sigmoide(x):
        # Clip para evitar overflow en exp
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
        
    # 1. Estrategia de Camuflaje (Crypsis)
    r1 = np.random.rand(N, dim)
    r2 = np.random.rand(N, dim)
    local_search = (1 - r1) * best + r2 * (population - best)
    population = np.where(sigmoide(local_search) >= 0.5, 1, 0)
    
    # 2. Estrategia de Oscurecimiento o Aclaramiento
    mask = np.random.rand(N, dim) < 0.5
    exploration = np.where(mask,
        best + beta * (population - best),
        best - alpha * (population - lb0)
    )
    population = np.where(sigmoide(exploration) >= 0.5, 1, 0)
    
    # 3. Estrategia de Expulsión de sangre
    mask2 = np.random.rand(N, dim) < 0.1
    r_val = np.random.rand(N, dim)
    projectile_motion = best + alpha * np.sin(r_val * np.pi)
    new_blood = np.where(sigmoide(projectile_motion) >= 0.5, 1, 0)
    population = np.where(mask2, new_blood, population)
    
    # 4. Movimiento para escapar
    walk = np.random.uniform(-1, 1, size=(N, dim))
    escape_motion = best + walk * (population - best)
    population = np.where(sigmoide(escape_motion) >= 0.5, 1, 0)
    
    return population

def iterarHLOABen(dim, population, best, lb, ub):
    population = np.array(population, dtype=float)
    N = len(population)
    lb = np.array(lb)
    ub = np.array(ub)

    alpha = 0.1
    beta = 0.3
    
    # 1. Estrategia de Camuflaje (Crypsis)
    r1 = np.random.rand(N, dim)
    r2 = np.random.rand(N, dim)
    local_search = (1 - r1) * best + r2 * (population - best)
    population = np.clip(local_search, lb, ub)
    
    # 2. Estrategia de Oscurecimiento o Aclaramiento
    mask = np.random.rand(N, dim) < 0.5
    exploration = np.where(mask, 
        best + beta * (population - best),
        best - alpha * (population - lb)
    )
    # En el original no hacian clip al salir del 2, pero sí asumo que es para población real, 
    # aunque en su código dice "population[i][j] = best[j] ..." directamente
    population = exploration
    
    # 3. Expulsión de sangre
    mask2 = np.random.rand(N, dim) < 0.1
    r_val = np.random.rand(N, dim)
    proj_motion = best + alpha * np.sin(r_val * np.pi)
    proj_motion = np.clip(proj_motion, lb, ub)
    population = np.where(mask2, proj_motion, population)
    
    # 4. Movimiento para escapar
    walk = np.random.uniform(-1, 1, size=(N, dim))
    escape_motion = best + walk * (population - best)
    population = np.clip(escape_motion, lb, ub)
    
    return population
