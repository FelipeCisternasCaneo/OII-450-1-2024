import random
import numpy as np

# Horned Lizard Optimization Algorithm (HLOA) - Legacy
# https://doi.org/10.1007/s10462-023-10653-7

def iterarHLOAScp(dim, population, best, lb0, ub0):
    """
    HLOA Binario para resolver problemas combinatoriales.
    """
    # Asegurarse de que lb y ub sean listas o arrays del tamaÃ±o de las dimensiones
    if not isinstance(lb0, (list, np.ndarray)):
        lb0 = [lb0] * dim
        
    if not isinstance(ub0, (list, np.ndarray)):
        ub0 = [ub0] * dim

    # Convertir la poblaciÃ³n a numpy array si no lo es
    population = np.array(population, dtype=float)

    # ParÃ¡metros de HLOA
    alpha = 0.1  # IntensificaciÃ³n (oscurecer la piel)
    beta = 0.3   # DiversificaciÃ³n (aclarar la piel)
    epsilon = 1e-10  # Evitar divisiones por cero

    # TamaÃ±o de la poblaciÃ³n
    N = len(population)

    # FunciÃ³n sigmoide para binarizaciÃ³n
    def sigmoide(x):
        return 1 / (1 + np.exp(-x))

    for i in range(N):
        for j in range(dim):
            # Estrategia de Camuflaje (Crypsis)
            r1 = random.random()
            r2 = random.random()
            local_search = (1 - r1) * best[j] + r2 * (population[i][j] - best[j])
            
            # Aplicar sigmoide y binarizar
            if sigmoide(local_search) >= 0.5:
                population[i][j] = 1
                
            else:
                population[i][j] = 0

            # Estrategia de Oscurecimiento o Aclaramiento de la piel
            if random.random() < 0.5:
                # Aclarar la piel (exploraciÃ³n global)
                exploration = best[j] + beta * (population[i][j] - best[j])
                
            else:
                # Oscurecer la piel (exploraciÃ³n local)
                exploration = best[j] - alpha * (population[i][j] - lb0[j])
                
            # Aplicar sigmoide y binarizar
            if sigmoide(exploration) >= 0.5:
                population[i][j] = 1
            
            else:
                population[i][j] = 0

            # Estrategia de ExpulsiÃ³n de sangre (Blood-Squirting)
            if random.random() < 0.1:
                projectile_motion = best[j] + alpha * np.sin(random.random() * np.pi)
                # Aplicar sigmoide y binarizar
                if sigmoide(projectile_motion) >= 0.5:
                    population[i][j] = 1
                    
                else:
                    population[i][j] = 0

            # Estrategia de Movimiento para escapar (Move-to-escape)
            walk = random.uniform(-1, 1)
            escape_motion = best[j] + walk * (population[i][j] - best[j])
            # Aplicar sigmoide y binarizar
            
            if sigmoide(escape_motion) >= 0.5:
                population[i][j] = 1
                
            else:
                population[i][j] = 0

    return population

def iterarHLOABen(dim, population, best, lb, ub):
    # Asegurarse de que population sea un numpy array
    population = np.array(population, dtype=float)

    # ParÃ¡metros de HLOA
    alpha = 0.1  # Controla la intensificaciÃ³n
    beta = 0.3   # Controla la diversificaciÃ³n

    # TamaÃ±o de la poblaciÃ³n
    N = len(population)
    
    for i in range(N):
        for j in range(dim):
            # Estrategia de Camuflaje (Crypsis)
            r1 = random.random()
            r2 = random.random()
            local_search = (1 - r1) * best[j] + r2 * (population[i][j] - best[j])
            population[i][j] = np.clip(local_search, lb[j], ub[j])

            # Estrategia de Oscurecimiento o Aclaramiento de la piel
            if random.random() < 0.5:
                # Aclarar la piel (exploraciÃ³n global)
                population[i][j] = best[j] + beta * (population[i][j] - best[j])
                
            else:
                # Oscurecer la piel (exploraciÃ³n local)
                population[i][j] = best[j] - alpha * (population[i][j] - lb[j])
            
            # ExpulsiÃ³n de sangre (Blood-Squirting)
            if random.random() < 0.1:
                projectile_motion = best[j] + alpha * np.sin(random.random() * np.pi)
                population[i][j] = np.clip(projectile_motion, lb[j], ub[j])
            
            # Movimiento para escapar (Move-to-escape)
            walk = random.uniform(-1, 1)
            escape_motion = best[j] + walk * (population[i][j] - best[j])
            population[i][j] = np.clip(escape_motion, lb[j], ub[j])

    # Devolver la poblaciÃ³n como numpy array (el solver espera un array, no una lista)
    return population

