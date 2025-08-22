import numpy as np

def iterarGOOSE(maxIter, iter, dim, population, best):
    """
    Implementación mejorada del algoritmo GOOSE con mejor balance exploración/explotación.
    
    Esta versión resuelve problemas de estancamiento temprano mediante:
    - Transición suave entre exploración y explotación
    - Múltiples estrategias de movimiento
    - Mecanismos de escape adaptativos
    - Parámetros auto-ajustables
    """
    N = len(population)
    bestX = np.asarray(best)
    population = np.array(population)
    
    # Progreso del algoritmo (0 a 1)
    progress = iter / maxIter
    
    # Función de transición sigmoidal para balance suave
    # Más exploración al inicio, transición gradual a explotación
    transition_point = 0.35  # Transición ligeramente más temprana
    steepness = 7.0  # Transición más definida
    exploration_ratio = 1.0 / (1.0 + np.exp(steepness * (progress - transition_point)))
    
    # Parámetros adaptativos mejorados
    alpha_max = 4.0  # Exploración máxima
    alpha_min = 0.25  # Exploración mínima
    alpha = alpha_max * exploration_ratio + alpha_min
    
    # Intensidad de movimiento adaptativa (curva exponencial)
    intensity = 3.0 * np.exp(-4 * progress) + 0.15  # Más intensidad inicial
    
    # Probabilidad de escape del estancamiento
    escape_prob = 0.12 * (1 - progress) + 0.015
    
    # Detectar diversidad de la población para adaptación
    pop_center = np.mean(population, axis=0)
    distances = [np.linalg.norm(ind - pop_center) for ind in population]
    diversity = np.std(distances) / (np.mean(distances) + 1e-10)
    
    # Ajustar parámetros según diversidad (estandarizado)
    if diversity < 0.1:  # Población muy convergida
        diversity_boost = 2.0
        escape_prob *= 3.0  # Más escape si hay baja diversidad
    else:
        diversity_boost = 1.0
    
    for i in range(N):
        # Parámetros originales del paper GOOSE
        rnd = np.random.rand()
        pro = np.random.rand()
        coe = max(np.random.rand(), 0.17)
        weight = np.random.randint(5, 26)
        tobj = np.random.randint(1, dim + 1)
        tsnd = np.random.randint(1, dim + 1)
        tot = (tobj + tsnd) / dim
        tavg = tot / 2
        
        # Decidir estrategia: exploración vs explotación
        if rnd < exploration_ratio:
            # EXPLORACIÓN MEJORADA
            
            # Estrategia múltiple basada en el índice del individuo
            strategy = i % 3
            
            if strategy == 0:
                # Estrategia 1: Hacia el mejor con perturbación grande
                direction = bestX - population[i]
                perturbation = np.random.normal(0, intensity * diversity_boost, dim)
                step_size = alpha * np.random.uniform(0.5, 2.0)
                X_new = population[i] + step_size * direction + perturbation
                
            elif strategy == 1:
                # Estrategia 2: Combinación con individuo aleatorio
                if N > 1:
                    random_idx = np.random.choice([j for j in range(N) if j != i])
                    random_ind = population[random_idx]
                    
                    # Combinación ponderada
                    w1, w2 = np.random.random(2)
                    w1, w2 = w1/(w1+w2), w2/(w1+w2)  # Normalizar
                    
                    target = w1 * bestX + w2 * random_ind
                    direction = target - population[i]
                    
                    step_size = alpha * np.random.uniform(0.3, 1.5)
                    perturbation = np.random.normal(0, intensity * 0.5, dim)
                    X_new = population[i] + step_size * direction + perturbation
                else:
                    # Fallback si solo hay un individuo
                    direction = bestX - population[i]
                    perturbation = np.random.normal(0, intensity, dim)
                    X_new = population[i] + alpha * direction + perturbation
                    
            else:
                # Estrategia 3: Lévy flight para exploración global
                # Implementación simplificada de Lévy flight
                levy_step = np.random.normal(0, 1, dim) * intensity
                levy_step *= np.random.uniform(0.1, 2.0) ** (-1.5)  # Distribución de cola pesada
                
                direction = bestX - population[i]
                X_new = population[i] + 0.5 * alpha * direction + levy_step * diversity_boost
                
        else:
            # EXPLOTACIÓN MEJORADA (ecuaciones originales del paper)
            
            if pro > 0.2 and weight >= 12:
                # Ecuación (12) - Vuelo de formación con alta velocidad
                v_ff = tobj * np.sqrt(weight / 9.81)
                d_s = 343.2 * tsnd
                d_g = 0.5 * d_s
                val = v_ff + d_g * (tavg ** 2)
            else:
                # Ecuación (13) - Vuelo individual con baja velocidad
                v_ff = tobj * (weight / 9.81)
                d_s = 343.2 * tsnd
                d_g = 0.5 * d_s
                val = v_ff * d_g * (tavg ** 2) * coe
            
            # Validar val
            if not np.isfinite(val) or abs(val) > 1e6:
                val = np.clip(val, -1e6, 1e6)
                if not np.isfinite(val):
                    val = 1.0
            
            # Ecuación (14) del paper con scaling adaptativo mejorado
            rand_vector = np.random.uniform(-1, 1, dim)
            direction = bestX - population[i]
            
            # Scaling más efectivo para mejor convergencia
            if abs(val) > 1000:
                scaling_factor = 0.008 * intensity
            elif abs(val) > 100:
                scaling_factor = 0.08 * intensity
            else:
                scaling_factor = 0.3 * intensity
            
            X_new = bestX + (val * scaling_factor) * rand_vector * direction
        
        # Mecanismo de escape del estancamiento (estandarizado)
        if np.random.random() < escape_prob:
            # Estrategia dual de escape
            if np.random.random() < 0.3:
                # Escape tipo 1: Salto aleatorio dirigido
                jump_direction = np.random.choice([-1, 1], dim)
                jump_magnitude = 0.5 * intensity * diversity_boost
                jump = jump_direction * jump_magnitude * np.random.uniform(0.5, 2.0, dim)
                X_new = X_new + jump
            else:
                # Escape tipo 2: Reinicialización parcial inteligente
                mask = np.random.random(dim) < 0.3  # 30% de las dimensiones
                if np.any(mask):
                    # Reinicializar en el rango actual de la población
                    pop_min = np.min(population, axis=0)
                    pop_max = np.max(population, axis=0)
                    # Evitar división por cero o rangos muy pequeños
                    range_size = pop_max - pop_min
                    valid_mask = range_size > 1e-6
                    if np.any(valid_mask & mask):
                        X_new[valid_mask & mask] = np.random.uniform(
                            pop_min[valid_mask & mask], 
                            pop_max[valid_mask & mask]
                        )
                    else:
                        # Fallback: usar rango estándar
                        X_new[mask] = np.random.uniform(-50, 50, np.sum(mask))
        
        # Validación final robusta (estandarizada)
        # Límites adaptativos para evitar valores extremos
        if progress < 0.5:
            safe_limit = 500.0  # Más permisivo al inicio
        else:
            safe_limit = 200.0  # Más restrictivo al final
            
        X_new = np.clip(X_new, -safe_limit, safe_limit)
        X_new = np.nan_to_num(X_new, nan=0.0, posinf=safe_limit, neginf=-safe_limit)
        
        # Actualizar individuo
        population[i] = X_new
    
    return population
