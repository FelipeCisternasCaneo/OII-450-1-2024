import numpy as np

def iterarTJO(maxIter, iter, dim, population, fitness, best, pBest, lb, ub, fo):
    """
    Traffic Jam Optimization (TJO) - 100% fiel al MATLAB original
    
    Args:
        maxIter: Máximo de iteraciones (T)
        iter: Iteración actual (t)
        dim: Dimensión del problema (nvars)
        population: Población actual (x)
        fitness: Fitness actual de la población (f)
        best: Mejor solución global (bestx)
        pBest: Memoria histórica de cada conductor (FlockMemoryX)
        lb: Límite inferior
        ub: Límite superior
        fo: Función objetivo (para evaluar)
    
    Returns:
        tuple: (población actualizada, fitness actualizado, pBest actualizado)
    """
    lb = np.array(lb)
    ub = np.array(ub)
    N = population.shape[0]
    
    # ========== PARÁMETROS ORIGINALES DEL PAPER ==========
    # MATLAB: a = linspace(0.5, 1.5, T)
    #         c = linspace(1.5, 0.5, T)
    a_start, a_end = 0.5, 1.5
    c_start, c_end = 1.5, 0.5
    
    # Calcular a(t) y c(t) para la iteración actual
    a_t = a_start + (iter - 1) / (maxIter - 1) * (a_end - a_start)
    c_t = c_start + (iter - 1) / (maxIter - 1) * (c_end - c_start)
    # =====================================================
    
    # r = t/T (MATLAB empieza en t=1)
    r = iter / maxIter
    
    best = np.array(best)
    
    # ========== 1. Calculate the best position for each driver ==========
    # BestX = (1-r).*FlockMemoryX + r.*bestx;
    BestX = (1 - r) * pBest + r * best
    
    # ========== 2. Drivers driving randomly causing traffic jam ==========
    # y = (1-r)*exp(-r)*sin(2*pi*rand(N,1)).*cos(2*pi*rand(N,1)).*c(t);
    # x = BestX + y.*((ub-lb).*rand(N,nvars)+lb);
    rand_sin = np.sin(2 * np.pi * np.random.rand(N, 1))
    rand_cos = np.cos(2 * np.pi * np.random.rand(N, 1))
    y = (1 - r) * np.exp(-r) * rand_sin * rand_cos * c_t
    
    rand_pos = (ub - lb) * np.random.rand(N, dim) + lb
    x = BestX + y * rand_pos
    
    # ========== 3. Drivers self-adjustment ==========
    # for i = 1:N
    #     if rand>0.5
    #         x(i,:) = x(i,:) + c(t)*sin(pi*rand).*(x(randi(N),:) - x(i,:));
    #     else
    #         x(i,:) = x(i,:) + c(t)*sin(pi*rand).*(BestX(randi(N),:) - x(i,:));
    #     end
    # end
    for i in range(N):
        rand_idx = np.random.randint(0, N)
        adjust_factor = c_t * np.sin(np.pi * np.random.rand())
        
        if np.random.rand() > 0.5:
            # Ajuste basado en otro conductor aleatorio
            x[i, :] = x[i, :] + adjust_factor * (x[rand_idx, :] - x[i, :])
        else:
            # Ajuste basado en BestX de otro conductor
            x[i, :] = x[i, :] + adjust_factor * (BestX[rand_idx, :] - x[i, :])
    
    # ========== 4. Traffic police directing drivers to drive ==========
    # x = BestX + a(t)*sin(2*pi*rand(N,1)).*(BestX - x);
    police_factor = a_t * np.sin(2 * np.pi * np.random.rand(N, 1))
    x = BestX + police_factor * (BestX - x)
    
    # ========== 5. Cross-border processing ==========
    # lbExtended = repmat(lb,[N,1]);
    # ubExtended = repmat(ub,[N,1]);
    # lbViolated = x < lbExtended;
    # ubViolated = x > ubExtended;
    # x(lbViolated) = lbExtended(lbViolated);
    # x(ubViolated) = ubExtended(ubViolated);
    x = np.clip(x, lb, ub)
    
    # ========== 6. Calculate fitness ==========
    # for i = 1:N
    #     f(i,:) = fun(x(i,:));
    # end
    new_fitness = np.zeros(N)
    for i in range(N):
        _, new_fitness[i] = fo(x[i, :])
    
    # ========== 7. Update memory (GREEDY SELECTION) ==========
    # UpdateMask = f < FlockMemoryF;
    # FlockMemoryF(UpdateMask) = f(UpdateMask);
    # FlockMemoryX(UpdateMask,:) = x(UpdateMask,:);
    update_mask = new_fitness < fitness
    
    # Actualizar solo las soluciones que mejoraron
    fitness[update_mask] = new_fitness[update_mask]
    pBest[update_mask, :] = x[update_mask, :]
    
    # Si no mejoró, mantener la población anterior (implícito, x ya tiene los nuevos)
    # Pero necesitamos retornar la población ACEPTADA (mezcla de nueva y anterior)
    population_updated = np.copy(pBest)  # Retornar la memoria (mejores encontradas)
    
    return population_updated, fitness, pBest