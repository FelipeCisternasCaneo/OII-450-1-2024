import numpy as np

def iterarGOOSE_original(dim, population, best):
    """
    Implementación original del algoritmo GOOSE siguiendo exactamente las ecuaciones del paper.
    
    Paper: "GOOSE Algorithm: A Powerful Optimization Tool for Real-World Engineering Challenges"
    
    Esta implementación funciona para problemas continuos (benchmarks).
    Para SCP, la binarización es manejada por el framework de discretización.
    
    Ecuaciones implementadas:
    - Ecuación (12): Vuelo de formación con alta velocidad
    - Ecuación (13): Vuelo individual con baja velocidad  
    - Ecuación (14): Actualización de posición en explotación
    - Ecuación (15): Actualización de posición en exploración
    """
    N = len(population)
    bestX = np.asarray(best)
    
    # Convertir la población a un array de NumPy para facilitar su manipulación
    population = np.array(population)
    
    for i in range(N):
        # Parámetros aleatorios según el paper GOOSE (Sección 3.2)
        rnd = np.random.rand()  # Número aleatorio para decidir fase
        pro = np.random.rand()  # Probabilidad para tipo de vuelo
        coe = np.random.rand()  # Coeficiente aleatorio
        weight = np.random.randint(5, 26)  # Peso del ganso: 5-25 kg
        tobj = np.random.randint(1, dim + 1)  # Tiempo objetivo
        tsnd = np.random.randint(1, dim + 1)  # Tiempo de sonido
        tot = (tobj + tsnd) / dim
        tavg = tot / 2

        if rnd >= 0.5:
            # FASE DE EXPLOTACIÓN (Exploitation Phase)
            # Ecuaciones (12) y (13) del paper
            if pro > 0.2 and weight >= 12:
                # Ecuación (12) - Vuelo de formación con alta velocidad
                v_ff = tobj * np.sqrt(weight / 9.81)  # Velocidad de vuelo en formación
                d_s = 343.2 * tsnd  # Distancia de sonido
                d_g = 0.5 * d_s     # Distancia del ganso
                val = v_ff + d_g * (tavg ** 2)
            else:
                # Ecuación (13) - Vuelo individual con baja velocidad
                v_ff = tobj * (weight / 9.81)  # Velocidad reducida
                d_s = 343.2 * tsnd
                d_g = 0.5 * d_s
                val = v_ff * d_g * (tavg ** 2) * coe

            # Validar que val no sea problemático
            if not np.isfinite(val) or abs(val) > 1e6:
                val = np.clip(val, -1e6, 1e6)  # Limitar valores extremos
                if not np.isfinite(val):
                    val = 1.0  # Valor seguro por defecto

            # Ecuación (14) del paper: X_new = X_best + val * rand(-1,1) * (X_best - X_current)
            rand_vector = np.random.uniform(-1, 1, dim)
            direction = bestX - population[i]
            
            # Scaling conservador para evitar valores extremos
            scaling_factor = 0.01  # Factor muy conservador
            X_new = bestX + (val * scaling_factor) * rand_vector * direction
            
        else:
            # FASE DE EXPLORACIÓN (Exploration Phase)
            # Ecuación (15) del paper: X_new = X_best + alpha * rand(-1,1) * (X_rand - X_current)
            
            alpha = 2.0  # Parámetro de exploración fijo según paper
            mn = max(tobj, tsnd)  # Factor de intensidad según el paper
            
            if N > 1:
                # Seleccionar individuo aleatorio diferente al actual
                candidates = [j for j in range(N) if j != i]
                if candidates:
                    rand_idx = np.random.choice(candidates)
                    random_individual = population[rand_idx]
                    
                    # Ecuación (15) - Exploración basada en individuo aleatorio
                    rand_vector = np.random.uniform(-1, 1, dim)
                    direction = random_individual - population[i]
                    
                    # Aplicar la ecuación con factor de intensidad
                    X_new = bestX + alpha * rand_vector * direction * (mn / dim)
                else:
                    # Fallback: movimiento aleatorio desde el mejor
                    rand_vector = np.random.uniform(-1, 1, dim)
                    X_new = bestX + alpha * rand_vector * (mn / dim)
            else:
                # Caso con un solo individuo: movimiento aleatorio
                rand_vector = np.random.uniform(-1, 1, dim)
                X_new = bestX + alpha * rand_vector * (mn / dim)

        # Validación básica para evitar valores extremos
        # Usar límites conservadores
        X_new = np.clip(X_new, -100, 100)
        
        # Manejar NaN o infinitos
        X_new = np.nan_to_num(X_new, nan=0.0, posinf=100.0, neginf=-100.0)

        # Actualizar individuo en la población
        population[i] = X_new

    return population
