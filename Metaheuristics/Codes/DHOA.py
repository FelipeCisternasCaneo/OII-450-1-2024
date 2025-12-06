# doa_algoritmo.py
# Metaheurística Dhole Optimization Algorithm (DOA)
#
# VERSIÓN CORREGIDA 3:
# Se elimina 'userData' y se corrige la llamada a 'fo'.
# Se asume que fo(vector) retorna una tupla, y el fitness es fo(vector)[1],
# basándose en el traceback y el ejemplo de SSO.

import numpy as np
import random

def iterarDHOA(
    maxIter,
    iter,
    dim,
    population,
    fitness,
    best,
    fo,  # <--- Esta es la función wrapper que devuelve una tupla
    lb,
    ub
):
    """
    Genera una población candidata usando una iteración del Dhole
    Optimization Algorithm (DOA) para un solver genérico.

    Parámetros (definidos por el solver):
    - maxIter (int): Maximo de iteraciones.
    - iter (int): Iteracion actual.
    - dim (int): Dimension del problema (número de variables).
    - population (np.ndarray): Poblacion actual (shape N x D).
    - fitness (np.ndarray): Fitness de la poblacion actual (shape N,).
    - best (np.ndarray): Mejor solucion global hasta ahora (prey_global).
    - fo (callable): Funcion objetivo (el wrapper 'f' del solver).
    - lb (float): Limite inferior escalar.
    - ub (float): Limite superior escalar.

    Retorna:
    - list: Una nueva población de soluciones candidatas (lista de arrays).
    """

    # --- 1. Inicialización de la Iteración (Parámetros por Iteración) ---

    num_dholes = population.shape[0] # N dholes
    poblacion_candidata = [] # Almacenará las nuevas posiciones

    mejor_indice_local = np.argmin(fitness)
    prey_local = population[mejor_indice_local]

    pmn = round(random.random() * 15 + 5)
    presa_objetivo = (prey_local + best) / 2.0
    c2 = 1 - (iter / maxIter)
    denominador_ps = 1 + np.exp(-0.5 * (pmn - 25))
    ps = (1.0 / denominador_ps) * 1.0

    # --- LLAMADA CORREGIDA A 'fo' ---
    # Basado en el ejemplo de SSO, fo(vector) devuelve una tupla
    # y el valor de fitness (float) está en el índice [1].
    try:
        fitness_presa_objetivo = fo(presa_objetivo)[1]
    except (IndexError, TypeError) as e:
        # Captura por si fo() no devuelve una tupla o es None
        print(f"Error en DOA al llamar a fo(presa_objetivo): {e}")
        print(f"Valor devuelto por fo: {fo(presa_objetivo)}")
        # Asumimos el peor caso para S
        fitness_presa_objetivo = float('inf')

    epsilon = 1e-20 # Para evitar divisiones por cero

    # --- 2. Bucle principal sobre cada Dhole (i) ---
    for i in range(num_dholes):
        
        posicion_actual = population[i]
        fitness_actual = fitness[i]
        
        nueva_posicion_i = np.zeros(dim)

        # --- Decisiones a Nivel de Dhole (i) ---
        vocalizacion = random.random()
        z = i
        while z == i:
            z = random.randint(0, num_dholes - 1)
        posicion_z = population[z]
        
        rand_s = random.random()
        if fitness_presa_objetivo < epsilon:
            tamano_S = 0.0 if fitness_actual < epsilon else float('inf')
        else:
            # Aquí 'fitness_actual' es un float (viene del solver)
            # y 'fitness_presa_objetivo' ahora también es un float.
            tamano_S = 3 * rand_s * (fitness_actual / fitness_presa_objetivo)

        presa_debilitada = None
        if tamano_S > 2:
            if tamano_S == float('inf'):
                factor_debilitamiento = 1.0
            else:
                factor_debilitamiento = np.exp(-1.0 / tamano_S)
            presa_debilitada = factor_debilitamiento * prey_local

        
        # --- Bucle por Dimensión (j) ---
        for j in range(dim):
            
            if vocalizacion < 0.5:
                # --- EXPLORACIÓN ---
                if pmn < 10:
                    # Ecuación (6)
                    rand_busqueda = random.random()
                    movimiento_j = c2 * rand_busqueda * (presa_objetivo[j] - posicion_actual[j])
                    nueva_posicion_i[j] = posicion_actual[j] + movimiento_j
                else:
                    # Ecuación (8)
                    nueva_posicion_i[j] = posicion_actual[j] - posicion_z[j] + presa_objetivo[j]
            else:
                # --- EXPLOTACIÓN ---
                if tamano_S > 2:
                    # Ecuación (12)
                    rand_ataque_grande = random.random()
                    cos_term = np.cos(2 * np.pi * rand_ataque_grande)
                    sin_term = np.sin(2 * np.pi * rand_ataque_grande)
                    
                    W_prey_j = presa_debilitada[j]
                    movimiento_j = W_prey_j * ps * (cos_term - sin_term * W_prey_j * ps)
                    nueva_posicion_i[j] = posicion_actual[j] + movimiento_j
                else:
                    # Ecuación (13)
                    rand_ataque_peq = random.random()
                    termino1_j = (posicion_actual[j] - best[j]) * ps
                    termino2_j = ps * rand_ataque_peq * posicion_actual[j]
                    nueva_posicion_i[j] = termino1_j + termino2_j
                    
        # --- Fin del bucle j ---
        
        nueva_posicion_i = np.clip(nueva_posicion_i, lb, ub)
        poblacion_candidata.append(nueva_posicion_i)

    # --- Fin del bule i ---
    
    return poblacion_candidata