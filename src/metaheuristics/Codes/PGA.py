import numpy as np
import math

def calcular_r_i (dim):
    return np.random.uniform(-1, 1, dim)

def calcular_beta ():
    # Factor de dirección β, selecciona aleatoriamente -1 o 1
    vector_direccion = [-1, 1]
    return np.random.choice(vector_direccion)

def iterarPGA(maxIter, iter, dim, population, fitness, best, 
              fo=None, objective_type='MIN'):
    """
    Implementación del Phototropic Growth Algorithm (PGA).
    
    Esta implementacion esta realizada en base al paper que presenta el PGA con:
    - División correcta X_L (primeras N_L células) y X_S (últimas N_S células)
    - Operadores de mitosis para ambas regiones
    - Cálculo preciso de curvatura según Eq. (17)
    - Cálculo correcto del Factor de Curvatura (FOC) según Eq. (18)
    - Cálculo correcto de la vecindad celular según Eq. (19)
    - Elongación celular según Eq. (21)
    
    Args:
        maxIter (int): Máximo de iteraciones (T)
        iter (int): Iteración actual (t)
        dim (int): Dimensión del problema (D)
        population (np.ndarray): Población actual (X)
        fitness (np.ndarray): Fitness de la población
        best (np.ndarray): Mejor solución global (X_Best)
        fo (callable): Función objetivo
        objective_type (str): 'MIN' o 'MAX'
    
    Returns:
        np.ndarray: Nueva población actualizada
    """
    N = population.shape[0]

    
    # Calcular factor de crecimiento α (Ecuación 5)
    alpha = math.exp(-iter / maxIter)
    
    # DIVISIÓN POBLACIONAL (Ecuaciones 1-4)
    # X_L = primeras N_L células, X_S = últimas N_S células (NO ordenadas por fitness)
    C1, C2 = 0.4, 0.6
    r1 = np.random.uniform(0, 1)
    N_L = int((C1 + (C2 - C1) * r1) * N)  # Ecuación 1
    N_S = N - N_L  # Ecuación 3
    
    # Según paper: X_L = primeras N_L células, X_S = últimas N_S células
    X_L = population[:N_L].copy()  # Células en luz (primeras N_L)
    X_S = population[N_L:].copy()  # Células en sombra (últimas N_S)
    fitness_l = fitness[:N_L]
    fitness_s = fitness[N_L:]
    
    # Mejores locales según paper
    if objective_type == 'MIN':
        best_l_idx = np.argmin(fitness_l)
        best_s_idx = np.argmin(fitness_s)
        best_global_fitness = np.min(fitness)
    else:
        best_l_idx = np.argmax(fitness_l)
        best_s_idx = np.argmax(fitness_s)
        best_global_fitness = np.max(fitness)
    
    X_best_L = X_L[best_l_idx]  # X_Lbest del paper
    X_best_S = X_S[best_s_idx]  # X_Sbest del paper

    """
    ====================================
    # Células en region luminosa (X_L) #
    ====================================
    """ 
    X_L_new = [] 
    new_fitness_l = []
    
    for i in range(len(X_L)):

        """
        # Region luminosa (X_L)

        # Primera célula hija 'X_L_new_1' 

        # Operador de mutación (Ecuación 9)

        Parametros de la ecuacion 9:
        r2 -, r3: Valores aleatorios entre -1 y 1.
        X_rand: Célula aleatoria de la población completa.
        beta (β): variable de direccion, con posibles valores {-1, 1}.
        alpha (α): Factor de crecimiento - Vease ecuacion 3.2 en informe investigacion o ecuacion 5 en paper del PGA.
        X_best_L: Mejor célula en la region luminosa (X_Lbest).

        Variable con resultado de la ecuacion:
        X_L_new_1
        """
        r2 = calcular_r_i(dim)
        r3 = calcular_r_i(dim)

        rand_idx = np.random.randint(0, N)
        X_rand = population[rand_idx]

        beta = calcular_beta()

        # Ecuación 9 - Operador de mutación
        X_L_new_1 = (X_rand + 
                   alpha * beta * r2 * np.abs(X_rand - X_L[i]) + 
                   alpha * beta * r3 * np.abs(X_best_L - X_L[i]))
        
        """
        # Region luminosa (X_L)

        # Segunda célula hija 'X_L_new_2'

        # Operador de auxinas - (Ecuación 13)

        Parametros de la ecuacion 13:
        X_L[i]: Celula padre actual en la region luminosa.
        r4: Valor aleatorio entre -1 y 1.
        alpha (α): Factor de crecimiento - Vease ecuacion 3.2 en informe investigacion o ecuacion 5 en paper del PGA.
        X_best_L: Mejor célula en la region luminosa.

        Variable con resultado de la ecuacion:
        X_L_new_2
        """
        r4 = calcular_r_i(dim)

        # Ecuación 13 exacta - Operador de auxinas
        X_L_new_2 = X_L[i] + alpha * r4 * np.abs(X_best_L - X_L[i])


        # Agregar hijas
        X_L_new.extend([X_L_new_1, X_L_new_2])

        # Evaluar fitness (en producción se usaría fo)
        if fo is not None:
            _, fit1 = fo(X_L_new_1)
            _, fit2 = fo(X_L_new_2)
            new_fitness_l.extend([fit1, fit2])
        else:
            # Placeholder
            new_fitness_l.extend([fitness_l[i], fitness_l[i]])
    
    """
    =====================================
    # Células en region sombreada (X_S) #
    =====================================
    """ 
    X_S_new= []
    new_fitness_s = []

    for i in range(len(X_S)):
        """
        # Regíon sombreada (X_S)

        # Primera célula hija de la region sombreada 'X_S_new_1'

        # Operador de mutación - (Ecuación 15)

        Parametros de la ecuacion 15:
        r5: Valor aleatorio entre -1 y 1.
        beta (β): variable de direccion, con posibles valores {-1, 1}.
        alpha (α): Factor de crecimiento - Vease ecuacion 3.2 en informe investigacion o ecuacion 5 en paper del PGA.
        X_rand: Célula aleatoria de la población completa.
        X_S[i]: Célula padre actual en la region sombreada.

        Variable con resultado de la ecuacion:
        X_S_new_1
        """
        r5 = calcular_r_i(dim)
        beta = calcular_beta()

        rand_idx = np.random.randint(0, N)
        X_rand = population[rand_idx]

        # Ecuación 15 - Operador de mutación
        X_S_new_1 = X_S[i] + alpha * beta * r5 * np.abs(X_S[i] - X_rand)
        
        """
        # Regíon sombreada (X_S)

        # Segunda célula hija de la region sombreada 'X_S_new_2' 

        # Operador de redistribucion de auxinas - (Ecuación 16)

        Parametros de la ecuacion 16:
        r6: Valor aleatorio entre -1 y 1.
        beta (β): variable de direccion, con posibles valores {-1, 1}.
        alpha (α): Factor de crecimiento - Vease ecuacion 3.2 en informe investigacion o ecuacion 5 en paper del PGA.
        X_S[i]: Célula padre actual en la region sombreada.
        X_L_rand: Celula aleatoria de la región luminosa (X_L).
        X_best_L: Mejor célula en la region luminosa (X_Lbest).

        Variable con resultado de la ecuacion:
        X_S_new_2
        """
        r6 = calcular_r_i(dim)

        beta = calcular_beta()

        light_idx = np.random.randint(0, len(X_L))
        X_L_rand = X_L[light_idx]

        # Ecuación 16 - Operador de redistribución de auxinas
        X_S_new_2 = X_L_rand + alpha * beta * r6 * np.abs(X_best_L - X_S[i])

        # Agregar hijas
        X_S_new.extend([X_S_new_1, X_S_new_2])

        if fo is not None:
            _, fit1 = fo(X_S_new_1)
            _, fit2 = fo(X_S_new_2)
            new_fitness_s.extend([fit1, fit2])
        else:
            new_fitness_s.extend([fitness_s[i], fitness_s[i]])
    
    # Convertir a arrays
    X_L_new = np.array(X_L_new)
    X_S_new = np.array(X_S_new)
    new_fitness_l = np.array(new_fitness_l)
    new_fitness_s = np.array(new_fitness_s)
    
    # FUSIONAR POBLACIÓN TEMPORAL
    temp_population = np.vstack([X_L_new, X_S_new])
    temp_fitness = np.concatenate([new_fitness_l, new_fitness_s])
    
    """
    # Fase de elongación - (Ecuaciones 17-21)

    -   La fase de elongación es una proceso en el cual 
        se balancea la exploracion con la vecinidad celular
        y la explotacion en el calculo de la curvatura.
    

    # Calculo de curvatura (Ecuación 17):

    Parametros:
    beta (β): variable de direccion, con posibles valores {-1, 1}.
    alpha (α): Factor de crecimiento - Vease ecuacion 3.2 en informe investigacion o ecuacion 5 en paper del PGA.
    Meanfitness(XL) Aptitud promedio de las células en XL
    Bestfitness(X) Mejor aptitud en toda la población.

    Variable con resultado de la ecuacion:
    curvature.
    """
    beta = calcular_beta()
    mean_fitness_X_L = np.mean(fitness_l)
    
    curvature = beta * (alpha - mean_fitness_X_L / best_global_fitness)

    # Aplicar elongación a cada célula (Ecuación 21)
    final_population = np.zeros_like(population)
    
    for i in range(N):
        """
        # Factor de curvatura (FOC) (Ecuación 21):
        
        Paramentros:
        r7, r8: Valores aleatorios entre -1 y 1.
        beta (β): variable de direccion, con posibles valores {-1, 1}.
        curvature: Curvatura calculada previamente.
        temp_population[i]: Célula actual en la población temporal, para no trabajar con la población original.
        X_best: Mejor célula en toda la población (X_Best).
        
        Variable con resultado de la ecuacion:
        FOC
        """
        r7 = calcular_r_i(dim)
        beta = calcular_beta()
        X_best = best
    
        # Factor de Curvatura - FOC (Ecuación 18)
        FOC = r7 * curvature * (temp_population[i] - X_best)
        
        """
        # Vecindad Celular (Ecuación 19):

        Parametros:
        r8: Valor aleatorio entre -1 y 1.
        alpha (α): Factor de crecimiento - Vease ecuacion 3.2 en informe investigacion o ecuacion 5 en paper del PGA.
        beta (β): variable de direccion, con posibles valores {-1, 1}.
        temp_population[i]: Célula actual en la población temporal, para no trabajar con la
        N: Tamaño de la población.

        Variable con resultado de la ecuacion:
        cell_vicinity
        """
        r8 = calcular_r_i(dim)

        # Vecindad Celular (Ecuación 19)
        cell_vicinity = (alpha * beta * r8 * 
                        (temp_population[i] + temp_population[(i + 1) % N]) / 2)
        
        """
        # Elongación Celular (Ecuación 21):

        Parametros:
        temp_population[i]: Célula actual en la población temporal, para no trabajar con la población original.
        FOC: Factor de curvatura calculado previamente.
        cell_vicinity: Vecindad celular calculada previamente.

        Variable con resultado de la ecuacion:
        new_position
        """

        # Elongación Celular (Ecuación 21)
        new_position = temp_population[i] + FOC + cell_vicinity
            
        final_population[i] = new_position
    
    return final_population