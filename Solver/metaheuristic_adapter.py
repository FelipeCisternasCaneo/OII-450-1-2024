"""
Metaheuristic Adapter (Envoltorio Universal)
=============================================
Envuelve cualquier metaheurística registrada en Metaheuristics/imports.py
bajo una interfaz unificada.

Esto elimina las cascadas masivas de if-elif en los archivos de population,
y expone un contrato simple:

    adapter = MetaheuristicAdapter("PSO", pop_size=25, dim=30, ...)
    new_pop, fitness, mh_state = adapter.run_iteration(
        population, fitness, mh_state, iter, maxIter, best, fo
    )

El `mh_state` es un diccionario que encapsula todas las variables internas
que cada MH necesita (vel, pBest, pBestScore, posibles_mejoras, etc.)
sin que el Solver tenga que saber cuáles son.
"""

import numpy as np

from Metaheuristics.imports import metaheuristics, MH_ARG_MAP, IterarPO


# ==================== CONSTANTES: Clasificación de MH ====================

# MH que requieren inicialización de pBest (memoria personal)
MH_NEEDS_PBEST = {'PSO', 'TJO'}

# MH que requiere inicialización de velocidad
MH_NEEDS_VEL = {'PSO'}

# MH que retorna (population, posibles_mejoras)
MH_RETURNS_MEJORAS = {'LOA'}

# MH que retorna (population, fitness, best) — evalúa internamente
MH_RETURNS_FIT_BEST = {'GOAT'}

# MH que retorna (population, fitness, pBest) — evalúa internamente
MH_RETURNS_FIT_PBEST = {'TJO'}

# MH que retorna solo un np.ndarray (sin vel)
MH_RETURNS_ARRAY_ONLY = {'APO', 'CLO'}

# MH con flujo completamente propio (PO usa clase, GA tiene params especiales)
MH_SPECIAL_FLOW = {'PO', 'GA'}


class MetaheuristicAdapter:
    """
    Adaptador universal que envuelve la lógica de cualquier metaheurística.
    
    El Solver Universal interactúa SÓLO con esta clase, sin necesidad de
    conocer los detalles de implementación de cada algoritmo.
    
    Args:
        mh_name:   Nombre de la metaheurística (e.g. 'PSO', 'GWO', 'TJO').
        pop_size:  Tamaño de la población.
        dim:       Dimensiones del problema.
        lb:        Límites inferiores (array).
        ub:        Límites superiores (array).
    """

    def __init__(self, mh_name, pop_size, dim, lb, ub):
        self.mh_name = mh_name
        self.pop_size = pop_size
        self.dim = dim
        self.lb = np.array(lb) if not isinstance(lb, np.ndarray) else lb
        self.ub = np.array(ub) if not isinstance(ub, np.ndarray) else ub

        # Resolver el nombre real de la MH (HLOA tiene variantes)
        self._resolved_name = mh_name

        # Validar que la MH exista (excepto PO que usa clase aparte)
        if mh_name != 'PO':
            if mh_name not in metaheuristics and mh_name not in ('HLOA',):
                raise ValueError(
                    f"Metaheurística '{mh_name}' no encontrada en "
                    f"Metaheuristics/imports.py"
                )

    def initialize_state(self, population=None):
        """
        Crea el diccionario de estado interno (mh_state) con las variables
        que esta MH en particular necesita.
        
        Args:
            population: Población inicial (necesaria para dimensionar pBest).
        
        Returns:
            dict: Estado interno de la MH.
        """
        state = {
            'vel': None,
            'pBest': None,
            'pBestScore': None,
            'posibles_mejoras': None,
        }

        if self.mh_name in MH_NEEDS_VEL:
            state['vel'] = np.zeros((self.pop_size, self.dim))

        if self.mh_name in MH_NEEDS_PBEST:
            state['pBestScore'] = np.full(self.pop_size, np.inf)
            if population is not None:
                state['pBest'] = population.copy()
            else:
                state['pBest'] = np.zeros((self.pop_size, self.dim))

        return state

    def resolve_mh_name(self, domain_type='BEN'):
        """
        Resuelve variantes de nombre (HLOA_BEN, HLOA_SCP).
        
        Args:
            domain_type: 'BEN' o 'SCP' para resolver variantes.
        """
        if self.mh_name == 'HLOA':
            self._resolved_name = f'HLOA_{domain_type}'
        else:
            self._resolved_name = self.mh_name

    def run_iteration(self, population, fitness, mh_state, iter_num,
                      max_iter, best, fo=None, userData=None):
        """
        Ejecuta UNA iteración de la metaheurística y retorna SIEMPRE
        el mismo contrato: (nueva_población, fitness, mh_state).
        
        Args:
            population:  Población actual (np.ndarray).
            fitness:     Vector de fitness actual (np.ndarray).
            mh_state:    Diccionario de estado interno de la MH.
            iter_num:    Número de iteración actual (1-indexed).
            max_iter:    Número máximo de iteraciones.
            best:        Mejor solución global actual.
            fo:          Función objetivo fo(x) -> (x, fitness).
            userData:    Diccionario de parámetros extra (GOAT, SRO, etc.)
        
        Returns:
            tuple: (nueva_población, fitness_actualizado, mh_state_actualizado)
        """
        mh = self._resolved_name

        # --- Caso especial: PO (usa clase propia) ---
        if self.mh_name == 'PO':
            return population, fitness, mh_state

        # --- Construir contexto de argumentos ---
        lb0 = self.lb[0] if len(self.lb) > 0 else None
        ub0 = self.ub[0] if len(self.ub) > 0 else None

        context = {
            'maxIter': max_iter,
            'iter': iter_num,
            'dim': self.dim,
            'population': population,
            'fitness': fitness,
            'best': best,
            'vel': mh_state.get('vel'),
            'pBest': mh_state.get('pBest'),
            'ub': self.ub,
            'lb': self.lb,
            'ub0': ub0,
            'lb0': lb0,
            'fo': fo,
            'objective_type': 'MIN',
            'userData': userData if userData is not None else {},
        }

        # CCMGO espera lb/ub como lista (hace isinstance(lb, list)),
        # no como np.ndarray. Convertir para compatibilidad.
        if mh == 'CCMGO':
            context['lb'] = self.lb.tolist()
            context['ub'] = self.ub.tolist()

        # Inyectar userData al contexto (GOAT usa jump_prob, filter_ratio, etc.)
        if userData:
            context.update(userData)

        # --- Obtener argumentos requeridos ---
        if mh not in MH_ARG_MAP:
            raise ValueError(
                f"MH_ARG_MAP no tiene entrada para '{mh}'. "
                f"Revisa Metaheuristics/imports.py."
            )

        required_args = MH_ARG_MAP[mh]
        kwargs = {}
        for arg_name in required_args:
            if arg_name not in context:
                raise KeyError(
                    f"Argumento '{arg_name}' requerido por {mh} "
                    f"no encontrado en el contexto del adapter."
                )
            kwargs[arg_name] = context[arg_name]

        # --- Ejecutar la MH ---
        mh_function = metaheuristics[mh]
        try:
            result = mh_function(**kwargs)
        except TypeError as e:
            raise TypeError(
                f"Error de tipo al llamar a {mh}. "
                f"Revisa MH_ARG_MAP['{mh}'] y la firma de la función."
            ) from e

        # --- Interpretar resultado según el tipo de MH ---
        new_population = None
        mh_state['posibles_mejoras'] = None  # Reset por defecto

        if self.mh_name in MH_RETURNS_MEJORAS:
            # LOA: retorna (population, posibles_mejoras)
            if isinstance(result, tuple) and len(result) == 2:
                new_population, mh_state['posibles_mejoras'] = result
            else:
                raise TypeError(
                    f"Retorno inesperado de {mh}. "
                    f"Se esperaba (population, posibles_mejoras)."
                )

        elif self.mh_name in MH_RETURNS_FIT_BEST:
            # GOAT: retorna (population, fitness, best)
            if isinstance(result, tuple) and len(result) == 3:
                new_population, fitness, _ = result
            else:
                raise TypeError(f"Retorno inesperado de {mh}.")

        elif self.mh_name in MH_RETURNS_FIT_PBEST:
            # TJO: retorna (population, fitness, pBest)
            if isinstance(result, tuple) and len(result) == 3:
                new_population, fitness, mh_state['pBest'] = result
            else:
                raise TypeError(f"Retorno inesperado de {mh}.")

        elif self.mh_name in MH_RETURNS_ARRAY_ONLY:
            # APO, CLO: retornan solo np.ndarray
            if isinstance(result, np.ndarray):
                new_population = result
            else:
                raise TypeError(f"Retorno inesperado de {mh}.")

        elif isinstance(result, tuple) and len(result) == 2:
            # Caso genérico: (population, vel)
            new_population, mh_state['vel'] = result

        elif isinstance(result, (np.ndarray, list)):
            # Caso genérico: solo retorna nueva población
            new_population = result

        else:
            raise TypeError(
                f"Tipo de retorno inesperado de {mh}: {type(result)}"
            )

        # Asegurar np.ndarray
        if not isinstance(new_population, np.ndarray):
            new_population = np.array(new_population)

        return new_population, fitness, mh_state

    def update_pbest(self, population, fitness, mh_state):
        """
        Actualiza pBest y pBestScore si la MH lo requiere (PSO, TJO).
        Llamar DESPUÉS de evaluar fitness.
        
        Args:
            population: Población actual.
            fitness:    Fitness actual.
            mh_state:   Estado interno.
        
        Returns:
            dict: mh_state actualizado.
        """
        if self.mh_name not in MH_NEEDS_PBEST:
            return mh_state

        pBest = mh_state['pBest']
        pBestScore = mh_state['pBestScore']

        if pBest is not None and pBestScore is not None:
            for i in range(population.shape[0]):
                if fitness[i] < pBestScore[i]:
                    pBestScore[i] = fitness[i]
                    pBest[i] = population[i].copy()

            mh_state['pBest'] = pBest
            mh_state['pBestScore'] = pBestScore

        return mh_state

    def __repr__(self):
        return (
            f"MetaheuristicAdapter("
            f"mh={self.mh_name}, resolved={self._resolved_name}, "
            f"pop={self.pop_size}, dim={self.dim})"
        )
