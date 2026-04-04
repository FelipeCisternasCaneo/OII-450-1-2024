"""
BEN Domain Manager (Problemas Benchmark Continuos)
===================================================
Dominio para funciones de benchmark clásicas (F1-F23, CEC2017, etc.)

Características:
    - Espacio de búsqueda continuo con límites [lb, ub].
    - Población inicializada con distribución uniforme.
    - Sin reparación especial (solo clip a límites).
    - Fitness evaluado directamente con la función objetivo.
"""

import numpy as np

from Solver.domain_managers.base_domain import BaseDomainManager
from Problem.Benchmark.Problem import fitness as benchmark_fitness
from BD.sqlite import BD


class BenDomainManager(BaseDomainManager):
    """
    Dominio para problemas de optimización continua (Benchmark).
    
    Args:
        function_name: Identificador de la función (e.g. 'F1', 'F10', 'F1CEC2017').
        dim:           Dimensiones del espacio de búsqueda.
        pop_size:      Tamaño de la población.
        lb:            Límite inferior (escalar o lista).
        ub:            Límite superior (escalar o lista).
    """

    def __init__(self, function_name, dim, pop_size, lb, ub):
        super().__init__(dim, pop_size, lb, ub)
        self.function_name = function_name

        # Cachear el óptimo conocido desde la BD
        bd = BD()
        optimo_raw = bd.obtenerOptimoInstancia(function_name)[0][0]
        self._optimum = optimo_raw * dim if function_name == 'F8' else optimo_raw

    def initialize_population(self):
        """
        Genera población con distribución uniforme en [lb, ub].
        
        Returns:
            np.ndarray: Población de forma (pop_size, dim).
        """
        return np.random.uniform(0, 1, (self.pop_size, self.dim)) * (self.ub - self.lb) + self.lb

    def evaluate(self, individual):
        """
        Evalúa un individuo usando la función de benchmark.
        Incrementa el contador de NFE.
        
        Args:
            individual (np.ndarray): Solución candidata.
        
        Returns:
            float: Valor de fitness.
        """
        individual = np.clip(individual, self.lb, self.ub)
        self.nfe += 1
        return float(benchmark_fitness(self.function_name, individual))

    def evaluate_with_clip(self, individual):
        """
        Evalúa un individuo aplicando clip y retornando también la solución clipeada.
        Compatible con la interfaz fo(x) -> (x, fitness) que usan algunas MH.
        
        Args:
            individual (np.ndarray): Solución candidata.
        
        Returns:
            tuple: (individuo_clipeado, fitness).
        """
        x = np.clip(individual, self.lb, self.ub)
        self.nfe += 1
        return x, float(benchmark_fitness(self.function_name, x))

    def get_optimum(self):
        """
        Retorna el valor óptimo conocido de la función de benchmark.
        
        Returns:
            float: Valor óptimo.
        """
        return self._optimum

    def fo(self, x):
        """
        Función objetivo para MH que evalúan internamente.
        Alias directo de evaluate_with_clip.
        
        Args:
            x (np.ndarray): Solución candidata.
        
        Returns:
            tuple: (individuo_clipeado, fitness).
        """
        return self.evaluate_with_clip(x)

    def process_new_population(self, population, fitness, mh_name, mh_state):
        """
        Procesa la población después de la iteración de la MH para Benchmarks.
        
        - CLO/TJO: ya evaluaron internamente, solo se pasa.
        - Resto: clip a bounds → evaluar → manejar LOA mejoras.
        
        Args:
            population: Población nueva del MH.
            fitness:    Vector de fitness.
            mh_name:    Nombre de la MH activa.
            mh_state:   Estado interno (contiene posibles_mejoras para LOA).
        
        Returns:
            tuple: (population_procesada, fitness_actualizado).
        """
        # Si la MH ya usa "fo" internamente, o es un caso especial CLO/TJO, no hacemos re-evaluación
        from Metaheuristics.imports import MH_ARG_MAP
        if mh_name in ('CLO', 'TJO') or 'fo' in MH_ARG_MAP.get(mh_name, []):
            return population, fitness

        # Para el resto: clip a bounds y evaluar
        population = self.clip_bounds(population)
        for i in range(population.shape[0]):
            fitness[i] = self.evaluate(population[i])

        # LOA: evaluar posibles mejoras y reemplazar si son superiores
        mejoras = mh_state.get('posibles_mejoras')
        if mejoras is not None:
            mejoras = self.clip_bounds(mejoras)
            for i in range(mejoras.shape[0]):
                mejora_fit = self.evaluate(mejoras[i])
                if mejora_fit < fitness[i]:
                    population[i] = mejoras[i]
                    fitness[i] = mejora_fit

        return population, fitness

    def __repr__(self):
        return (
            f"BenDomainManager("
            f"func={self.function_name}, dim={self.dim}, "
            f"pop={self.pop_size}, nfe={self.nfe})"
        )
