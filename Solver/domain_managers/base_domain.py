"""
Base Domain Manager (Clase Abstracta)
======================================
Define el contrato que todo dominio de problema debe implementar.

Un "dominio" responde a la pregunta: "¿Cómo se genera, evalúa y repara
una solución para ESTE tipo de problema?"

El Universal Solver solamente interactúa con esta interfaz,
sin saber si por detrás se resuelve un Benchmark continuo,
un Set Covering Problem, un Knapsack, o Feature Selection.
"""

from abc import ABC, abstractmethod
import numpy as np

from Diversity.imports import diversidadHussain


class BaseDomainManager(ABC):
    """
    Contrato base para todos los dominios de optimización.
    
    Responsabilidades:
        - Definir las dimensiones y límites del problema.
        - Generar la población inicial.
        - Evaluar el fitness de un individuo o de toda la población.
        - Reparar soluciones infactibles (si aplica).
        - Obtener el valor óptimo conocido (si aplica).
    
    Attributes:
        dim (int):         Número de dimensiones del problema.
        pop_size (int):    Tamaño de la población.
        lb (np.ndarray):   Límites inferiores por dimensión.
        ub (np.ndarray):   Límites superiores por dimensión.
        nfe (int):         Contador de evaluaciones de la función objetivo (FE).
    """

    def __init__(self, dim, pop_size, lb, ub):
        """
        Args:
            dim:      Dimensiones del problema.
            pop_size: Tamaño de la población.
            lb:       Límite inferior (escalar o array).
            ub:       Límite superior (escalar o array).
        """
        self.dim = dim
        self.pop_size = pop_size
        self.nfe = 0

        # Normalizar lb/ub a arrays numpy
        if not isinstance(lb, np.ndarray):
            lb = np.array(lb) if isinstance(lb, list) else np.full(dim, lb)
        if not isinstance(ub, np.ndarray):
            ub = np.array(ub) if isinstance(ub, list) else np.full(dim, ub)
        
        self.lb = lb
        self.ub = ub

    # ==================== MÉTODOS ABSTRACTOS ====================

    @abstractmethod
    def initialize_population(self):
        """
        Genera la población inicial según la naturaleza del dominio.
        
        Returns:
            np.ndarray: Población de forma (pop_size, dim).
        """
        pass

    @abstractmethod
    def evaluate(self, individual):
        """
        Evalúa el fitness de un individuo.
        Debe incrementar self.nfe internamente.
        
        Args:
            individual (np.ndarray): Solución candidata de forma (dim,).
        
        Returns:
            float: Valor de fitness del individuo.
        """
        pass

    @abstractmethod
    def fo(self, x):
        """
        Función objetivo para MH que evalúan internamente (TJO, GOAT, etc.).
        Debe retornar la solución procesada y su fitness.
        
        Args:
            x (np.ndarray): Solución candidata.
        
        Returns:
            tuple: (solución_procesada, fitness_value).
        """
        pass

    @abstractmethod
    def process_new_population(self, population, fitness, mh_name, mh_state):
        """
        Procesa la población completa después de la iteración de la MH.
        Incluye: clip/binarización, reparación, evaluación de fitness,
        y manejo de mejoras alternativas (LOA).
        
        Args:
            population: Población nueva del MH.
            fitness:    Vector de fitness (puede estar desactualizado).
            mh_name:    Nombre de la MH activa.
            mh_state:   Estado interno de la MH (contiene posibles_mejoras, etc.).
        
        Returns:
            tuple: (population_procesada, fitness_actualizado).
        """
        pass

    # ==================== MÉTODOS CONCRETOS (COMUNES) ====================

    def evaluate_population(self, population):
        """
        Evalúa el fitness de toda la población.
        
        Args:
            population (np.ndarray): Población de forma (pop_size, dim).
        
        Returns:
            np.ndarray: Vector de fitness de forma (pop_size,).
        """
        fitness = np.zeros(population.shape[0])
        for i in range(population.shape[0]):
            fitness[i] = self.evaluate(population[i])
        return fitness

    def clip_bounds(self, population):
        """
        Aplica los límites del dominio a la población.
        
        Args:
            population (np.ndarray): Población a limitar.
        
        Returns:
            np.ndarray: Población con valores dentro de [lb, ub].
        """
        return np.clip(population, self.lb, self.ub)

    def find_best(self, population, fitness):
        """
        Encuentra la mejor solución de la población.
        
        Args:
            population (np.ndarray): Población actual.
            fitness (np.ndarray):    Fitness de cada individuo.
        
        Returns:
            tuple: (best_solution, best_fitness)
        """
        best_idx = np.argmin(fitness)
        return population[best_idx].copy(), fitness[best_idx]

    def update_best(self, population, fitness, current_best, current_best_fitness):
        """
        Actualiza la mejor solución global si se encontró una mejor.
        
        Args:
            population:            Población actual.
            fitness:               Vector de fitness.
            current_best:          Mejor solución conocida hasta ahora.
            current_best_fitness:  Mejor fitness conocido hasta ahora.
        
        Returns:
            tuple: (best_solution, best_fitness) actualizados.
        """
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < current_best_fitness:
            return population[best_idx].copy(), fitness[best_idx]
        return current_best, current_best_fitness

    def compute_diversity(self, population):
        """
        Calcula la diversidad de Hussain de la población.
        
        Args:
            population (np.ndarray): Población actual.
        
        Returns:
            float: Índice de diversidad.
        """
        return diversidadHussain(population)

    def repair(self, individual):
        """
        Repara una solución infactible. Por defecto solo aplica clip.
        Los dominios con restricciones complejas (SCP, KP) deben sobreescribirlo.
        
        Args:
            individual (np.ndarray): Solución a reparar.
        
        Returns:
            np.ndarray: Solución reparada.
        """
        return np.clip(individual, self.lb, self.ub)

    def get_optimum(self):
        """
        Retorna el valor óptimo conocido del problema (si existe).
        
        Returns:
            float or None: Valor óptimo, o None si no se conoce.
        """
        return None

    def reset_nfe(self):
        """Reinicia el contador de evaluaciones de función."""
        self.nfe = 0

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"dim={self.dim}, pop={self.pop_size}, nfe={self.nfe})"
        )
