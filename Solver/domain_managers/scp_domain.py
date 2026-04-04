"""
SCP Domain Manager (Set Covering Problem)
==========================================
Dominio para problemas de Set Covering (SCP) y Unicost SCP (USCP).

Características:
    - Espacio de búsqueda binario {0, 1}.
    - Población inicializada con distribución binaria uniforme.
    - Reparación de soluciones infactibles (simple o complex).
    - Binarización configurable (DS parameter).
    - Soporte opcional para mapas caóticos.
"""

import numpy as np

from Solver.domain_managers.base_domain import BaseDomainManager
from Problem.SCP.problem import SCP
from Problem.USCP.problem import USCP
from Discretization import discretization as b


class ScpDomainManager(BaseDomainManager):
    """
    Dominio para Set Covering Problem (SCP/USCP).
    
    Args:
        instance_name: Nombre de la instancia (e.g. 'scp41').
        pop_size:      Tamaño de la población.
        repair_type:   Tipo de reparación ('simple' o 'complex').
        ds:            Esquema de discretización (e.g. 'V3-ELIT').
        unicost:       True para usar USCP en vez de SCP.
    """

    def __init__(self, instance_name, pop_size, repair_type, ds, unicost=False):
        # Cargar la instancia del problema
        self.instance = USCP(instance_name) if unicost else SCP(instance_name)
        
        dim = self.instance.getColumns()
        lb = np.zeros(dim)
        ub = np.ones(dim)
        
        super().__init__(dim, pop_size, lb, ub)
        
        self.instance_name = instance_name
        self.repair_type = repair_type
        self.ds = ds
        self.unicost = unicost

        # Estado mutable por iteración (para fo() y process_new_population)
        self._matrixBin = None
        self._current_best = None

    def initialize_population(self):
        """
        Genera población binaria aleatoria {0, 1}.
        
        Returns:
            np.ndarray: Población binaria de forma (pop_size, dim).
        """
        return np.random.randint(low=0, high=2, size=(self.pop_size, self.dim))

    def evaluate(self, individual):
        """
        Evalúa un individuo: test de factibilidad, reparación si es infactible,
        y cálculo de fitness (costo total de columnas seleccionadas).
        
        Args:
            individual (np.ndarray): Solución candidata binaria.
        
        Returns:
            float: Valor de fitness (costo).
        """
        flag, _ = self.instance.factibilityTest(individual)
        if not flag:
            individual = self.instance.repair(individual, self.repair_type)
        
        self.nfe += 1
        return self.instance.fitness(individual)

    def evaluate_and_repair(self, individual):
        """
        Evalúa un individuo retornando también la solución reparada.
        Útil para actualizar la población con la versión reparada.
        
        Args:
            individual (np.ndarray): Solución candidata binaria.
        
        Returns:
            tuple: (individuo_reparado, fitness).
        """
        flag, _ = self.instance.factibilityTest(individual)
        if not flag:
            individual = self.instance.repair(individual, self.repair_type)
        
        self.nfe += 1
        return individual, self.instance.fitness(individual)

    def binarize(self, continuous_solution, best, previous_binary):
        """
        Aplica el esquema de discretización (DS) a una solución continua.
        
        Args:
            continuous_solution: Solución en espacio continuo.
            best:               Mejor solución global actual.
            previous_binary:    Solución binaria anterior del individuo.
        
        Returns:
            np.ndarray: Solución binarizada.
        """
        return b.aplicarBinarizacion(continuous_solution, self.ds, best, previous_binary)

    def binarize_and_evaluate(self, individual, best, previous_binary):
        """
        Pipeline completo: binarizar → reparar → evaluar.
        
        Args:
            individual:       Solución en espacio continuo.
            best:             Mejor solución global actual.
            previous_binary:  Solución binaria anterior del individuo.
        
        Returns:
            tuple: (individuo_binario_reparado, fitness).
        """
        binary = self.binarize(individual, best, previous_binary)
        return self.evaluate_and_repair(binary)

    def repair(self, individual):
        """
        Repara una solución SCP infactible usando el tipo configurado.
        
        Args:
            individual (np.ndarray): Solución binaria.
        
        Returns:
            np.ndarray: Solución reparada.
        """
        flag, _ = self.instance.factibilityTest(individual)
        if not flag:
            return self.instance.repair(individual, self.repair_type)
        return individual

    def get_optimum(self):
        """
        Retorna el óptimo conocido de la instancia SCP.
        
        Returns:
            float: Valor óptimo.
        """
        return self.instance.getOptimum()

    def set_iteration_state(self, best, matrixBin):
        """
        Actualiza el estado mutable que fo() y process_new_population()
        necesitan en cada iteración.
        
        Args:
            best:      Mejor solución global actual.
            matrixBin: Copia de la población binaria actual.
        """
        self._current_best = best
        self._matrixBin = matrixBin

    def fo(self, x):
        """
        Función objetivo para MH que evalúan internamente (TJO, GOAT, etc.).
        Utiliza el estado de iteración (_current_best, _matrixBin) para
        binarizar, reparar y evaluar una solución.
        
        IMPORTANTE: Llamar set_iteration_state() antes de cada iteración.
        
        Args:
            x (np.ndarray): Solución candidata en espacio continuo.
        
        Returns:
            tuple: (solución_binaria_reparada, fitness).
        """
        # Usar la última fila de matrixBin como referencia (comportamiento legacy)
        if self._matrixBin is not None:
            prev_binary = self._matrixBin[-1]
        else:
            prev_binary = np.zeros(self.dim)

        best = self._current_best if self._current_best is not None else np.zeros(self.dim)

        x_bin = self.binarize(x, best, prev_binary)
        _, fitness = self.evaluate_and_repair(x_bin)
        return x, fitness

    def process_new_population(self, population, fitness, mh_name, mh_state):
        """
        Procesa la población después de la iteración de la MH para SCP/USCP.
        
        Pipeline: binarizar (excepto GA) → testFactibilidad → reparar → evaluar.
        También maneja LOA posibles_mejoras.
        
        Args:
            population: Población nueva del MH (espacio continuo).
            fitness:    Vector de fitness (será recalculado).
            mh_name:    Nombre de la MH activa ('GA' salta binarización).
            mh_state:   Estado interno (contiene posibles_mejoras para LOA).
        
        Returns:
            tuple: (population_binaria_reparada, fitness_actualizado).
        """
        best = self._current_best if self._current_best is not None else np.zeros(self.dim)

        # Binarizar (GA produce directamente soluciones binarias)
        if mh_name != 'GA' and self._matrixBin is not None:
            for i in range(population.shape[0]):
                population[i] = self.binarize(population[i], best, self._matrixBin[i])

        # Reparar + Evaluar cada individuo
        for i in range(population.shape[0]):
            population[i], fitness[i] = self.evaluate_and_repair(population[i])

        # LOA: evaluar posibles mejoras y reemplazar si son superiores
        mejoras = mh_state.get('posibles_mejoras')
        if mejoras is not None:
            for i in range(mejoras.shape[0]):
                _, mejora_fit = self.fo(mejoras[i])
                if mejora_fit < fitness[i]:
                    population[i] = mejoras[i]
                    fitness[i] = mejora_fit

        # Actualizar matrixBin con la población procesada
        self._matrixBin = population.copy()

        return population, fitness

    def get_instance(self):
        """
        Retorna la instancia del problema SCP/USCP subyacente.
        Útil para acceder a métodos específicos de la instancia.
        
        Returns:
            SCP or USCP: Instancia del problema.
        """
        return self.instance

    def __repr__(self):
        tipo = "USCP" if self.unicost else "SCP"
        return (
            f"ScpDomainManager("
            f"tipo={tipo}, inst={self.instance_name}, "
            f"dim={self.dim}, ds={self.ds}, nfe={self.nfe})"
        )
