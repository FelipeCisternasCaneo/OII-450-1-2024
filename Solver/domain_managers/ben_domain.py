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
        self._optimum = optimo_raw * dim if function_name == "F8" else optimo_raw

    # ──────────────────────────────────────────────────────────────────────────
    # Propiedades de identidad (contrato BaseDomainManager)
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def label(self):
        """Identificador para nombres de archivo CSV y BD."""
        return self.function_name

    @property
    def domain_type(self):
        """Tipo de dominio para resolución de variantes."""
        return "BEN"

    def get_console_label(self, mh_name):
        """Etiqueta de consola: función | dim | MH."""
        return f"{self.function_name} | dim: {self.dim} | {mh_name}"

    def initialize_population(self):
        """
        Genera población con distribución uniforme en [lb, ub].

        Returns:
            np.ndarray: Población de forma (pop_size, dim).
        """
        return (
            np.random.uniform(0, 1, (self.pop_size, self.dim)) * (self.ub - self.lb)
            + self.lb
        )

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

        if mh_name in ("CLO", "TJO") or "fo" in MH_ARG_MAP.get(mh_name, []):
            return population, fitness

        # Para el resto: clip a bounds y evaluar
        population = self.clip_bounds(population)
        for i in range(population.shape[0]):
            fitness[i] = self.evaluate(population[i])

        # LOA: evaluar posibles mejoras y reemplazar si son superiores
        mejoras = mh_state.get("posibles_mejoras")
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


# ==================== EJECUCIÓN DE EXPERIMENTO ====================


def _execute_ben_experiment(id, data, datosInstancia, parametros):
    """Ejecuta un problema Benchmark usando el Universal Solver.

    Esta función encapsula la lógica que antes vivía en main.py:ejecutar_ben().
    Se registra en el Domain Registry como el execute_experiment de BEN.

    Args:
        id:               ID del experimento en la BD.
        data:             Fila completa del experimento (data[0]).
        datosInstancia:   Datos de la instancia desde BD.
        parametros:       Dict de parámetros parseados.
    """
    from Solver.universal_solver import universal_solver
    from Solver.termination_manager import TerminationCriteria

    experimento = data[0][1]
    parametrosInstancia = datosInstancia[0][4]

    dim = int(experimento.split(" ")[1])
    lb = float(parametrosInstancia.split(",")[0].split(":")[1])
    ub = float(parametrosInstancia.split(",")[1].split(":")[1])

    mh_name = parametros["mh"]
    pop_size = int(parametros["pop"])
    function_name = parametros["instancia"]

    domain = BenDomainManager(function_name, dim, pop_size, lb, ub)

    # Construir criterios de término
    max_iter_raw = parametros.get("iter")
    max_iter = int(max_iter_raw) if max_iter_raw not in (None, "") else None
    max_fe_raw = parametros.get("max_fe", parametros.get("fe"))
    max_fe = int(max_fe_raw) if max_fe_raw not in (None, "") else None
    termination = TerminationCriteria(max_iter=max_iter, max_fe=max_fe)

    universal_solver(id, mh_name, domain, termination)


# ==================== REGISTRO EN EL DOMAIN REGISTRY ====================


def _insert_ben_instances(bd):
    """Wrapper para insertar instancias BEN + CEC2017 en la BD."""
    bd.insertarInstanciasBEN()
    bd.insertarInstanciasCEC2017()


def _register():
    """Registra el dominio BEN en el registry."""
    from Solver.domain_managers.registry import register, DomainEntry

    register(
        DomainEntry(
            domain_type="BEN",
            config_key="ben",
            execute_experiment=_execute_ben_experiment,
            insert_instances=_insert_ben_instances,
            analysis_meta={
                "sub": "BEN",
                "inst_key": "BEN",
                "uses_bin": False,
                "title_prefix": "",
                "obtenerArchivos_kwargs": {"incluir_binarizacion": False},
            },
            instance_dir=None,
            default_extra_params="",
        )
    )


_register()
