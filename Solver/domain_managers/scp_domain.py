"""
SCP Domain Manager (Set Covering Problem)
==========================================
Dominio para problemas de Set Covering (SCP) y Unicost SCP (USCP).

Características:
    - Espacio de búsqueda binario {0, 1}.
    - Población inicializada con distribución binaria uniforme.
    - Reparación de soluciones infactibles (simple o complex).
    - Binarización configurable (DS parameter).
    - Soporte opcional para mapas caóticos (pregenera secuencia completa).
"""

import numpy as np

from Solver.domain_managers.base_domain import BaseDomainManager
from Problem.SCP.problem import SCP
from Problem.USCP.problem import USCP
from Discretization import discretization as b


# Mapas caóticos válidos (misma lista que solverSCP_Chaotic.py)
_VALID_CHAOTIC_MAPS = {
    "LOG",
    "SINE",
    "TENT",
    "CIRCLE",
    "SINGER",
    "SINU",
    "PIECE",
    "CHEB",
    "GAUS",
}


class ScpDomainManager(BaseDomainManager):
    """
    Dominio para Set Covering Problem (SCP/USCP).

    Soporta opcionalmente mapas caóticos para reemplazar los valores
    aleatorios en la función de binarización. Cuando ``chaotic_map_name``
    está activo se pregenera una secuencia de longitud
    ``chaotic_max_iter × pop_size × dim`` y se usa la misma fórmula de
    índice base que el solver legado ``population_SCP_Chaotic.py``:

        chaotic_index = (iter_counter * pop_size * dim) + (i * dim)

    donde ``iter_counter`` parte en 0 (equivalente a ``iter=1`` del legado).

    Args:
        instance_name:    Nombre de la instancia (e.g. 'scp41').
        pop_size:         Tamaño de la población.
        repair_type:      Tipo de reparación ('simple' o 'complex').
        ds:               Esquema de discretización (e.g. 'V3-ELIT').
        unicost:          True para usar USCP en vez de SCP.
        chaotic_map_name: Sigla del mapa caótico (e.g. 'LOG'). None → no caótico.
        chaotic_max_iter: Iteraciones máximas del experimento. Requerido cuando
                          chaotic_map_name está activo. Si falta se lanza ValueError
                          en lugar de degradar silenciosamente.
        chaotic_x0:       Condición inicial del mapa caótico (default 0.7).
    """

    def __init__(
        self,
        instance_name,
        pop_size,
        repair_type,
        ds,
        unicost=False,
        chaotic_map_name=None,
        chaotic_max_iter=None,
        chaotic_x0=0.7,
    ):
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

        # ── Modo caótico ──────────────────────────────────────────────────────
        self._chaotic_enabled = chaotic_map_name is not None

        if self._chaotic_enabled:
            # Validar nombre de mapa
            map_upper = chaotic_map_name.upper()
            if map_upper not in _VALID_CHAOTIC_MAPS:
                raise ValueError(
                    f"Mapa caótico '{chaotic_map_name}' inválido. "
                    f"Opciones: {sorted(_VALID_CHAOTIC_MAPS)}"
                )

            # Fallar rápido si falta max_iter (no degradar silenciosamente a 100)
            if chaotic_max_iter is None:
                raise ValueError(
                    "chaotic_max_iter es requerido cuando chaotic_map_name está activo. "
                    "Proporcione el número máximo de iteraciones del experimento para "
                    "pregenerar la secuencia caótica con la longitud correcta."
                )

            # Import tardío para no romper uso no-caótico
            from ChaoticMaps import get_chaotic_map

            self._chaotic_map_name = map_upper
            self._chaotic_max_iter = int(chaotic_max_iter)
            self._chaotic_x0 = float(chaotic_x0)

            # Pregenerar secuencia: longitud = max_iter * pop_size * dim
            sequence_length = self._chaotic_max_iter * pop_size * dim
            chaotic_func = get_chaotic_map(map_upper)
            self._chaotic_sequence = chaotic_func(
                x0=self._chaotic_x0, quantity=sequence_length
            )
            self._chaotic_sequence_len = sequence_length

            # Contador de iteraciones caóticas (parte en 0, equivale a iter=1 del legado)
            self._chaotic_iter_counter = 0
            self._chaotic_last_base_index = 0
        else:
            self._chaotic_map_name = None
            self._chaotic_max_iter = None
            self._chaotic_x0 = chaotic_x0
            self._chaotic_sequence = None
            self._chaotic_sequence_len = 0
            self._chaotic_iter_counter = 0
            self._chaotic_last_base_index = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Métodos de población
    # ──────────────────────────────────────────────────────────────────────────

    def initialize_population(self):
        """
        Genera población binaria aleatoria {0, 1}.

        Returns:
            np.ndarray: Población binaria de forma (pop_size, dim).
        """
        return np.random.randint(low=0, high=2, size=(self.pop_size, self.dim))

    # ──────────────────────────────────────────────────────────────────────────
    # Métodos de evaluación
    # ──────────────────────────────────────────────────────────────────────────

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

    # ──────────────────────────────────────────────────────────────────────────
    # Métodos de binarización
    # ──────────────────────────────────────────────────────────────────────────

    def binarize(self, continuous_solution, best, previous_binary, chaotic_index=None):
        """
        Aplica el esquema de discretización (DS) a una solución continua.

        En modo caótico, extrae los valores del slice
        ``chaotic_index : chaotic_index + dim`` de la secuencia pregenerada y los
        pasa a ``aplicarBinarizacion``. Si ``chaotic_index`` es None, usa los valores
        aleatorios estándar (comportamiento no-caótico idéntico al original).

        Args:
            continuous_solution: Solución en espacio continuo.
            best:               Mejor solución global actual.
            previous_binary:    Solución binaria anterior del individuo.
            chaotic_index (int, optional): Índice de inicio en la secuencia caótica.
                                           None → sin mapa caótico (aleatorio estándar).

        Returns:
            np.ndarray: Solución binarizada.
        """
        if chaotic_index is not None and self._chaotic_sequence is not None:
            return b.aplicarBinarizacion(
                continuous_solution,
                self.ds,
                best,
                previous_binary,
                chaotic_map=self._chaotic_sequence,
                chaotic_index=chaotic_index,
            )
        # Non-chaotic path: identical to original behavior
        return b.aplicarBinarizacion(
            continuous_solution, self.ds, best, previous_binary
        )

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

    # ──────────────────────────────────────────────────────────────────────────
    # Reparación
    # ──────────────────────────────────────────────────────────────────────────

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

    # ──────────────────────────────────────────────────────────────────────────
    # Acceso a instancia
    # ──────────────────────────────────────────────────────────────────────────

    def get_optimum(self):
        """
        Retorna el óptimo conocido de la instancia SCP.

        Returns:
            float: Valor óptimo.
        """
        return self.instance.getOptimum()

    def get_instance(self):
        """
        Retorna la instancia del problema SCP/USCP subyacente.
        Útil para acceder a métodos específicos de la instancia.

        Returns:
            SCP or USCP: Instancia del problema.
        """
        return self.instance

    # ──────────────────────────────────────────────────────────────────────────
    # Estado caótico
    # ──────────────────────────────────────────────────────────────────────────

    def get_chaotic_debug_state(self):
        """
        Retorna un diccionario pequeño con el estado caótico observable.
        Útil para pruebas unitarias y el harness AB sin exponer la secuencia completa.

        Returns:
            dict con claves:
                enabled (bool)        – True si el modo caótico está activo.
                map_name (str|None)   – Sigla del mapa caótico activo.
                sequence_length (int) – Longitud de la secuencia pregenerada.
                iter_counter (int)    – Iteraciones procesadas desde la creación.
                last_base_index (int) – Último índice base calculado.
        """
        return {
            "enabled": self._chaotic_enabled,
            "map_name": self._chaotic_map_name,
            "sequence_length": self._chaotic_sequence_len,
            "iter_counter": self._chaotic_iter_counter,
            "last_base_index": self._chaotic_last_base_index,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Estado de iteración (universal_solver.py lo invoca antes de cada iter)
    # ──────────────────────────────────────────────────────────────────────────

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

    # ──────────────────────────────────────────────────────────────────────────
    # fo(): función objetivo para MH con evaluación interna
    # ──────────────────────────────────────────────────────────────────────────

    def fo(self, x):
        """
        Función objetivo para MH que evalúan internamente (TJO, GOAT, etc.).
        Utiliza el estado de iteración (_current_best, _matrixBin) para
        binarizar, reparar y evaluar una solución.

        En modo caótico usa ``chaotic_index=0`` (igual que el solver legado
        ``solverSCP_Chaotic.fo``): fo() no tiene contexto de iter/i y el legado
        siempre usó el inicio de la secuencia como fallback.

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

        best = (
            self._current_best if self._current_best is not None else np.zeros(self.dim)
        )

        # En modo caótico: chaotic_index=0 (compatible con legado)
        if self._chaotic_enabled:
            x_bin = self.binarize(x, best, prev_binary, chaotic_index=0)
        else:
            x_bin = self.binarize(x, best, prev_binary)

        x_repaired, fitness = self.evaluate_and_repair(x_bin)
        return x_repaired, fitness

    # ──────────────────────────────────────────────────────────────────────────
    # process_new_population(): binarización + reparación de toda la población
    # ──────────────────────────────────────────────────────────────────────────

    def process_new_population(self, population, fitness, mh_name, mh_state):
        """
        Procesa la población después de la iteración de la MH para SCP/USCP.

        Pipeline: binarizar (excepto GA) → testFactibilidad → reparar → evaluar.
        También maneja LOA posibles_mejoras.

        En modo caótico aplica la fórmula de índice base del solver legado:
            chaotic_index = (iter_counter * pop_size * dim) + (i * dim)
        y avanza ``_chaotic_iter_counter`` al finalizar cada llamada.

        Args:
            population: Población nueva del MH (espacio continuo).
            fitness:    Vector de fitness (será recalculado).
            mh_name:    Nombre de la MH activa ('GA' salta binarización).
            mh_state:   Estado interno (contiene posibles_mejoras para LOA).

        Returns:
            tuple: (population_binaria_reparada, fitness_actualizado).
        """
        best = (
            self._current_best if self._current_best is not None else np.zeros(self.dim)
        )

        # Binarizar (GA produce directamente soluciones binarias)
        if mh_name != "GA" and self._matrixBin is not None:
            if self._chaotic_enabled:
                # Fórmula legacy: base = iter_counter * pop_size * dim
                base = self._chaotic_iter_counter * self.pop_size * self.dim
                self._chaotic_last_base_index = base
                for i in range(population.shape[0]):
                    chaotic_index = base + (i * self.dim)
                    population[i] = self.binarize(
                        population[i],
                        best,
                        self._matrixBin[i],
                        chaotic_index=chaotic_index,
                    )
            else:
                for i in range(population.shape[0]):
                    population[i] = self.binarize(
                        population[i], best, self._matrixBin[i]
                    )

        # Reparar + Evaluar cada individuo
        for i in range(population.shape[0]):
            population[i], fitness[i] = self.evaluate_and_repair(population[i])

        # LOA: evaluar posibles mejoras y reemplazar si son superiores
        mejoras = mh_state.get("posibles_mejoras")
        if mejoras is not None:
            for i in range(mejoras.shape[0]):
                mejora_bin, mejora_fit = self.fo(mejoras[i])
                if mejora_fit < fitness[i]:
                    population[i] = mejora_bin
                    fitness[i] = mejora_fit

        # Actualizar matrixBin con la población procesada
        self._matrixBin = population.copy()

        # Avanzar el contador de iteraciones caóticas
        if self._chaotic_enabled:
            self._chaotic_iter_counter += 1

        return population, fitness

    # ──────────────────────────────────────────────────────────────────────────
    # Representación
    # ──────────────────────────────────────────────────────────────────────────

    def __repr__(self):
        tipo = "USCP" if self.unicost else "SCP"
        chaotic_info = (
            f", map={self._chaotic_map_name}" if self._chaotic_enabled else ""
        )
        return (
            f"ScpDomainManager("
            f"tipo={tipo}, inst={self.instance_name}, "
            f"dim={self.dim}, ds={self.ds}, nfe={self.nfe}{chaotic_info})"
        )
