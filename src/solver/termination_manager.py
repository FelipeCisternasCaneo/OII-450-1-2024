"""
Módulo de Criterio de Parada Flexible (Termination Manager)
==========================================================
Permite elegir entre parada por Iteraciones, por Function Evaluations (FE),
o ambos simultáneamente (el primero que se cumpla).

Uso básico:
    # Solo por iteraciones (comportamiento legacy)
    tc = TerminationCriteria(max_iter=500)

    # Solo por evaluaciones de función
    tc = TerminationCriteria(max_fe=100000)

    # Ambos (el primero que se cumpla)
    tc = TerminationCriteria(max_iter=500, max_fe=100000)

    # En el bucle del solver:
    while not tc.is_met():
        ...
        tc.increment_iter()
        tc.add_fe(pop_size)   # o tc.add_fe(1) por cada evaluación individual
"""


class TerminationCriteria:
    """
    Criterio de terminación flexible para el framework de metaheurísticas.

    Soporta dos modos de parada independientes que pueden combinarse:
      - max_iter: Número máximo de iteraciones del bucle principal.
      - max_fe:   Número máximo de evaluaciones de la función objetivo (FE).

    Cuando ambos se especifican, la ejecución termina en cuanto
    CUALQUIERA de los dos límites se alcance primero.

    Attributes:
        current_iter (int): Iteración actual (comienza en 0).
        current_fe (int):   Evaluaciones de función acumuladas.
    """

    def __init__(self, max_iter=None, max_fe=None):
        """
        Args:
            max_iter: Límite máximo de iteraciones. None = sin límite por iteraciones.
            max_fe:   Límite máximo de evaluaciones de función. None = sin límite por FE.

        Raises:
            ValueError: Si no se proporciona al menos un criterio de parada.
        """
        if max_iter is None and max_fe is None:
            raise ValueError(
                "Debe especificarse al menos un criterio de parada: "
                "'max_iter' y/o 'max_fe'."
            )

        self.max_iter = max_iter
        self.max_fe = max_fe
        self.current_iter = 0
        self.current_fe = 0

    def is_met(self):
        """
        Verifica si alguno de los criterios de parada se ha alcanzado.

        Returns:
            bool: True si la ejecución debe terminar.
        """
        if self.max_iter is not None and self.current_iter >= self.max_iter:
            return True
        if self.max_fe is not None and self.current_fe >= self.max_fe:
            return True
        return False

    def increment_iter(self):
        """Incrementa el contador de iteraciones en 1."""
        self.current_iter += 1

    def add_fe(self, count=1):
        """
        Acumula evaluaciones de función.

        Args:
            count: Número de evaluaciones a sumar (default: 1).
                   Típicamente será el tamaño de la población si se evalúa
                   toda la población de golpe.
        """
        self.current_fe += count

    def remaining_fe(self):
        """
        Retorna cuántas evaluaciones quedan antes de alcanzar max_fe.

        Returns:
            int o None: Evaluaciones restantes, o None si no hay límite por FE.
        """
        if self.max_fe is None:
            return None
        return max(0, self.max_fe - self.current_fe)

    def remaining_iter(self):
        """
        Retorna cuántas iteraciones quedan antes de alcanzar max_iter.

        Returns:
            int o None: Iteraciones restantes, o None si no hay límite por iteraciones.
        """
        if self.max_iter is None:
            return None
        return max(0, self.max_iter - self.current_iter)

    def progress(self):
        """
        Retorna el progreso como fracción [0.0, 1.0] basado en el criterio
        más avanzado de los dos.

        Returns:
            float: Progreso entre 0.0 (inicio) y 1.0 (terminado).
        """
        progress_values = []

        if self.max_iter is not None and self.max_iter > 0:
            progress_values.append(self.current_iter / self.max_iter)

        if self.max_fe is not None and self.max_fe > 0:
            progress_values.append(self.current_fe / self.max_fe)

        if not progress_values:
            return 0.0

        return min(1.0, max(progress_values))

    def reset(self):
        """Reinicia ambos contadores a cero para una nueva corrida."""
        self.current_iter = 0
        self.current_fe = 0

    def __repr__(self):
        parts = []
        if self.max_iter is not None:
            parts.append(f"iter={self.current_iter}/{self.max_iter}")
        if self.max_fe is not None:
            parts.append(f"fe={self.current_fe}/{self.max_fe}")
        return f"TerminationCriteria({', '.join(parts)})"


# Metaheurísticas que ejecutan 2 rondas de evaluación de función por iteración.
# Esto afecta la estimación de iteraciones efectivas cuando solo se dispone de max_fe.
_DOUBLE_FE_MHS = frozenset({"SBOA", "SSO", "TDO"})


def resolve_effective_max_iter(termination, pop_size, mh_name):
    """
    Resuelve el número efectivo de iteraciones máximas para una MH.

    Cuando ``termination.max_iter`` está disponible se retorna directamente.
    Si solo se dispone de ``max_fe`` (modo FE-only), se estima el número de
    iteraciones a partir del presupuesto de evaluaciones de función, teniendo
    en cuenta que algunas metaheurísticas (SBOA, SSO, TDO) realizan 2 rondas
    de evaluación por iteración.

    Esta función es pura: NO modifica ``termination`` ni ningún estado externo.

    Args:
        termination: Instancia de TerminationCriteria.
        pop_size:    Tamaño de la población del experimento.
        mh_name:     Nombre de la metaheurística (e.g. 'GWO', 'SBOA').

    Returns:
        int: Número efectivo de iteraciones máximas (>= 1).
    """
    if termination.max_iter is not None:
        return termination.max_iter

    fe_mult = 2 if mh_name in _DOUBLE_FE_MHS else 1
    return max(1, (termination.max_fe or 10000) // (pop_size * fe_mult))
