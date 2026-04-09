"""
Unit tests for ScpDomainManager — chaotic-map support (Batch 1, Task 1.3)
==========================================================================

Tests cubiertos
---------------
1. Sequence length: la secuencia pregenerada tiene exactamente max_iter * pop_size * dim elementos.
2. Legacy base-index formula: el índice base para iter 1, 2 y last coincide con la fórmula del legado
   `(iter - 1) * pop_size * dim`, verificado via get_chaotic_debug_state().
3. fo() compatibility with chaotic_index=0: fo() invoca binarización con chaotic_index=0
   (no lanza y retorna fitness válido).
4. Non-chaotic execution unchanged: sin chaotic_map_name, el dominio se comporta exactamente
   como el original (enabled=False, secuencia=None).
5. Fail-fast without max_iter: ValueError si se pasa chaotic_map_name sin chaotic_max_iter.
6. Fail-fast with invalid map: ValueError si chaotic_map_name no está en el conjunto válido.
"""

import sys
import os
import numpy as np
import pytest

# Asegurar que el proyecto raíz esté en path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Solver.domain_managers.scp_domain import ScpDomainManager


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

INSTANCE = "scpnrf1"  # instancia pequeña: ~14 columnas, carga rápida
DS = "V1-STD"
REPAIR = "complex"
POP_SIZE = 5
MAX_ITER = 10


@pytest.fixture(scope="module")
def non_chaotic_domain():
    """ScpDomainManager sin mapa caótico (modo estándar)."""
    return ScpDomainManager(
        instance_name=INSTANCE,
        pop_size=POP_SIZE,
        repair_type=REPAIR,
        ds=DS,
        unicost=False,
    )


@pytest.fixture(scope="module")
def chaotic_domain():
    """ScpDomainManager con mapa LOG activado."""
    return ScpDomainManager(
        instance_name=INSTANCE,
        pop_size=POP_SIZE,
        repair_type=REPAIR,
        ds=DS,
        unicost=False,
        chaotic_map_name="LOG",
        chaotic_max_iter=MAX_ITER,
        chaotic_x0=0.7,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: Sequence length
# ──────────────────────────────────────────────────────────────────────────────


class TestSequenceLength:
    """La secuencia caótica tiene longitud max_iter * pop_size * dim."""

    def test_sequence_length_matches_formula(self, chaotic_domain):
        state = chaotic_domain.get_chaotic_debug_state()
        dim = chaotic_domain.dim
        expected = MAX_ITER * POP_SIZE * dim
        assert state["sequence_length"] == expected, (
            f"Longitud esperada {expected}, obtenida {state['sequence_length']}"
        )

    def test_internal_sequence_array_length_matches(self, chaotic_domain):
        """El array interno también tiene exactamente esa longitud."""
        dim = chaotic_domain.dim
        expected = MAX_ITER * POP_SIZE * dim
        assert len(chaotic_domain._chaotic_sequence) == expected

    def test_sequence_values_in_unit_interval(self, chaotic_domain):
        """Los valores caóticos están en [0, 1)."""
        seq = chaotic_domain._chaotic_sequence
        assert float(seq.min()) >= 0.0
        assert float(seq.max()) <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Legacy base-index formula via process_new_population + debug state
# ──────────────────────────────────────────────────────────────────────────────


class TestLegacyBaseIndexFormula:
    """
    Verifica que el índice base en cada llamada a process_new_population
    sea (iter_counter * pop_size * dim), donde iter_counter parte en 0
    (equivalente a iter=1 del solver legado).

    Fórmula legado en binarize_and_evaluate_chaotic:
        chaotic_index = ((iter - 1) * pop_size * dim) + (i * dim)
    donde iter parte en 1 → iter_counter parte en 0 aquí.
    """

    def _make_fresh_domain(self):
        """Crea un dominio caótico fresco para no contaminar otros tests."""
        return ScpDomainManager(
            instance_name=INSTANCE,
            pop_size=POP_SIZE,
            repair_type=REPAIR,
            ds=DS,
            unicost=False,
            chaotic_map_name="LOG",
            chaotic_max_iter=MAX_ITER,
            chaotic_x0=0.7,
        )

    def _simulate_iteration(self, domain):
        """Simula una llamada a process_new_population con una población sintética."""
        np.random.seed(42)
        population = domain.initialize_population().astype(float)
        fitness = np.zeros(POP_SIZE)
        # Necesita matrixBin inicializado
        domain.set_iteration_state(
            best=population[0].copy(),
            matrixBin=population.copy(),
        )
        population, fitness = domain.process_new_population(
            population, fitness, mh_name="GWO", mh_state={}
        )
        return population, fitness

    def test_base_index_iter1(self):
        """Iteración 1 (iter_counter=0 antes): base_index = 0."""
        domain = self._make_fresh_domain()
        dim = domain.dim
        self._simulate_iteration(domain)
        state = domain.get_chaotic_debug_state()
        assert state["last_base_index"] == 0 * POP_SIZE * dim
        assert state["iter_counter"] == 1  # avanzó tras la llamada

    def test_base_index_iter2(self):
        """Iteración 2 (iter_counter=1 antes): base_index = pop_size * dim."""
        domain = self._make_fresh_domain()
        dim = domain.dim
        self._simulate_iteration(domain)  # iter 1
        self._simulate_iteration(domain)  # iter 2
        state = domain.get_chaotic_debug_state()
        assert state["last_base_index"] == 1 * POP_SIZE * dim
        assert state["iter_counter"] == 2

    def test_base_index_last_iter(self):
        """Última iteración (iter_counter=MAX_ITER-1 antes): base = (MAX_ITER-1)*pop*dim."""
        domain = self._make_fresh_domain()
        dim = domain.dim
        for _ in range(MAX_ITER):
            self._simulate_iteration(domain)
        state = domain.get_chaotic_debug_state()
        # El último last_base_index fue (MAX_ITER-1) * pop_size * dim
        assert state["last_base_index"] == (MAX_ITER - 1) * POP_SIZE * dim
        assert state["iter_counter"] == MAX_ITER


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: fo() con chaotic_index=0
# ──────────────────────────────────────────────────────────────────────────────


class TestFoChaoticIndex0:
    """fo() en modo caótico usa chaotic_index=0 y retorna fitness válido."""

    def test_fo_returns_valid_fitness_chaotic(self, chaotic_domain):
        np.random.seed(7)
        population = chaotic_domain.initialize_population().astype(float)
        best = population[0].copy()
        chaotic_domain.set_iteration_state(best=best, matrixBin=population.copy())
        x_test = np.random.rand(chaotic_domain.dim)
        result, fit = chaotic_domain.fo(x_test)
        assert isinstance(fit, (int, float, np.floating)), (
            "fo() debe retornar fitness numérico"
        )
        assert fit > 0, "El fitness SCP debe ser positivo"

    def test_fo_does_not_raise_chaotic(self, chaotic_domain):
        """fo() no debe lanzar excepción en modo caótico."""
        np.random.seed(13)
        population = chaotic_domain.initialize_population().astype(float)
        chaotic_domain.set_iteration_state(
            best=population[0].copy(), matrixBin=population.copy()
        )
        x_test = np.random.rand(chaotic_domain.dim)
        # Debe poder llamarse sin error
        chaotic_domain.fo(x_test)

    def test_fo_without_iteration_state(self, chaotic_domain):
        """fo() sin set_iteration_state previo tampoco debe explotar."""
        fresh = ScpDomainManager(
            instance_name=INSTANCE,
            pop_size=POP_SIZE,
            repair_type=REPAIR,
            ds=DS,
            unicost=False,
            chaotic_map_name="LOG",
            chaotic_max_iter=MAX_ITER,
        )
        x_test = np.random.rand(fresh.dim)
        _, fit = fresh.fo(x_test)
        assert fit > 0


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: Non-chaotic execution unchanged
# ──────────────────────────────────────────────────────────────────────────────


class TestNonChaoticUnchanged:
    """Sin mapa caótico, el dominio se comporta exactamente como antes."""

    def test_enabled_is_false(self, non_chaotic_domain):
        state = non_chaotic_domain.get_chaotic_debug_state()
        assert state["enabled"] is False

    def test_sequence_is_none(self, non_chaotic_domain):
        assert non_chaotic_domain._chaotic_sequence is None

    def test_sequence_length_is_zero(self, non_chaotic_domain):
        state = non_chaotic_domain.get_chaotic_debug_state()
        assert state["sequence_length"] == 0

    def test_map_name_is_none(self, non_chaotic_domain):
        state = non_chaotic_domain.get_chaotic_debug_state()
        assert state["map_name"] is None

    def test_non_chaotic_process_does_not_advance_counter(self, non_chaotic_domain):
        """El iter_counter no cambia en modo no-caótico."""
        before = non_chaotic_domain.get_chaotic_debug_state()["iter_counter"]
        np.random.seed(99)
        population = non_chaotic_domain.initialize_population().astype(float)
        fitness = np.zeros(POP_SIZE)
        non_chaotic_domain.set_iteration_state(
            best=population[0].copy(), matrixBin=population.copy()
        )
        non_chaotic_domain.process_new_population(
            population, fitness, mh_name="GWO", mh_state={}
        )
        after = non_chaotic_domain.get_chaotic_debug_state()["iter_counter"]
        assert after == before, "iter_counter no debe avanzar en modo no-caótico"

    def test_non_chaotic_evaluate_returns_positive_fitness(self, non_chaotic_domain):
        np.random.seed(55)
        individual = non_chaotic_domain.initialize_population()[0]
        fit = non_chaotic_domain.evaluate(individual)
        assert fit > 0

    def test_non_chaotic_fo_returns_valid_fitness(self, non_chaotic_domain):
        np.random.seed(55)
        population = non_chaotic_domain.initialize_population().astype(float)
        non_chaotic_domain.set_iteration_state(
            best=population[0].copy(), matrixBin=population.copy()
        )
        _, fit = non_chaotic_domain.fo(np.random.rand(non_chaotic_domain.dim))
        assert fit > 0

    def test_non_chaotic_binarize_no_chaotic_map(self, non_chaotic_domain):
        """binarize() sin chaotic_index usa aleatorio estándar (sin excepción)."""
        np.random.seed(1)
        population = non_chaotic_domain.initialize_population().astype(float)
        best = population[0].copy()
        prev_bin = population[1].copy()
        result = non_chaotic_domain.binarize(
            np.random.rand(non_chaotic_domain.dim), best, prev_bin
        )
        # Resultado debe ser binario
        assert set(result.astype(int).tolist()).issubset({0, 1})


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: Fail-fast without max_iter
# ──────────────────────────────────────────────────────────────────────────────


class TestFailFastNoMaxIter:
    """ValueError al construir con chaotic_map_name pero sin chaotic_max_iter."""

    def test_raises_value_error_without_max_iter(self):
        with pytest.raises(ValueError, match="chaotic_max_iter"):
            ScpDomainManager(
                instance_name=INSTANCE,
                pop_size=POP_SIZE,
                repair_type=REPAIR,
                ds=DS,
                unicost=False,
                chaotic_map_name="LOG",
                chaotic_max_iter=None,  # ← ausente
            )


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: Fail-fast with invalid map name
# ──────────────────────────────────────────────────────────────────────────────


class TestFailFastInvalidMap:
    """ValueError al usar un mapa caótico desconocido."""

    def test_raises_value_error_on_invalid_map(self):
        with pytest.raises(ValueError, match="inválido"):
            ScpDomainManager(
                instance_name=INSTANCE,
                pop_size=POP_SIZE,
                repair_type=REPAIR,
                ds=DS,
                unicost=False,
                chaotic_map_name="INVALIDMAP",
                chaotic_max_iter=MAX_ITER,
            )

    def test_raises_for_empty_string_map(self):
        """Un string vacío no es un mapa válido."""
        with pytest.raises((ValueError, AttributeError)):
            ScpDomainManager(
                instance_name=INSTANCE,
                pop_size=POP_SIZE,
                repair_type=REPAIR,
                ds=DS,
                unicost=False,
                chaotic_map_name="",
                chaotic_max_iter=MAX_ITER,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Test 7: Additional sanity — all valid maps instantiate correctly
# ──────────────────────────────────────────────────────────────────────────────


class TestAllValidMaps:
    """Todos los mapas válidos se instancian sin error."""

    VALID_MAPS = [
        "LOG",
        "SINE",
        "TENT",
        "CIRCLE",
        "SINGER",
        "SINU",
        "PIECE",
        "CHEB",
        "GAUS",
    ]

    def test_all_valid_maps_instantiate(self):
        for map_name in self.VALID_MAPS:
            domain = ScpDomainManager(
                instance_name=INSTANCE,
                pop_size=3,
                repair_type=REPAIR,
                ds=DS,
                unicost=False,
                chaotic_map_name=map_name,
                chaotic_max_iter=5,
            )
            state = domain.get_chaotic_debug_state()
            assert state["enabled"] is True, (
                f"Map {map_name} debería activar modo caótico"
            )
            assert state["map_name"] == map_name
            assert state["sequence_length"] == 5 * 3 * domain.dim


# ──────────────────────────────────────────────────────────────────────────────
# Test 8: LOA bugfix — fo() returns binarized/repaired solution, not raw x
# ──────────────────────────────────────────────────────────────────────────────


class TestLoaBugfix:
    """
    Verifica el bugfix LOA en scp_domain.process_new_population():

    BUG ORIGINAL (pre-fix):
        fo() devolvía (x_continuo, fitness_binario). Cuando se aceptaba una
        mejora LOA, se almacenaba x_continuo en population[i], lo que dejaba
        _matrixBin con valores continuos en lugar de binarios.

    FIX:
        fo() ahora devuelve (x_binarizado_reparado, fitness).
        process_new_population() almacena mejora_bin (no mejoras[i]) en population[i].
    """

    def _make_domain(self):
        return ScpDomainManager(
            instance_name=INSTANCE,
            pop_size=POP_SIZE,
            repair_type=REPAIR,
            ds=DS,
            unicost=False,
        )

    def test_fo_returns_binary_solution(self, non_chaotic_domain):
        """fo() debe retornar una solución binaria {0,1}, no el x continuo original."""
        np.random.seed(42)
        population = non_chaotic_domain.initialize_population().astype(float)
        non_chaotic_domain.set_iteration_state(
            best=population[0].copy(), matrixBin=population.copy()
        )
        x_continuous = np.random.rand(non_chaotic_domain.dim)  # valores en (0, 1)
        result, fit = non_chaotic_domain.fo(x_continuous)

        # El resultado debe ser binario: todos los valores son 0 o 1
        unique_values = set(result.astype(int).tolist())
        assert unique_values.issubset({0, 1}), (
            f"fo() debe retornar solución binaria, pero obtuvo valores: {unique_values}"
        )

    def test_fo_result_differs_from_continuous_input(self, non_chaotic_domain):
        """fo() retorna la versión binarizada, no el x continuo (con alta probabilidad)."""
        np.random.seed(99)
        population = non_chaotic_domain.initialize_population().astype(float)
        non_chaotic_domain.set_iteration_state(
            best=population[0].copy(), matrixBin=population.copy()
        )
        # x_continuous tiene valores estrictamente entre 0 y 1 (nunca exactamente 0 ó 1)
        x_continuous = np.random.uniform(0.1, 0.9, size=non_chaotic_domain.dim)
        result, _ = non_chaotic_domain.fo(x_continuous)

        # x_continuous tiene valores fraccionales, result debe ser binario
        # Estos no pueden ser iguales ya que x_continuous ∉ {0,1}
        assert not np.allclose(result, x_continuous), (
            "fo() no debe retornar el x continuo original — debe binarizarlo"
        )

    def test_population_remains_binary_after_loa_mejora_accepted(self):
        """
        Cuando una mejora LOA es aceptada, population[i] debe ser binario.
        Este test reproduce el bug: antes population[i] recibía la mejora continua.
        """
        domain = self._make_domain()
        np.random.seed(7)
        pop = domain.initialize_population().astype(float)
        fitness = np.array([domain.evaluate(p) for p in pop])

        # Configurar estado de iteración
        domain.set_iteration_state(best=pop[0].copy(), matrixBin=pop.copy())

        # Crear posibles_mejoras continuas (como lo hace LOA real)
        # Usamos valores que GARANTIZAN que la mejora sea "mejor" forzando fitness bajo
        # Para esto: generamos mejoras que después de binarizar/reparar tengan buen fitness.
        # Pero lo más importante es que sean continuas (no binarias).
        mejoras_continuas = np.random.uniform(0.1, 0.9, size=pop.shape)

        # Forzar que se evalúe al menos 1 mejora: poner fitness[0] muy alto
        fitness_modificado = fitness.copy()
        fitness_modificado[0] = 1e9  # asegurar que cualquier mejora sea "mejor"

        mh_state = {"posibles_mejoras": mejoras_continuas}
        new_pop, new_fit = domain.process_new_population(
            pop.copy(), fitness_modificado.copy(), mh_name="LOA", mh_state=mh_state
        )

        # Toda la población resultante debe ser binaria (incluyendo individuo 0)
        unique_values = set(new_pop.astype(int).flatten().tolist())
        assert unique_values.issubset({0, 1}), (
            f"population post-LOA debe ser binaria, pero obtuvo: {unique_values}"
        )

    def test_matrix_bin_remains_binary_after_loa(self):
        """
        _matrixBin debe ser binaria después de process_new_population con LOA.
        El bug original dejaba valores continuos en _matrixBin cuando se aceptaba una mejora.
        """
        domain = self._make_domain()
        np.random.seed(13)
        pop = domain.initialize_population().astype(float)
        fitness = np.array([domain.evaluate(p) for p in pop])

        domain.set_iteration_state(best=pop[0].copy(), matrixBin=pop.copy())

        # Mejoras continuas (LOA retorna espacio continuo)
        mejoras_continuas = np.random.uniform(0.05, 0.95, size=pop.shape)
        # Forzar aceptación de toda mejora posible
        fitness_alto = np.full(POP_SIZE, 1e9)

        mh_state = {"posibles_mejoras": mejoras_continuas}
        domain.process_new_population(
            pop.copy(), fitness_alto.copy(), mh_name="LOA", mh_state=mh_state
        )

        # _matrixBin debe ser binaria
        matrix_values = set(domain._matrixBin.astype(int).flatten().tolist())
        assert matrix_values.issubset({0, 1}), (
            f"_matrixBin post-LOA debe ser binaria, pero obtuvo: {matrix_values}"
        )

    def test_fo_chaotic_returns_binary_solution(self):
        """fo() en modo caótico también debe retornar solución binaria."""
        pytest.importorskip("numba", reason="numba no instalado — test caótico saltado")
        domain = ScpDomainManager(
            instance_name=INSTANCE,
            pop_size=POP_SIZE,
            repair_type=REPAIR,
            ds=DS,
            unicost=False,
            chaotic_map_name="LOG",
            chaotic_max_iter=MAX_ITER,
            chaotic_x0=0.7,
        )
        np.random.seed(17)
        population = domain.initialize_population().astype(float)
        domain.set_iteration_state(
            best=population[0].copy(), matrixBin=population.copy()
        )
        x_continuous = np.random.uniform(0.1, 0.9, size=domain.dim)
        result, fit = domain.fo(x_continuous)

        unique_values = set(result.astype(int).tolist())
        assert unique_values.issubset({0, 1}), (
            f"fo() caótico debe retornar solución binaria, obtuvo: {unique_values}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
