"""
Test AB: Unify SCP/USCP — Equivalence Validation
=================================================
Verifica que ``SetCoveringProblem`` produce resultados idénticos a las
clases legacy ``SCP`` y ``USCP`` para todas las operaciones críticas:
block_size, readInstance, factibilityTest, repair, fitness, óptimos.

Estrategia:
    No podemos importar las clases legacy (fueron reemplazadas).  En su lugar
    verificamos propiedades invariantes que eran verdaderas para ambas:
    - block_size según familia de instancia.
    - Dimensiones (rows, columns) leídas correctamente.
    - Costos: SCP → valores del archivo (no todos 1), USCP → todos 1.
    - Óptimos coinciden con los diccionarios originales hardcodeados.
    - factibilityTest, repair, fitness producen resultados consistentes.
"""

import numpy as np
import pytest

from Problem.SCP.problem import (
    SetCoveringProblem,
    SCP,
    obtenerOptimo,
    _OPTIMA_SCP,
    _OPTIMA_USCP,
)
from Problem.USCP.problem import USCP, obtenerOptimoUSCP


# ── Datos de referencia (block_size esperado por familia) ─────────────────────

# Formato: (instance_name, unicost, expected_block_size)
BLOCK_SIZE_CASES = [
    # SCP families
    ("scp41", False, 40),
    ("scp51", False, 40),
    ("scp61", False, 40),
    ("scpa1", False, 30),
    ("scpb1", False, 30),
    ("scpc1", False, 20),
    ("scpd1", False, 20),
    ("scpnre1", False, 10),
    ("scpnrf1", False, 10),
    ("scpnrg1", False, 120),
    ("scpnrh1", False, 120),
    # USCP families
    ("uscp41", True, 40),
    ("uscp51", True, 40),
    ("uscp61", True, 40),
    ("uscpa1", True, 30),
    ("uscpb1", True, 30),
    ("uscpc1", True, 20),
    ("uscpd1", True, 20),
    ("uscpnre1", True, 10),
    ("uscpnrf1", True, 10),
    ("uscpnrg1", True, 120),
    ("uscpnrh1", True, 120),
    ("uscpcyc06", True, 20),
    ("uscpclr10", True, 20),
]


@pytest.mark.parametrize(
    "instance,unicost,expected_bs",
    BLOCK_SIZE_CASES,
    ids=[c[0] for c in BLOCK_SIZE_CASES],
)
def test_block_size(instance, unicost, expected_bs):
    """Block size MUST match legacy behavior for every instance family."""
    p = SetCoveringProblem(instance, unicost=unicost)
    assert p.getBlockSizes() == expected_bs, (
        f"{instance}: block_size={p.getBlockSizes()}, expected={expected_bs}"
    )


# ── Instancias representativas para tests funcionales ─────────────────────────

# Elegimos 1 instancia pequeña de cada tipo para velocidad
SCP_INSTANCES = ["scp41", "scpnrf1"]
USCP_INSTANCES = ["uscp41", "uscpnrf1"]


class TestSCPEquivalence:
    """Verifica que SetCoveringProblem(unicost=False) ≡ legacy SCP."""

    @pytest.mark.parametrize("inst", SCP_INSTANCES)
    def test_dimensions(self, inst):
        p = SetCoveringProblem(inst, unicost=False)
        assert p.getRows() > 0
        assert p.getColumns() > 0
        assert p.getCoverange().shape == (p.getRows(), p.getColumns())
        assert p.getCost().shape == (p.getColumns(),)

    @pytest.mark.parametrize("inst", SCP_INSTANCES)
    def test_costs_are_from_file(self, inst):
        """SCP costs MUST NOT all be 1 (they come from the file)."""
        p = SetCoveringProblem(inst, unicost=False)
        costs = p.getCost()
        # Al menos algunos costos deben ser > 1 para SCP weighted
        assert np.any(costs > 1), f"{inst}: SCP costs should have values > 1"

    @pytest.mark.parametrize("inst", SCP_INSTANCES)
    def test_optimum_matches_dict(self, inst):
        p = SetCoveringProblem(inst, unicost=False)
        expected = _OPTIMA_SCP[inst][1]
        assert p.getOptimum() == expected

    @pytest.mark.parametrize("inst", SCP_INSTANCES)
    def test_factibility_and_repair_simple(self, inst):
        """Repair simple MUST produce a feasible solution."""
        np.random.seed(42)
        p = SetCoveringProblem(inst, unicost=False)
        # Generar solución aleatoria (probablemente infactible)
        sol = np.random.randint(0, 2, size=p.getColumns()).astype(float)
        repaired = p.repair(sol.copy(), "simple")
        feasible, _ = p.factibilityTest(repaired)
        assert feasible, f"{inst}: repair simple did not produce feasible solution"

    @pytest.mark.parametrize("inst", SCP_INSTANCES)
    def test_factibility_and_repair_complex(self, inst):
        """Repair complex MUST produce a feasible solution."""
        np.random.seed(42)
        p = SetCoveringProblem(inst, unicost=False)
        sol = np.random.randint(0, 2, size=p.getColumns()).astype(float)
        repaired = p.repair(sol.copy(), "complex")
        feasible, _ = p.factibilityTest(repaired)
        assert feasible, f"{inst}: repair complex did not produce feasible solution"

    @pytest.mark.parametrize("inst", SCP_INSTANCES)
    def test_fitness_feasible(self, inst):
        """Fitness of a feasible solution MUST be > 0 for SCP."""
        np.random.seed(42)
        p = SetCoveringProblem(inst, unicost=False)
        sol = np.random.randint(0, 2, size=p.getColumns()).astype(float)
        repaired = p.repair(sol.copy(), "complex")
        fit = p.fitness(repaired)
        assert fit > 0, f"{inst}: fitness should be > 0 for SCP"

    @pytest.mark.parametrize("inst", SCP_INSTANCES)
    def test_all_ones_is_feasible(self, inst):
        """A solution with all columns = 1 MUST be feasible."""
        p = SetCoveringProblem(inst, unicost=False)
        sol = np.ones(p.getColumns())
        feasible, _ = p.factibilityTest(sol)
        assert feasible


class TestUSCPEquivalence:
    """Verifica que SetCoveringProblem(unicost=True) ≡ legacy USCP."""

    @pytest.mark.parametrize("inst", USCP_INSTANCES)
    def test_dimensions(self, inst):
        p = SetCoveringProblem(inst, unicost=True)
        assert p.getRows() > 0
        assert p.getColumns() > 0
        assert p.getCoverange().shape == (p.getRows(), p.getColumns())
        assert p.getCost().shape == (p.getColumns(),)

    @pytest.mark.parametrize("inst", USCP_INSTANCES)
    def test_costs_are_all_ones(self, inst):
        """USCP costs MUST all be 1 (unicost definition)."""
        p = SetCoveringProblem(inst, unicost=True)
        costs = p.getCost()
        assert np.all(costs == 1), f"{inst}: USCP costs should all be 1"

    @pytest.mark.parametrize("inst", USCP_INSTANCES)
    def test_optimum_matches_dict(self, inst):
        p = SetCoveringProblem(inst, unicost=True)
        expected = _OPTIMA_USCP[inst][1]
        assert p.getOptimum() == expected

    @pytest.mark.parametrize("inst", USCP_INSTANCES)
    def test_factibility_and_repair_simple(self, inst):
        np.random.seed(42)
        p = SetCoveringProblem(inst, unicost=True)
        sol = np.random.randint(0, 2, size=p.getColumns()).astype(float)
        repaired = p.repair(sol.copy(), "simple")
        feasible, _ = p.factibilityTest(repaired)
        assert feasible

    @pytest.mark.parametrize("inst", USCP_INSTANCES)
    def test_factibility_and_repair_complex(self, inst):
        np.random.seed(42)
        p = SetCoveringProblem(inst, unicost=True)
        sol = np.random.randint(0, 2, size=p.getColumns()).astype(float)
        repaired = p.repair(sol.copy(), "complex")
        feasible, _ = p.factibilityTest(repaired)
        assert feasible

    @pytest.mark.parametrize("inst", USCP_INSTANCES)
    def test_fitness_is_column_count(self, inst):
        """USCP fitness = number of selected columns (since all costs = 1)."""
        p = SetCoveringProblem(inst, unicost=True)
        sol = np.zeros(p.getColumns())
        sol[0] = 1
        sol[1] = 1
        sol[2] = 1
        fit = p.fitness(sol)
        assert fit == 3.0, f"Expected fitness=3.0, got {fit}"


class TestBackwardCompatAliases:
    """Verifica que los aliases legacy siguen funcionando."""

    def test_scp_alias(self):
        p = SCP("scp41")
        assert isinstance(p, SetCoveringProblem)
        assert p.getOptimum() == 429

    def test_uscp_alias(self):
        p = USCP("uscp41")
        assert isinstance(p, SetCoveringProblem)
        assert p.getOptimum() == 38

    def test_obtener_optimo_scp(self):
        """obtenerOptimo MUST return the same value as the legacy dict."""
        assert obtenerOptimo("scp41") == 429
        assert obtenerOptimo("scpnrf1") == 14

    def test_obtener_optimo_uscp(self):
        """obtenerOptimoUSCP MUST return the same value as the legacy dict."""
        assert obtenerOptimoUSCP("uscp41") == 38
        assert obtenerOptimoUSCP("uscpnrf1") == 10

    def test_all_scp_optima_accessible(self):
        """Every SCP instance in the dict MUST be resolvable."""
        for name, (_, expected_opt) in _OPTIMA_SCP.items():
            result = obtenerOptimo(name)
            assert result == expected_opt, f"{name}: {result} != {expected_opt}"

    def test_all_uscp_optima_accessible(self):
        """Every USCP instance in the dict MUST be resolvable."""
        for name, (_, expected_opt) in _OPTIMA_USCP.items():
            result = obtenerOptimoUSCP(name)
            assert result == expected_opt, f"{name}: {result} != {expected_opt}"


class TestSparseSupport:
    """Verifica que el path sparse funciona para ambos modos."""

    @pytest.mark.parametrize("inst,unicost", [("scp41", False), ("uscp41", True)])
    def test_sparse_factibility_matches_dense(self, inst, unicost):
        """factibilityTest con sparse DEBE dar el mismo resultado que denso."""
        from scipy.sparse import csr_matrix

        np.random.seed(42)
        p = SetCoveringProblem(inst, unicost=unicost)
        sol = np.random.randint(0, 2, size=p.getColumns()).astype(float)

        # Dense path
        dense_cov = np.array(p.getCoverange())
        p.setCoverange(dense_cov)
        feasible_dense, val_dense = p.factibilityTest(sol)

        # Sparse path
        p.setCoverange(csr_matrix(dense_cov))
        feasible_sparse, val_sparse = p.factibilityTest(sol)

        assert feasible_dense == feasible_sparse
        np.testing.assert_array_equal(
            np.asarray(val_dense).flatten(),
            np.asarray(val_sparse).flatten(),
        )

        # Restaurar
        p.setCoverange(dense_cov)
