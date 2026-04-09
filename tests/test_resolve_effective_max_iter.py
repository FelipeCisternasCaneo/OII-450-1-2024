"""
Unit tests for resolve_effective_max_iter helper
=================================================

Tests cubiertos
---------------
1. Retorna max_iter directamente cuando está disponible.
2. Estima desde max_fe cuando max_iter es None (MH estándar, fe_mult=1).
3. Aplica fe_mult=2 para SBOA, SSO y TDO (doble ronda de evaluación).
4. Nunca retorna 0: mínimo es 1 incluso con presupuesto de FE muy pequeño.
5. max_fe fallback a 10000 cuando max_fe también es None (solo aplica cuando
   max_iter=None, lo cual no debería ocurrir en producción porque
   TerminationCriteria requiere al menos uno, pero la función debe ser robusta).
6. max_iter tiene precedencia sobre max_fe cuando ambos están definidos.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Solver.termination_manager import TerminationCriteria, resolve_effective_max_iter


# ─────────────────────────────────────────────────────────────────────────────
# 1. max_iter disponible → retorna directamente
# ─────────────────────────────────────────────────────────────────────────────


def test_returns_max_iter_when_set():
    tc = TerminationCriteria(max_iter=500)
    result = resolve_effective_max_iter(tc, pop_size=25, mh_name="GWO")
    assert result == 500


def test_max_iter_ignores_max_fe_when_both_set():
    """max_iter tiene precedencia incluso si max_fe está presente."""
    tc = TerminationCriteria(max_iter=300, max_fe=100000)
    result = resolve_effective_max_iter(tc, pop_size=25, mh_name="GWO")
    assert result == 300


# ─────────────────────────────────────────────────────────────────────────────
# 2. Solo max_fe → estimación con fe_mult=1 para MH estándar
# ─────────────────────────────────────────────────────────────────────────────


def test_estimates_from_max_fe_standard_mh():
    """max_fe=10000, pop_size=25 → 10000 // 25 = 400 iteraciones."""
    tc = TerminationCriteria(max_fe=10000)
    result = resolve_effective_max_iter(tc, pop_size=25, mh_name="GWO")
    assert result == 400


def test_estimates_from_max_fe_woa():
    """WOA no está en el conjunto double-FE, usa fe_mult=1."""
    tc = TerminationCriteria(max_fe=5000)
    result = resolve_effective_max_iter(tc, pop_size=10, mh_name="WOA")
    assert result == 500  # 5000 // (10 * 1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. fe_mult=2 para SBOA, SSO, TDO
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mh_name", ["SBOA", "SSO", "TDO"])
def test_double_fe_mult_applied(mh_name):
    """SBOA/SSO/TDO realizan 2 rondas de evaluación por iteración."""
    tc = TerminationCriteria(max_fe=10000)
    result = resolve_effective_max_iter(tc, pop_size=25, mh_name=mh_name)
    # 10000 // (25 * 2) = 200
    assert result == 200


def test_fe_mult_is_one_for_non_double_mhs():
    """Confirma que otras MH no sufren el divisor 2."""
    tc = TerminationCriteria(max_fe=10000)
    sboa_result = resolve_effective_max_iter(tc, pop_size=25, mh_name="SBOA")
    gwo_result = resolve_effective_max_iter(tc, pop_size=25, mh_name="GWO")
    assert sboa_result == gwo_result // 2


# ─────────────────────────────────────────────────────────────────────────────
# 4. Nunca retorna 0: mínimo es 1
# ─────────────────────────────────────────────────────────────────────────────


def test_minimum_one_with_large_pop():
    """Presupuesto muy pequeño vs población grande → siempre >= 1."""
    tc = TerminationCriteria(max_fe=1)
    result = resolve_effective_max_iter(tc, pop_size=1000, mh_name="GWO")
    assert result >= 1


def test_minimum_one_exact_boundary():
    """max_fe=25, pop_size=25 → exactamente 1."""
    tc = TerminationCriteria(max_fe=25)
    result = resolve_effective_max_iter(tc, pop_size=25, mh_name="GWO")
    assert result == 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. No modifica el objeto termination
# ─────────────────────────────────────────────────────────────────────────────


def test_does_not_mutate_termination():
    """La función es pura: no debe alterar el estado de TerminationCriteria."""
    tc = TerminationCriteria(max_fe=10000)
    tc.current_fe = 3000
    tc.current_iter = 5

    resolve_effective_max_iter(tc, pop_size=25, mh_name="GWO")

    assert tc.max_iter is None
    assert tc.max_fe == 10000
    assert tc.current_fe == 3000
    assert tc.current_iter == 5


def test_does_not_mutate_termination_with_max_iter():
    """Igual cuando max_iter está definido."""
    tc = TerminationCriteria(max_iter=200)
    tc.current_iter = 10

    resolve_effective_max_iter(tc, pop_size=25, mh_name="GWO")

    assert tc.max_iter == 200
    assert tc.current_iter == 10
