r"""
test_ab_chaotic_migration.py — AB harness: migrated chaotic solver validation
==============================================================================

NOTE (post-cleanup): ``Solver/solverSCP_Chaotic.py`` was deleted as part of
the legacy cleanup (Batch 5 of migrate-chaotic-maps-solver-to-universal).
The live AB comparison tests (``TestABChaoticMigration``) are preserved but
marked as ``pytest.skip`` because the legacy baseline no longer exists.

The following test classes remain fully functional and continue to run:
  - ``TestConfigValidation``  — harness config-drift and FE-only rejection
  - ``TestMetricHelpers``     — metric computation helpers
  - ``TestResultCaptor``      — BD interception smoke test

Rejects (config validation still in effect):
  - FE-only termination (no max_iter)
  - Config drift (any difference in seed/instance/MH/DS/map/repair/pop/iter)

Run with::

    .\.venv\Scripts\python.exe -m pytest tests/test_ab_chaotic_migration.py -v

To run a specific case::

    .\.venv\Scripts\python.exe -m pytest tests/test_ab_chaotic_migration.py::TestABChaoticMigration::test_ab_gwo_log_scp41 -v -s
"""

import sys
import os

import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.chaotic_ab_common import (
    validate_comparable_config,
    ConfigDriftError,
    ResultCaptor,
    ABMetrics,
    compare_ab_metrics,
    parse_csv_bytes,
    extract_best_so_far,
    compute_auc_normalized,
)

# ──────────────────────────────────────────────────────────────────────────────
# Config drift / FE-only rejection tests (Task 2.3)
# ──────────────────────────────────────────────────────────────────────────────


class TestConfigValidation:
    """Harness rejects invalid or non-comparable AB configs."""

    _BASE = dict(
        seed=42,
        mh="GWO",
        max_iter=20,
        pop_size=10,
        instance="scp41",
        ds_base="V1-STD",
        chaotic_map_name="LOG",
        repair_type="complex",
        unicost=False,
    )

    def test_identical_configs_pass(self):
        validate_comparable_config(self._BASE.copy(), self._BASE.copy())

    def test_seed_drift_raises(self):
        cfg_a = self._BASE.copy()
        cfg_b = self._BASE.copy()
        cfg_b["seed"] = 99
        with pytest.raises(ConfigDriftError, match="seed"):
            validate_comparable_config(cfg_a, cfg_b)

    def test_max_iter_drift_raises(self):
        cfg_a = self._BASE.copy()
        cfg_b = self._BASE.copy()
        cfg_b["max_iter"] = 100
        with pytest.raises(ConfigDriftError, match="max_iter"):
            validate_comparable_config(cfg_a, cfg_b)

    def test_instance_drift_raises(self):
        cfg_a = self._BASE.copy()
        cfg_b = self._BASE.copy()
        cfg_b["instance"] = "scp42"
        with pytest.raises(ConfigDriftError, match="instance"):
            validate_comparable_config(cfg_a, cfg_b)

    def test_mh_drift_raises(self):
        cfg_a = self._BASE.copy()
        cfg_b = self._BASE.copy()
        cfg_b["mh"] = "WOA"
        with pytest.raises(ConfigDriftError, match="mh"):
            validate_comparable_config(cfg_a, cfg_b)

    def test_chaotic_map_drift_raises(self):
        cfg_a = self._BASE.copy()
        cfg_b = self._BASE.copy()
        cfg_b["chaotic_map_name"] = "SINE"
        with pytest.raises(ConfigDriftError, match="chaotic_map_name"):
            validate_comparable_config(cfg_a, cfg_b)

    def test_ds_base_drift_raises(self):
        cfg_a = self._BASE.copy()
        cfg_b = self._BASE.copy()
        cfg_b["ds_base"] = "V3-ELIT"
        with pytest.raises(ConfigDriftError, match="ds_base"):
            validate_comparable_config(cfg_a, cfg_b)

    def test_repair_drift_raises(self):
        cfg_a = self._BASE.copy()
        cfg_b = self._BASE.copy()
        cfg_b["repair_type"] = "simple"
        with pytest.raises(ConfigDriftError, match="repair_type"):
            validate_comparable_config(cfg_a, cfg_b)

    def test_fe_only_raises(self):
        """FE-only configs (max_iter=None) must be rejected."""
        cfg_a = self._BASE.copy()
        cfg_a["max_iter"] = None
        cfg_b = self._BASE.copy()
        cfg_b["max_iter"] = None
        with pytest.raises(ConfigDriftError, match="max_iter|FE"):
            validate_comparable_config(cfg_a, cfg_b)

    def test_pop_size_drift_raises(self):
        cfg_a = self._BASE.copy()
        cfg_b = self._BASE.copy()
        cfg_b["pop_size"] = 20
        with pytest.raises(ConfigDriftError, match="pop_size"):
            validate_comparable_config(cfg_a, cfg_b)

    def test_unicost_drift_raises(self):
        cfg_a = self._BASE.copy()
        cfg_b = self._BASE.copy()
        cfg_b["unicost"] = True
        with pytest.raises(ConfigDriftError, match="unicost"):
            validate_comparable_config(cfg_a, cfg_b)


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers unit tests (Task 2.3 — helper correctness)
# ──────────────────────────────────────────────────────────────────────────────


class TestMetricHelpers:
    """Validate the metric computation helpers in isolation."""

    def test_extract_best_so_far_monotone(self):
        """best_so_far must be monotonically non-increasing."""
        raw = np.array([10.0, 12.0, 9.0, 8.0, 11.0, 7.0])
        curve = extract_best_so_far(
            __import__("pandas").DataFrame(
                {"iter": range(len(raw)), "best_fitness": raw}
            )
        )
        assert len(curve) == len(raw)
        diffs = np.diff(curve)
        assert (diffs <= 0).all(), "Curve must be monotone non-increasing"

    def test_auc_flat_curve_returns_one(self):
        """A flat curve (no improvement) should return AUC=1."""
        curve = np.array([100.0, 100.0, 100.0, 100.0])
        assert compute_auc_normalized(curve) == 1.0

    def test_auc_instant_convergence_returns_zero(self):
        """A curve that drops immediately should return AUC≈0."""
        # curve[0] = 100, all others = 0 → area ≈ 0
        curve = np.array([100.0] + [0.0] * 99)
        auc = compute_auc_normalized(curve)
        # With trapezoidal rule the first step contributes ~0.5/100 = ~0.005
        assert auc < 0.02, f"Expected near-0 AUC, got {auc}"

    def test_auc_monotone_decrease_in_range(self):
        """Linear decrease from 10 to 0 should give AUC=0.5."""
        n = 101
        curve = np.linspace(10.0, 0.0, n)
        auc = compute_auc_normalized(curve)
        assert abs(auc - 0.5) < 0.02, f"Expected ~0.5, got {auc}"

    def test_parse_csv_bytes_basic(self):
        csv_content = (
            "iter,best_fitness,mean_fitness,std_fitness,time,"
            "XPL,XPT,DIV,GAP,RDP,ENT,Divj_mean,Divj_min,Divj_max\n"
            "0,100.000000,120.0,5.0,0.01,0.5,0.5,1.0,0.1,0.01,0.8,0.3,0.1,0.5\n"
            "1,95.000000,115.0,4.5,0.02,0.4,0.6,0.9,0.09,0.009,0.75,0.28,0.09,0.48\n"
        )
        df = parse_csv_bytes(csv_content.encode("utf-8"))
        assert "iter" in df.columns
        assert "best_fitness" in df.columns
        assert len(df) == 2

    def test_parse_csv_bytes_universal_solver_format(self):
        """universal_solver CSV has an extra 'nfe' column — must still parse."""
        csv_content = (
            "iter,nfe,best_fitness,mean_fitness,std_fitness,"
            "time,XPL,XPT,DIV,GAP,RDP,ENT,Divj_mean,Divj_min,Divj_max\n"
            "0,10,100.000000,120.0,5.0,0.01,0.5,0.5,1.0,0.1,0.01,0.8,0.3,0.1,0.5\n"
            "1,20,90.000000,110.0,4.0,0.02,0.4,0.6,0.9,0.09,0.009,0.75,0.28,0.09,0.48\n"
        )
        df = parse_csv_bytes(csv_content.encode("utf-8"))
        assert "best_fitness" in df.columns
        assert len(df) == 2

    def test_compare_ab_metrics_identical_is_equivalent(self):
        """Two identical metrics must be declared equivalent."""
        from tests.chaotic_ab_common import ABMetrics, compare_ab_metrics

        m = ABMetrics(
            label="test",
            final_fitness=100.0,
            feasible=True,
            best_so_far=np.array([100.0, 95.0, 90.0]),
            auc_normalized=0.8,
            runtime_seconds=1.0,
        )
        result = compare_ab_metrics(m, m)
        assert result["equivalent"] is True

    def test_compare_ab_metrics_feasibility_mismatch_fails(self):
        """Feasibility mismatch must set equivalent=False."""
        from tests.chaotic_ab_common import ABMetrics, compare_ab_metrics

        m_a = ABMetrics(
            label="a", final_fitness=100.0, feasible=True, runtime_seconds=1.0
        )
        m_b = ABMetrics(
            label="b", final_fitness=100.0, feasible=False, runtime_seconds=1.0
        )
        result = compare_ab_metrics(m_a, m_b)
        assert result["equivalent"] is False
        assert result["feasibility_match"] is False

    def test_compare_ab_metrics_large_fitness_delta_fails(self):
        """A relative fitness delta > 10% must fail equivalence."""
        from tests.chaotic_ab_common import ABMetrics, compare_ab_metrics

        m_a = ABMetrics(
            label="a", final_fitness=100.0, feasible=True, runtime_seconds=1.0
        )
        m_b = ABMetrics(
            label="b", final_fitness=120.0, feasible=True, runtime_seconds=1.0
        )
        result = compare_ab_metrics(m_a, m_b)
        assert result["equivalent"] is False
        assert result["fitness_delta_rel"] > 0.10


# ──────────────────────────────────────────────────────────────────────────────
# Live AB runs — legacy vs migrated
# (SKIPPED: legacy solver deleted in Batch 5 cleanup)
# ──────────────────────────────────────────────────────────────────────────────

# Tolerance for the live AB runs (can be wider than production thresholds
# because a short run of only 20–30 iterations may not fully converge)
_LIVE_FITNESS_TOL = 0.25  # 25% tolerance for short live runs
_LIVE_AUC_TOL = 0.30  # 30 pp — advisory only

_SKIP_AB_REASON = (
    "Legacy solver (Solver/solverSCP_Chaotic.py) was deleted in Batch 5 cleanup. "
    "AB comparison is no longer possible — migration was validated before deletion."
)


@pytest.mark.skip(reason=_SKIP_AB_REASON)
class TestABChaoticMigration:
    """
    Paired AB runs: legacy solverSCP_Chaotic vs migrated ScpDomainManager+universal_solver.

    SKIPPED: The legacy solver was deleted after the migration was fully validated
    (Batches 1–4 of migrate-chaotic-maps-solver-to-universal). These test stubs
    are retained for documentation purposes only.
    """

    def test_ab_gwo_log_scp41(self):
        """GWO + LOG map on scp41 — baseline AB case."""

    def test_ab_gwo_sine_scp41(self):
        """GWO + SINE map on scp41 — different chaotic map."""

    def test_ab_gwo_tent_scp41(self):
        """GWO + TENT map on scp41 — different chaotic map."""

    def test_ab_woa_log_scp41(self):
        """WOA + LOG map on scp41 — different metaheuristic."""

    def test_ab_gwo_log_scp41_seed2(self):
        """GWO + LOG, seed=99 — second seed for reproducibility check."""

    def test_ab_gwo_log_scp41_seed3(self):
        """GWO + LOG, seed=123 — third seed."""

    def test_ab_gwo_log_uscp41(self):
        """GWO + LOG map on uscp41 with unicost=True — baseline USCP AB case."""

    def test_ab_gwo_sine_uscp41(self):
        """GWO + SINE map on uscp41 with unicost=True — second USCP chaotic map."""


# ──────────────────────────────────────────────────────────────────────────────
# Result captor smoke test
# ──────────────────────────────────────────────────────────────────────────────


class TestResultCaptor:
    """Verify that the ResultCaptor intercepts BD calls without touching the DB."""

    def test_captor_intercepts_iteraciones(self):
        """insertarIteraciones must be intercepted and stored."""
        from BD.sqlite import BD

        captured_data = {}

        class FakeCaptor(ResultCaptor):
            pass

        captor = FakeCaptor()
        with captor:
            bd = BD()
            bd.insertarIteraciones("test_run", b"col1,col2\n1,2\n", 12345)
            bd.insertarResultados(429.0, 1.23, np.ones(5), 12345)

        assert captor.captured.iterations_csv_bytes == b"col1,col2\n1,2\n"
        assert captor.captured.fitness == 429.0
        assert captor.captured.runtime_seconds == 1.23
        assert captor.captured.best_solution is not None

    def test_captor_does_not_touch_real_db(self, tmp_path, monkeypatch):
        """No file should be created at the real DB path during a captured run."""
        fake_db_path = str(tmp_path / "test_sentinel.db")
        monkeypatch.setenv("OII_DB_PATH", fake_db_path)

        from BD.sqlite import BD

        captor = ResultCaptor()
        with captor:
            bd = BD()
            bd.insertarIteraciones("x", b"a,b\n1,2\n", 1)

        # DB file should NOT have been created because conectar was patched
        assert not os.path.exists(fake_db_path), (
            "ResultCaptor should not create the real DB file"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point for direct script execution
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print(
        "NOTE: Legacy AB harness is no longer available — "
        "Solver/solverSCP_Chaotic.py was deleted in Batch 5 cleanup.\n"
        "Run the full test suite instead:\n"
        "  .\\venv\\Scripts\\python.exe -m pytest tests/test_ab_chaotic_migration.py -v\n"
        "  .\\venv\\Scripts\\python.exe -m pytest tests/test_scp_domain_chaotic.py -v\n"
    )
    sys.exit(0)
