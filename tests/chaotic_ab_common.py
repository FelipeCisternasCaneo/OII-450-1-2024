"""
chaotic_ab_common.py — Shared utilities for the Chaotic AB harness
===================================================================

Provides:
  - seed_all(seed)                 : seed numpy + random deterministically
  - validate_comparable_config()   : raise if configs are not AB-comparable
  - parse_csv_bytes(binary)        : decode CSV bytes captured from solvers
  - extract_best_so_far(df)        : best-fitness curve (Series)
  - compute_auc_normalized(curve)  : area under best-so-far curve, normalised to [0,1]
  - ResultCaptor                   : lightweight context-manager that intercepts
                                     BD.insertarIteraciones / BD.insertarResultados
  - ABMetrics (dataclass)          : container for a single solver's observable metrics
  - compare_ab_metrics()           : compare two ABMetrics and return a comparison dict
  - FEASIBILITY_TOLERANCE          : max allowed feasibility ratio difference
  - FITNESS_REL_TOLERANCE          : max allowed relative fitness gap
  - AUC_TOLERANCE                  : max allowed AUC difference
"""

import io
import random
import warnings
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import patch

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Tolerances (can be overridden per-test)
# ──────────────────────────────────────────────────────────────────────────────

FEASIBILITY_TOLERANCE: float = 0.0  # no feasibility mismatch allowed
FITNESS_REL_TOLERANCE: float = 0.10  # ≤10 % relative difference in best fitness
AUC_TOLERANCE: float = 0.10  # ≤10 pp difference in normalised AUC


# ──────────────────────────────────────────────────────────────────────────────
# Seed helper
# ──────────────────────────────────────────────────────────────────────────────


def seed_all(seed: int) -> None:
    """
    Seed both ``numpy.random`` and the stdlib ``random`` module.

    Call this immediately before initialising the population in each leg of
    an AB run so that both solvers start from an identical RNG state.

    Args:
        seed: Any non-negative integer.
    """
    np.random.seed(seed)
    random.seed(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Config validation
# ──────────────────────────────────────────────────────────────────────────────


class ConfigDriftError(ValueError):
    """Raised when two compared configs differ in an AB-critical field."""


def validate_comparable_config(
    legacy_cfg: dict,
    migrated_cfg: dict,
) -> None:
    """
    Ensure that the legacy and migrated runs are executed under the same
    controlled conditions.  Raises ``ConfigDriftError`` (a ``ValueError``
    subclass) if any critical field differs.

    Critical fields checked:
        seed, instance, mh, ds_base, chaotic_map_name, repair_type, pop_size,
        max_iter, unicost

    AB comparisons with FE-only termination are explicitly rejected because the
    legacy solver (``solverSCP_Chaotic``) does not implement FE semantics and
    would silently degrade to 100 iterations.

    Args:
        legacy_cfg:   dict describing the legacy run parameters.
        migrated_cfg: dict describing the migrated run parameters.

    Raises:
        ConfigDriftError: if any critical field mismatches or if max_iter is
                          absent in either config (FE-only runs).
    """
    critical_fields = [
        "seed",
        "instance",
        "mh",
        "ds_base",
        "chaotic_map_name",
        "repair_type",
        "pop_size",
        "max_iter",
        "unicost",
    ]

    for key in critical_fields:
        lv = legacy_cfg.get(key)
        mv = migrated_cfg.get(key)
        if lv != mv:
            raise ConfigDriftError(
                f"AB config drift detected on '{key}': "
                f"legacy={lv!r} vs migrated={mv!r}. "
                "Comparison is invalid."
            )

    # Reject FE-only configs (legacy solver does not support FE termination)
    max_iter_val = legacy_cfg.get("max_iter")
    if max_iter_val is None:
        raise ConfigDriftError(
            "AB comparison requires 'max_iter' to be set in both configs. "
            "FE-only termination is not supported by the legacy solver and "
            "cannot produce a valid AB baseline."
        )


# ──────────────────────────────────────────────────────────────────────────────
# CSV parsing
# ──────────────────────────────────────────────────────────────────────────────


def parse_csv_bytes(binary: bytes) -> pd.DataFrame:
    """
    Decode raw CSV bytes (as stored in BD.insertarIteraciones) into a DataFrame.

    The universal_solver CSV has an extra ``nfe`` column compared to the legacy
    solver.  Both are parsed by selecting the common subset of columns.

    Common columns (always present):
        iter, best_fitness, mean_fitness, std_fitness, time,
        XPL, XPT, DIV, GAP, RDP, ENT, Divj_mean, Divj_min, Divj_max

    Args:
        binary: raw bytes of the CSV file.

    Returns:
        pd.DataFrame with at least an ``iter`` and ``best_fitness`` column.

    Raises:
        ValueError: if the bytes cannot be decoded or parsed as CSV.
    """
    try:
        text = binary.decode("utf-8", errors="replace")
    except Exception as exc:
        raise ValueError(f"Cannot decode CSV bytes: {exc}") from exc

    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        raise ValueError(f"Cannot parse CSV content: {exc}") from exc

    if "iter" not in df.columns or "best_fitness" not in df.columns:
        raise ValueError(
            f"CSV missing required columns 'iter' or 'best_fitness'. "
            f"Found: {list(df.columns)}"
        )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────────────


def extract_best_so_far(df: pd.DataFrame) -> np.ndarray:
    """
    Compute the best-so-far fitness curve from a parsed CSV DataFrame.

    The returned array has one value per row (sorted by ``iter``), where each
    value is the running minimum of ``best_fitness`` up to that iteration.

    Args:
        df: DataFrame from ``parse_csv_bytes``.

    Returns:
        np.ndarray of shape (n_iters,) with monotonically non-increasing
        fitness values.
    """
    df_sorted = df.sort_values("iter").reset_index(drop=True)
    # Use copy=True so we own the array and can modify it in-place
    curve = df_sorted["best_fitness"].to_numpy(dtype=float).copy()
    # Enforce monotone non-increase (running minimum)
    for i in range(1, len(curve)):
        if curve[i] > curve[i - 1]:
            curve[i] = curve[i - 1]
    return curve


def compute_auc_normalized(curve: np.ndarray) -> float:
    """
    Compute the normalised area under the best-so-far curve.

    Normalisation is done by dividing by ``(n_iters * (curve[0] - curve[-1]))``
    where ``curve[0]`` is the initial fitness and ``curve[-1]`` is the final
    fitness.  If the curve is flat (no improvement), the AUC is 1.0 (worst
    case: all area is wasted at the initial fitness level).

    Note: A lower AUC value means the solver converged faster (less area above
    the optimum trajectory).

    Args:
        curve: monotone non-increasing fitness array (from ``extract_best_so_far``).

    Returns:
        float in [0.0, 1.0], where 0.0 = perfect convergence from the start
        and 1.0 = no convergence at all.
    """
    if len(curve) == 0:
        return float("nan")

    n = len(curve)
    f0 = float(curve[0])
    f_last = float(curve[-1])
    span = f0 - f_last

    if span <= 0.0:
        # No improvement — AUC = 1 (entire area is "unimproved")
        return 1.0

    # Area under the curve using trapezoidal rule
    # Shift so that the final value is 0 and initial is span
    shifted = curve - f_last
    # numpy 2.x renamed np.trapz → np.trapezoid; support both for compat
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    raw_auc = float(_trapz(shifted)) / n
    # Normalise to [0, 1] by dividing by the maximum possible rectangle
    normalised = raw_auc / span
    return float(np.clip(normalised, 0.0, 1.0))


def is_feasible(best_solution: np.ndarray, instance) -> bool:
    """
    Check feasibility of a binary solution against an SCP/USCP instance.

    Args:
        best_solution: binary array of shape (dim,).
        instance:      SCP or USCP instance with a ``factibilityTest`` method.

    Returns:
        bool — True if the solution satisfies all constraints.
    """
    flag, _ = instance.factibilityTest(best_solution)
    return bool(flag)


# ──────────────────────────────────────────────────────────────────────────────
# Result captor
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class CapturedResult:
    """Raw data captured from one solver run via BD interception."""

    iterations_csv_bytes: Optional[bytes] = (
        None  # raw bytes passed to insertarIteraciones
    )
    fitness: Optional[float] = None  # best fitness from insertarResultados
    runtime_seconds: Optional[float] = None  # runtime from insertarResultados
    best_solution: Optional[np.ndarray] = None  # best solution from insertarResultados


class ResultCaptor:
    """
    Context manager that intercepts ``BD.insertarIteraciones`` and
    ``BD.insertarResultados`` without touching the real database.

    Usage::

        captor = ResultCaptor()
        with captor:
            solverSCP_Chaotic(...)          # or universal_solver(...)
        result = captor.captured

    The ``captured`` attribute is a ``CapturedResult`` populated after the
    solver exits the context.

    Note: This patches the BD class at the module level used by both the legacy
    solver (``Solver.solverSCP_Chaotic``) and the universal solver
    (``Solver.universal_solver``).  The patch is reentrant-safe because it uses
    ``unittest.mock.patch`` internally.
    """

    def __init__(self):
        self.captured = CapturedResult()
        self._patches = []

    def _make_insertar_iteraciones(self):
        captured = self.captured

        def fake_insertarIteraciones(self_bd, nombre, binary, exp_id):
            captured.iterations_csv_bytes = binary

        return fake_insertarIteraciones

    def _make_insertar_resultados(self):
        captured = self.captured

        def fake_insertarResultados(self_bd, best_fitness, runtime, best, exp_id):
            captured.fitness = float(best_fitness)
            captured.runtime_seconds = float(runtime)
            if isinstance(best, np.ndarray):
                captured.best_solution = best.copy()
            else:
                captured.best_solution = np.array(best)

        return fake_insertarResultados

    def _make_actualizar_experimento(self):
        def fake_actualizar(self_bd, exp_id, estado):
            pass  # No-op: no DB writes during AB tests

        return fake_actualizar

    def __enter__(self):
        # Patch the BD class methods used by both solvers
        bd_module = "BD.sqlite.BD"
        p1 = patch(
            f"{bd_module}.insertarIteraciones",
            self._make_insertar_iteraciones(),
        )
        p2 = patch(
            f"{bd_module}.insertarResultados",
            self._make_insertar_resultados(),
        )
        p3 = patch(
            f"{bd_module}.actualizarExperimento",
            self._make_actualizar_experimento(),
        )
        # Also patch conectar/desconectar so no real DB file is needed
        p4 = patch(f"{bd_module}.conectar", lambda self_bd: None)
        p5 = patch(f"{bd_module}.desconectar", lambda self_bd: None)

        self._patches = [p1, p2, p3, p4, p5]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for p in reversed(self._patches):
            p.stop()
        self._patches = []
        return False  # do not suppress exceptions


# ──────────────────────────────────────────────────────────────────────────────
# ABMetrics dataclass
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ABMetrics:
    """
    Observable metrics for a single solver run.

    Attributes:
        label:           Human-readable name ('legacy' or 'migrated').
        final_fitness:   Best fitness value at the end of the run.
        feasible:        Whether the best solution is feasible.
        best_so_far:     Monotone non-increasing fitness curve (per iter).
        auc_normalized:  Normalised area under the best-so-far curve.
        runtime_seconds: Wall-clock runtime of the solver call.
        df_iters:        Full parsed CSV DataFrame (for deeper inspection).
    """

    label: str = "unknown"
    final_fitness: float = float("inf")
    feasible: bool = False
    best_so_far: np.ndarray = field(default_factory=lambda: np.array([]))
    auc_normalized: float = float("nan")
    runtime_seconds: float = float("nan")
    df_iters: Optional[pd.DataFrame] = None

    @classmethod
    def from_captured(
        cls,
        label: str,
        captured: CapturedResult,
        instance,
    ) -> "ABMetrics":
        """
        Build an ABMetrics from a CapturedResult.

        Args:
            label:    'legacy' or 'migrated'.
            captured: CapturedResult from a ResultCaptor.
            instance: SCP or USCP instance (for feasibility check).

        Returns:
            ABMetrics populated from the captured data.
        """
        metrics = cls(label=label)

        if captured.fitness is not None:
            metrics.final_fitness = captured.fitness

        if captured.runtime_seconds is not None:
            metrics.runtime_seconds = captured.runtime_seconds

        if captured.best_solution is not None:
            metrics.feasible = is_feasible(captured.best_solution, instance)

        if captured.iterations_csv_bytes is not None:
            try:
                df = parse_csv_bytes(captured.iterations_csv_bytes)
                metrics.df_iters = df
                curve = extract_best_so_far(df)
                metrics.best_so_far = curve
                metrics.auc_normalized = compute_auc_normalized(curve)
            except Exception as exc:
                warnings.warn(
                    f"[ABMetrics.from_captured] Could not parse CSV for '{label}': {exc}"
                )

        return metrics


# ──────────────────────────────────────────────────────────────────────────────
# AB comparison
# ──────────────────────────────────────────────────────────────────────────────


def compare_ab_metrics(
    legacy: ABMetrics,
    migrated: ABMetrics,
    fitness_rel_tol: float = FITNESS_REL_TOLERANCE,
    auc_tol: float = AUC_TOLERANCE,
    feasibility_tol: float = FEASIBILITY_TOLERANCE,
) -> dict:
    """
    Compare two ABMetrics objects and return a structured comparison dict.

    Rules (from design spec):
      - Feasibility mismatch → equivalent=False  (hard gate)
      - Relative fitness delta > fitness_rel_tol → equivalent=False
      - AUC delta > auc_tol → equivalent=False  (soft advisory, logged)
      - Runtime ratio is logged but NOT a hard gate

    Returns a dict with keys:
        equivalent          (bool)   — overall equivalence verdict
        feasibility_match   (bool)   — both runs have the same feasibility status
        fitness_delta_rel   (float)  — |Δfitness| / legacy_fitness
        auc_delta           (float)  — |Δauc|
        runtime_ratio       (float)  — migrated_runtime / legacy_runtime  (or nan)
        details             (str)    — human-readable summary
        legacy_fitness      (float)
        migrated_fitness    (float)
        legacy_auc          (float)
        migrated_auc        (float)
        legacy_feasible     (bool)
        migrated_feasible   (bool)
        legacy_runtime_s    (float)
        migrated_runtime_s  (float)
    """
    feasibility_match = legacy.feasible == migrated.feasible
    feasibility_delta = abs(int(legacy.feasible) - int(migrated.feasible))

    # Relative fitness delta (guard against zero denominator)
    if legacy.final_fitness != 0.0 and not np.isnan(legacy.final_fitness):
        fitness_delta_rel = abs(legacy.final_fitness - migrated.final_fitness) / abs(
            legacy.final_fitness
        )
    else:
        fitness_delta_rel = abs(legacy.final_fitness - migrated.final_fitness)

    auc_delta = abs(legacy.auc_normalized - migrated.auc_normalized)

    # Runtime ratio
    if legacy.runtime_seconds and legacy.runtime_seconds > 0:
        runtime_ratio = migrated.runtime_seconds / legacy.runtime_seconds
    else:
        runtime_ratio = float("nan")

    # Equivalence verdict
    reasons = []
    equivalent = True

    if not feasibility_match and feasibility_delta > feasibility_tol:
        equivalent = False
        reasons.append(
            f"Feasibility mismatch: legacy={legacy.feasible}, migrated={migrated.feasible}"
        )

    if fitness_delta_rel > fitness_rel_tol:
        equivalent = False
        reasons.append(
            f"Fitness delta {fitness_delta_rel:.2%} > tolerance {fitness_rel_tol:.2%}"
        )

    if not np.isnan(auc_delta) and auc_delta > auc_tol:
        # AUC is a soft advisory only (logged but not a hard gate per design)
        reasons.append(
            f"[advisory] AUC delta {auc_delta:.4f} > tolerance {auc_tol:.4f} "
            "(soft - does not invalidate equivalence alone)"
        )

    details = ("EQUIVALENT [PASS]" if equivalent else "NOT EQUIVALENT [FAIL]") + (
        ("\n  Reasons:\n    " + "\n    ".join(reasons)) if reasons else ""
    )

    return {
        "equivalent": equivalent,
        "feasibility_match": feasibility_match,
        "fitness_delta_rel": fitness_delta_rel,
        "auc_delta": auc_delta,
        "runtime_ratio": runtime_ratio,
        "details": details,
        "legacy_fitness": legacy.final_fitness,
        "migrated_fitness": migrated.final_fitness,
        "legacy_auc": legacy.auc_normalized,
        "migrated_auc": migrated.auc_normalized,
        "legacy_feasible": legacy.feasible,
        "migrated_feasible": migrated.feasible,
        "legacy_runtime_s": legacy.runtime_seconds,
        "migrated_runtime_s": migrated.runtime_seconds,
    }
