"""
Test: Domain Registry
=====================
Valida que el Domain Registry funciona correctamente según la spec:
- Registro exitoso y funciones puras (register, get, get_all)
- Registro duplicado lanza ValueError
- Consulta de dominio inexistente lanza KeyError descriptivo
- DomainEntry con metadata completa
- Dominio sin inserter no produce error
- get_all retorna copia defensiva
- Auto-registro de BEN/SCP/USCP (requiere dependencias completas)
- PROBLEMS de analisis.py se construye correctamente desde el registry
"""

import importlib.util
import os
import sys

import pytest

# ── Import registry directamente (sin pasar por __init__.py) ─────────────────
# Esto evita que __init__.py dispare auto-registro de ben_domain/scp_domain,
# que a su vez importan dependencias pesadas.

_REGISTRY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "Solver", "domain_managers", "registry.py"
)
_spec = importlib.util.spec_from_file_location(
    "Solver.domain_managers.registry", os.path.abspath(_REGISTRY_PATH)
)
_registry_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_registry_mod)

# Extraer símbolos del módulo
DomainEntry = _registry_mod.DomainEntry
register = _registry_mod.register
get = _registry_mod.get
get_all = _registry_mod.get_all
_REGISTRY = _registry_mod._REGISTRY


# ── Helper: ¿están disponibles las dependencias pesadas? ─────────────────────


def _can_import_domain_managers():
    """Verifica si el auto-registro de dominios funciona."""
    try:
        from Solver.domain_managers import ensure_registered

        ensure_registered()
        return True
    except ImportError:
        return False


_HAS_DOMAINS = _can_import_domain_managers()
requires_domains = pytest.mark.skipif(
    not _HAS_DOMAINS,
    reason="Requiere dependencias completas para auto-registro de dominios",
)


# ── Tests de funciones puras (sin dependencias pesadas) ──────────────────────


class TestDomainEntryCreation:
    """DomainEntry se crea correctamente con valores por defecto."""

    def test_minimal_entry(self):
        entry = DomainEntry(
            domain_type="MOCK",
            config_key="mock",
            execute_experiment=lambda *a, **kw: None,
        )
        assert entry.domain_type == "MOCK"
        assert entry.config_key == "mock"
        assert entry.insert_instances is None
        assert entry.analysis_meta == {}
        assert entry.instance_dir is None
        assert entry.default_extra_params == ""

    def test_full_entry(self):
        inserter = lambda bd: None
        entry = DomainEntry(
            domain_type="FULL",
            config_key="full",
            execute_experiment=lambda *a, **kw: None,
            insert_instances=inserter,
            analysis_meta={"sub": "FULL", "uses_bin": False},
            instance_dir="./instances/",
            default_extra_params=",extra:1",
        )
        assert entry.insert_instances is inserter
        assert entry.analysis_meta["sub"] == "FULL"
        assert entry.instance_dir == "./instances/"
        assert entry.default_extra_params == ",extra:1"

    def test_entry_is_frozen(self):
        """DomainEntry es frozen dataclass — no se puede mutar."""
        entry = DomainEntry(
            domain_type="FROZEN",
            config_key="frozen",
            execute_experiment=lambda *a, **kw: None,
        )
        with pytest.raises(AttributeError):
            entry.domain_type = "CHANGED"

    def test_none_inserter_is_valid(self):
        """Un DomainEntry con insert_instances=None DEBE ser válido."""
        entry = DomainEntry(
            domain_type="NO_INSERTER",
            config_key="noi",
            execute_experiment=lambda *a, **kw: None,
            insert_instances=None,
        )
        assert entry.insert_instances is None


class TestRegistryFunctions:
    """Tests de register/get/get_all con entries de prueba aisladas."""

    @pytest.fixture(autouse=True)
    def _clean_registry(self):
        """Guarda y restaura el registry para aislar tests."""
        original = dict(_REGISTRY)
        yield
        _REGISTRY.clear()
        _REGISTRY.update(original)

    def _make_entry(self, dtype: str) -> DomainEntry:
        return DomainEntry(
            domain_type=dtype,
            config_key=dtype.lower(),
            execute_experiment=lambda *a, **kw: None,
        )

    def test_register_and_get(self):
        """GIVEN un entry WHEN register + get THEN retorna el mismo entry."""
        entry = self._make_entry("TEST_A")
        register(entry)
        retrieved = get("TEST_A")
        assert retrieved is entry

    def test_register_duplicate_raises_value_error(self):
        """GIVEN entry registrado WHEN register duplicado THEN ValueError."""
        entry = self._make_entry("TEST_DUP")
        register(entry)
        duplicate = self._make_entry("TEST_DUP")
        with pytest.raises(ValueError, match="ya está registrado"):
            register(duplicate)

    def test_get_nonexistent_raises_key_error(self):
        """GIVEN entry no registrado WHEN get THEN KeyError con mensaje descriptivo."""
        with pytest.raises(KeyError, match="no está registrado"):
            get("NONEXISTENT_DOMAIN")

    def test_key_error_lists_available_domains(self):
        """El KeyError DEBE listar los dominios disponibles."""
        entry = self._make_entry("LISTED")
        register(entry)
        with pytest.raises(KeyError, match="LISTED"):
            get("MISSING")

    def test_get_all_returns_dict(self):
        """get_all() retorna un diccionario."""
        result = get_all()
        assert isinstance(result, dict)

    def test_get_all_returns_copy(self):
        """get_all() DEBE retornar una copia, no la referencia interna."""
        entry = self._make_entry("COPY_TEST")
        register(entry)
        copy = get_all()
        original_len = len(copy)
        copy["FAKE"] = None
        assert len(get_all()) == original_len, (
            "Modificar get_all() NO debe afectar _REGISTRY"
        )

    def test_get_all_contains_registered(self):
        """get_all() DEBE contener todas las entries registradas."""
        e1 = self._make_entry("GA_1")
        e2 = self._make_entry("GA_2")
        register(e1)
        register(e2)
        all_domains = get_all()
        assert "GA_1" in all_domains
        assert "GA_2" in all_domains

    def test_get_empty_registry_key_error_shows_ninguno(self):
        """Si el registry está vacío, KeyError muestra '(ninguno)'."""
        _REGISTRY.clear()
        with pytest.raises(KeyError, match="ninguno"):
            get("ANYTHING")


# ── Tests de auto-registro (requieren dependencias completas) ─────────────────
# Estos tests usan el registry REAL (importado normalmente, no via importlib.util)
# porque ensure_registered() registra dominios en el módulo Solver.domain_managers.registry
# del sistema de imports estándar de Python.


@requires_domains
class TestAutoRegistration:
    """Al importar Solver.domain_managers se registran BEN, SCP, USCP."""

    def test_all_domains_registered(self):
        from Solver.domain_managers.registry import get_all as real_get_all

        all_domains = real_get_all()
        assert "BEN" in all_domains, "BEN debería estar registrado"
        assert "SCP" in all_domains, "SCP debería estar registrado"
        assert "USCP" in all_domains, "USCP debería estar registrado"

    def test_get_ben(self):
        from Solver.domain_managers.registry import get as real_get

        entry = real_get("BEN")
        assert entry.domain_type == "BEN"
        assert entry.config_key == "ben"

    def test_get_scp(self):
        from Solver.domain_managers.registry import get as real_get

        entry = real_get("SCP")
        assert entry.domain_type == "SCP"
        assert entry.config_key == "scp"

    def test_get_uscp(self):
        from Solver.domain_managers.registry import get as real_get

        entry = real_get("USCP")
        assert entry.domain_type == "USCP"
        assert entry.config_key == "uscp"

    def test_exactly_three_domains_minimum(self):
        """Al menos BEN, SCP, USCP deben estar registrados."""
        from Solver.domain_managers.registry import get_all as real_get_all

        all_domains = real_get_all()
        expected = {"BEN", "SCP", "USCP"}
        assert expected.issubset(set(all_domains.keys()))


@requires_domains
class TestAnalysisMetadata:
    """Cada dominio registrado DEBE tener analysis_meta con las keys requeridas."""

    REQUIRED_KEYS = {
        "sub",
        "inst_key",
        "uses_bin",
        "title_prefix",
        "obtenerArchivos_kwargs",
    }

    @pytest.fixture(params=["BEN", "SCP", "USCP"])
    def entry(self, request):
        from Solver.domain_managers.registry import get as real_get

        return real_get(request.param)

    def test_analysis_meta_has_required_keys(self, entry):
        meta = entry.analysis_meta
        missing = self.REQUIRED_KEYS - set(meta.keys())
        assert not missing, (
            f"{entry.domain_type}: faltan keys en analysis_meta: {missing}"
        )

    def test_analysis_meta_sub_matches_domain_type(self, entry):
        assert entry.analysis_meta["sub"] == entry.domain_type

    def test_analysis_meta_inst_key_matches_domain_type(self, entry):
        assert entry.analysis_meta["inst_key"] == entry.domain_type


@requires_domains
class TestBenSpecificMeta:
    """BEN DEBE tener uses_bin=False y obtenerArchivos_kwargs sin binarización."""

    def test_ben_no_binarization(self):
        from Solver.domain_managers.registry import get as real_get

        entry = real_get("BEN")
        assert entry.analysis_meta["uses_bin"] is False
        assert entry.analysis_meta["obtenerArchivos_kwargs"] == {
            "incluir_binarizacion": False
        }


@requires_domains
class TestScpBinarization:
    """SCP y USCP DEBEN tener uses_bin=True."""

    @pytest.fixture(params=["SCP", "USCP"])
    def entry(self, request):
        from Solver.domain_managers.registry import get as real_get

        return real_get(request.param)

    def test_uses_binarization(self, entry):
        assert entry.analysis_meta["uses_bin"] is True


@requires_domains
class TestExecuteExperiment:
    """Cada dominio registrado DEBE tener un execute_experiment callable."""

    @pytest.fixture(params=["BEN", "SCP", "USCP"])
    def entry(self, request):
        from Solver.domain_managers.registry import get as real_get

        return real_get(request.param)

    def test_execute_experiment_is_callable(self, entry):
        assert callable(entry.execute_experiment), (
            f"{entry.domain_type}: execute_experiment no es callable"
        )

    def test_insert_instances_is_callable(self, entry):
        """Los 3 dominios existentes SÍ tienen inserter."""
        assert callable(entry.insert_instances), (
            f"{entry.domain_type}: insert_instances debería ser callable"
        )


@requires_domains
class TestAnalisisProblemsFromRegistry:
    """analisis.py DEBE construir PROBLEMS consultando el registry."""

    def test_problems_matches_registry(self):
        from Solver.domain_managers.registry import get_all as real_get_all

        problems = {
            dtype: entry.analysis_meta for dtype, entry in real_get_all().items()
        }
        assert set(problems.keys()) == set(real_get_all().keys())

    def test_problems_ben_meta_correct(self):
        from Solver.domain_managers.registry import get_all as real_get_all

        problems = {
            dtype: entry.analysis_meta for dtype, entry in real_get_all().items()
        }
        ben = problems["BEN"]
        assert ben["sub"] == "BEN"
        assert ben["inst_key"] == "BEN"
        assert ben["uses_bin"] is False
        assert ben["title_prefix"] == ""
        assert ben["obtenerArchivos_kwargs"] == {"incluir_binarizacion": False}

    def test_problems_scp_meta_correct(self):
        from Solver.domain_managers.registry import get_all as real_get_all

        problems = {
            dtype: entry.analysis_meta for dtype, entry in real_get_all().items()
        }
        scp = problems["SCP"]
        assert scp["sub"] == "SCP"
        assert scp["inst_key"] == "SCP"
        assert scp["uses_bin"] is True
        assert scp["title_prefix"] == "scp"
        assert scp["obtenerArchivos_kwargs"] == {}

    def test_problems_uscp_meta_correct(self):
        from Solver.domain_managers.registry import get_all as real_get_all

        problems = {
            dtype: entry.analysis_meta for dtype, entry in real_get_all().items()
        }
        uscp = problems["USCP"]
        assert uscp["sub"] == "USCP"
        assert uscp["inst_key"] == "USCP"
        assert uscp["uses_bin"] is True
        assert uscp["title_prefix"] == "uscp"
        assert uscp["obtenerArchivos_kwargs"] == {}
