"""
Domain Registry
================
Registro central de dominios de problema disponibles en el framework.

Cada DomainManager concreto se registra aquí al importarse, proporcionando
su factory de ejecución, inserter de instancias BD, y metadata para scripts.

Los consumidores (main.py, BD/sqlite.py, Scripts/) consultan el registry
en vez de hardcodear listas de dominios.

Agregar un dominio nuevo = implementar DomainManager + llamar register().
"""

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class DomainEntry:
    """Entrada de registro para un dominio de problema.

    Attributes:
        domain_type:          Identificador único del dominio ("BEN", "SCP", "USCP").
        config_key:           Clave en experiments_config.json ("ben", "scp", "uscp").
        execute_experiment:   Callable(id, data, datosInstancia, parametros) -> None.
                              Encapsula toda la lógica de setup + universal_solver.
        insert_instances:     Callable(bd) -> None. Inserta instancias en la BD.
                              Puede ser None si el dominio no tiene inserter.
        analysis_meta:        Metadata para Scripts/analisis.py.
                              Keys: sub, inst_key, uses_bin, title_prefix,
                              obtenerArchivos_kwargs.
        instance_dir:         Directorio de instancias (e.g. "./Problem/SCP/Instances/")
                              o None para dominios sin archivos de instancia.
        default_extra_params: Parámetros extra por defecto para poblarDB
                              (e.g. ",repair:complex,cros:0.4;mut:0.50" o "").
    """

    domain_type: str
    config_key: str
    execute_experiment: Callable
    insert_instances: Optional[Callable] = None
    analysis_meta: dict = field(default_factory=dict)
    instance_dir: Optional[str] = None
    default_extra_params: str = ""


# ==================== REGISTRY ====================

_REGISTRY: dict[str, DomainEntry] = {}


def register(entry: DomainEntry) -> None:
    """Registra un dominio en el registry.

    Args:
        entry: DomainEntry con toda la metadata del dominio.

    Raises:
        ValueError: Si ya existe un dominio con el mismo domain_type.
    """
    if entry.domain_type in _REGISTRY:
        raise ValueError(
            f"Dominio '{entry.domain_type}' ya está registrado. "
            f"No se puede registrar dos veces."
        )
    _REGISTRY[entry.domain_type] = entry


def get(domain_type: str) -> DomainEntry:
    """Obtiene la entrada de un dominio por su tipo.

    Args:
        domain_type: Identificador del dominio (e.g. "BEN", "SCP").

    Returns:
        DomainEntry correspondiente.

    Raises:
        KeyError: Si el dominio no está registrado, con mensaje
                  que lista los dominios disponibles.
    """
    if domain_type not in _REGISTRY:
        disponibles = sorted(_REGISTRY.keys()) if _REGISTRY else ["(ninguno)"]
        raise KeyError(
            f"Dominio '{domain_type}' no está registrado. "
            f"Dominios disponibles: {', '.join(disponibles)}"
        )
    return _REGISTRY[domain_type]


def get_all() -> dict[str, DomainEntry]:
    """Retorna una copia del registry completo.

    Returns:
        Diccionario {domain_type: DomainEntry} con todos los dominios registrados.
    """
    return dict(_REGISTRY)
