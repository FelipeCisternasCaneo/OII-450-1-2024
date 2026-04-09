"""
Domain Managers package
========================
Contiene los domain managers concretos (BEN, SCP/USCP) y el registry.

El auto-registro de dominios se dispara llamando a ``ensure_registered()``
o importando este paquete en un contexto que necesite el registry completo
(main.py, Scripts/).  Importar sub-módulos individuales (e.g. scp_domain)
NO dispara el auto-registro de otros dominios.
"""

_registered = False


def ensure_registered():
    """Importa todos los domain modules para disparar su _register().

    Es idempotente: solo ejecuta los imports la primera vez.
    Los consumidores del registry (main.py, poblarDB.py, analisis.py, sqlite.py)
    DEBEN llamar a esta función antes de usar get()/get_all().
    """
    global _registered
    if _registered:
        return
    # Importar cada domain module para que ejecute su _register() al final
    from Solver.domain_managers import ben_domain  # noqa: F401  — registra BEN
    from Solver.domain_managers import scp_domain  # noqa: F401  — registra SCP + USCP

    _registered = True
