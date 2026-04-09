"""
USCP (Unicost Set Covering Problem) — Re-export Module
========================================================
Módulo de compatibilidad que re-exporta ``SetCoveringProblem`` con
``unicost=True`` bajo el alias ``USCP``.

Uso::

    from Problem.USCP.problem import USCP
    p = USCP('uscp41')

Toda la lógica vive en ``Problem.SCP.problem.SetCoveringProblem``.
"""

import os

from Problem.SCP.problem import SetCoveringProblem, _OPTIMA_USCP


def USCP(instance):
    """Alias de compatibilidad: equivale a ``SetCoveringProblem(instance, unicost=True)``."""
    return SetCoveringProblem(instance, unicost=True)


def obtenerOptimoUSCP(archivoInstancia):
    """Función libre de compatibilidad para BD/sqlite.py."""
    instancia = os.path.basename(archivoInstancia).replace(".txt", "")
    return _OPTIMA_USCP.get(instancia, [None])[1]


# Alias para que el diccionario sea accesible si alguien lo referenciaba
orden = _OPTIMA_USCP
