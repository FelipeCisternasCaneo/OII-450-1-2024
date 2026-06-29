"""
USCP (Unicost Set Covering Problem) — Re-export Module
========================================================
Módulo de compatibilidad que re-exporta ``SetCoveringProblem`` con
``unicost=True`` bajo el alias ``USCP``.

Uso::

    from problem.USCP.problem import USCP
    p = USCP('uscp41')

Toda la lógica vive en ``Problem.SCP.problem.SetCoveringProblem``.
"""

import os

from problem.SCP.problem import SetCoveringProblem
from problem.optima import OPTIMA_USCP


def USCP(instance):
    """Alias de compatibilidad: equivale a ``SetCoveringProblem(instance, unicost=True)``."""
    return SetCoveringProblem(instance, unicost=True)


def obtenerOptimoUSCP(archivoInstancia):
    """Función libre de compatibilidad para BD/sqlite.py."""
    instancia = os.path.basename(archivoInstancia).replace(".txt", "")
    return OPTIMA_USCP.get(instancia, [None])[1]


# Alias para que el diccionario sea accesible si alguien lo referenciaba
orden = OPTIMA_USCP
