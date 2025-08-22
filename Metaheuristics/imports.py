# Metaheuristics/imports.py

# --- Importaciones de las implementaciones

from .Codes.AOA import iterarAOA
from .Codes.EBWOA import iterarEBWOA
from .Codes.EOO import iterarEOO
from .Codes.FLO import iterarFLO
from .Codes.FOX import iterarFOX
from .Codes.GA import iterarGA
from .Codes.GOA import iterarGOA
from .Codes.GWO import iterarGWO
from .Codes.HBA import iterarHBA
from .Codes.HLOA import iterarHLOAScp, iterarHLOABen
from .Codes.LOA import iterarLOA
from .Codes.NO import iterarNO
from .Codes.PO import IterarPO
from .Codes.POA import iterarPOA
from .Codes.PSA import iterarPSA
from .Codes.PSO import iterarPSO
from .Codes.QSO import iterarQSO
from .Codes.RSA import iterarRSA
from .Codes.SBOA import iterarSBOA
from .Codes.SCA import iterarSCA
from .Codes.SHO import iterarSHO
from .Codes.TDO import iterarTDO
from .Codes.WOA import iterarWOA
from .Codes.WOM import iterarWOM
from .Codes.SSO import iterarSSO
from .Codes.DLO import iterarDLO
from .Codes.DOA import iterarDOA
from .Codes.EHO import iterarEHO
from .Codes.DRA import iterarDRA
from .Codes.SRO import iterarSRO
from .Codes.GOOSE_original import iterarGOOSE_original
from .Codes.GOOSE2 import iterarGOOSE
# --- Diccionario central de metaheurísticas

metaheuristics = {
    "AOA": iterarAOA,
    "EBWOA": iterarEBWOA,
    "EOO": iterarEOO,
    "FLO": iterarFLO,
    "FOX": iterarFOX,
    "GA": iterarGA,
    "GOA": iterarGOA,
    "GWO": iterarGWO,
    "HBA": iterarHBA,
    "HLOA_BEN": iterarHLOABen,
    "HLOA_SCP": iterarHLOAScp,
    "LOA": iterarLOA,
    "NO": iterarNO,
    "POA": iterarPOA,
    "PSA": iterarPSA,
    "PSO": iterarPSO,
    "QSO": iterarQSO,
    "RSA": iterarRSA,
    "SBOA": iterarSBOA,
    "SCA": iterarSCA,
    "SHO": iterarSHO,
    "SSO": iterarSSO,
    "TDO": iterarTDO,
    "WOA": iterarWOA,
    "WOM": iterarWOM,
    "DLO": iterarDLO,
    "DOA": iterarDOA,
    "EHO": iterarEHO,
    "DRA": iterarDRA,
    "SRO": iterarSRO,
    "GOOSE2": iterarGOOSE,
    "GOOSE": iterarGOOSE_original,
}

# --- Mapa de argumentos requeridos (MH_ARG_MAP) ---

MH_ARG_MAP = {
    # Clave MH: (Tupla de argumentos proporcionados por el usuario)

    # A
    'AOA':   ('maxIter', 'iter', 'dim', 'population', 'best', 'lb0', 'ub0'),
    
    # D
    'DOA':  ('maxIter', 'iter', 'dim', 'population', 'best', 'fo', 'lb', 'ub'),
    'DLO':   ('iter', 'maxIter','dim', 'population', 'fitness', 'best', 'lb', 'ub', 'fo'),
    'DRA':   ('maxIter', 'iter', 'dim', 'population', 'fitness', 'best', 'fo', 'lb', 'ub', 'objective_type'),

    # E
    'EBWOA': ('maxIter', 'iter', 'dim', 'population', 'best', 'lb0', 'ub0'),
    'EHO':   ('maxIter', 'iter', 'dim', 'population', 'best', 'lb', 'ub', 'fitness', 'fo'),
    'EOO':   ('maxIter', 'iter', 'population', 'best'),

    # F
    'FLO':   ('iter', 'population', 'fitness', 'fo', 'objective_type', 'lb0', 'ub0'),
    'FOX':   ('maxIter', 'iter', 'dim', 'population', 'best'),

    # G
    'GA':    ('population', 'fitness', 'cross', 'muta'),
    'GOA':   ('maxIter', 'iter', 'dim', 'population', 'best', 'fitness', 'fo', 'objective_type'),
    'GOOSE': ('dim', 'population', 'best'),
    'GOOSE2': ('maxIter', 'iter', 'dim', 'population', 'best'),
    'GWO':   ('maxIter', 'iter', 'dim', 'population', 'fitness', 'objective_type'),

    # H
    'HBA':   ('maxIter', 'iter', 'population', 'best', 'fitness', 'fo', 'objective_type'),
    'HLOA_BEN':('dim', 'population', 'best', 'lb', 'ub'),
    'HLOA_SCP':('dim', 'population', 'best', 'lb0', 'ub0'),

    # L
    'LOA':   ('iter', 'dim', 'population', 'best', 'lb0', 'ub0'),

    # N
    'NO':    ('maxIter', 'iter', 'dim', 'population', 'best'),

    # P
    'POA':   ('iter', 'dim', 'population', 'fitness', 'fo', 'lb0', 'ub0', 'objective_type'),
    'PSA':   ('maxIter', 'iter', 'dim', 'population', 'best'),
    'PSO':   ('maxIter', 'iter', 'dim', 'population', 'best', 'pBest', 'vel', 'ub0'),

    # Q
    'QSO':   ('population', 'best', 'lb', 'ub'),

    # R
    'RSA':   ('maxIter', 'iter', 'dim', 'population', 'best', 'lb0', 'ub0'),

    # S
    'SBOA':  ('maxIter', 'iter', 'dim', 'population', 'fitness', 'best', 'fo'),
    'SCA':   ('maxIter', 'iter', 'population', 'best'),
    'SHO':   ('maxIter', 'iter', 'dim', 'population', 'best', 'fo', 'objective_type'),
    'SSO':   ('maxIter', 'iter', 'dim', 'population', 'fitness', 'best', 'fo', 'lb', 'ub'),
    'SRO':   ('maxIter', 'iter', 'dim', 'population', 'best', 'fo', 'vel', 'userData'),

    # T
    'TDO':   ('maxIter', 'iter', 'dim', 'population', 'fitness', 'fo', 'objective_type'),

    # W
    'WOA':   ('maxIter', 'iter', 'dim', 'population', 'best'),
    'WOM':   ('iter', 'dim', 'population', 'fitness', 'fo', 'lb', 'ub'),
}