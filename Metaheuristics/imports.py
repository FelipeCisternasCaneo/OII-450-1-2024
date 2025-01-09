from Metaheuristics.Codes.GA import iterarGA
from Metaheuristics.Codes.GWO import iterarGWO
from Metaheuristics.Codes.PSA import iterarPSA
from Metaheuristics.Codes.SCA import iterarSCA
from Metaheuristics.Codes.WOA import iterarWOA
from Metaheuristics.Codes.PSO import iterarPSO
from Metaheuristics.Codes.FOX import iterarFOX
from Metaheuristics.Codes.EOO import iterarEOO
from Metaheuristics.Codes.RSA import iterarRSA
from Metaheuristics.Codes.GOA import iterarGOA
from Metaheuristics.Codes.HBA import iterarHBA
from Metaheuristics.Codes.TDO import iterarTDO
from Metaheuristics.Codes.SHO import iterarSHO
from Metaheuristics.Codes.SBOA import iterarSBOA
from Metaheuristics.Codes.EBWOA import iterarEBWOA
from Metaheuristics.Codes.EHO import iterarEHO
from Metaheuristics.Codes.FLO import iterarFLO
from Metaheuristics.Codes.HLOA import iterarHLOAScp, iterarHLOABen
from Metaheuristics.Codes.LOA import iterarLOA
from Metaheuristics.Codes.NO import iterarNO
from Metaheuristics.Codes.PO import IterarPO
from Metaheuristics.Codes.POA import iterarPOA
from Metaheuristics.Codes.QSO import iterarQSO
from Metaheuristics.Codes.WOM import iterarWOM
from Metaheuristics.Codes.AOA import iterarAOA

# Diccionario central de metaheurísticas
metaheuristics = {
    "GA": iterarGA,
    "GWO": iterarGWO,
    "PSA": iterarPSA,
    "SCA": iterarSCA,
    "WOA": iterarWOA,
    "PSO": iterarPSO,
    "FOX": iterarFOX,
    "EOO": iterarEOO,
    "RSA": iterarRSA,
    "GOA": iterarGOA,
    "HBA": iterarHBA,
    "TDO": iterarTDO,
    "SHO": iterarSHO,
    "SBOA": iterarSBOA,
    "EBWOA": iterarEBWOA,
    "EHO": iterarEHO,
    "FLO": iterarFLO,
    "HLOA": iterarHLOAScp,
    "HLOA": iterarHLOABen,
    "LOA": iterarLOA,
    "NO": iterarNO,
    "PO": IterarPO,
    "POA": iterarPOA,
    "QSO": iterarQSO,
    "WOM": iterarWOM,
    "AOA": iterarAOA
}