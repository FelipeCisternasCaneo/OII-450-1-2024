import numpy as np

from scipy import special as scyesp
from .get_top_k import get_top_k

# === Funciones de Transferencia ===
TRANSFER_FUNCTIONS = {
    "S1": lambda x: 1 / (1 + np.exp(-2 * np.array(x))),
    "S2": lambda x: 1 / (1 + np.exp(-np.array(x))),
    "S3": lambda x: 1 / (1 + np.exp(-np.array(x) / 2)),
    "S4": lambda x: 1 / (1 + np.exp(-np.array(x) / 3)),
    
    "V1": lambda x: np.abs(scyesp.erf(np.sqrt(np.pi) / 2 * np.array(x))),
    "V2": lambda x: np.abs(np.tanh(np.array(x))),
    "V3": lambda x: np.abs(np.array(x) / np.sqrt(1 + np.array(x) ** 2)),
    "V4": lambda x: np.abs((2 / np.pi) * np.arctan((np.pi / 2) * np.array(x))),
    
    "X1": lambda x: 1 / (1 + np.exp(2 * np.array(x))),
    "X2": lambda x: 1 / (1 + np.exp(np.array(x))),
    "X3": lambda x: 1 / (1 + np.exp(np.array(x) / 2)),
    "X4": lambda x: 1 / (1 + np.exp(np.array(x) / 3)),
    
    "Z1": lambda x: np.sqrt(1 - np.exp(-np.abs(x))),
    "Z2": lambda x: np.sqrt(1 - np.exp(-2 * np.abs(x))),
    "Z3": lambda x: np.sqrt(1 - np.exp(-5 * np.abs(x))),
    "Z4": lambda x: np.sqrt(1 - np.exp(-10 * np.abs(x))),
}

# === Funciones de Binarización ===
'''def gvp_binarization(continuous_values):
    """
    Implementa el enfoque de Great Value Priority (GVP) con optimización.
    
    Args:
        continuous_values (np.ndarray): Valores continuos a transformar.
        top_k (float): Porcentaje de elementos que serán binarizados como 1 (depende de la metaheurística, default = 0.06).
    
    Returns:
        np.ndarray: Representación binaria.
    """
    
    top_k = get_top_k(mh = "SBOA")
    
    continuous_values = np.array(continuous_values)
    num_top = int(top_k * len(continuous_values))
    
    # Encuentra el umbral del top_k
    threshold = np.partition(-continuous_values, num_top - 1)[num_top - 1]
    
    # Binariza según el umbral
    binary_values = (continuous_values >= -threshold).astype(int)
    
    return binary_values'''

def gvp_binarization_numba(X, num_activated):
    partition_indices = np.argpartition(-X, num_activated)[:num_activated]
    B = np.zeros_like(X, dtype=np.int32)
    B[partition_indices] = 1
    
    return B

BINARIZATION_FUNCTIONS = {
    "STD": lambda step1, bestBin, indBin: np.where(step1 >= np.random.rand(len(step1)), 1, 0),
    "COM": lambda step1, bestBin, indBin: np.where(step1 >= np.random.rand(len(step1)), 1 - np.array(indBin), 0),
    "ELIT": lambda step1, bestBin, indBin: np.where(step1 >= np.random.rand(len(step1)), np.array(bestBin), 0),
    "PS": lambda step1, bestBin, indBin: np.where(
        step1 >= (1 / 2) * (1 + 1 / 3),
        1,
        np.where((step1 > 1 / 3) & (step1 <= (1 / 2) * (1 + 1 / 3)), indBin, 0),
    ),
    "GVP": lambda step1: gvp_binarization_numba(step1, num_activated=int(0.035 * len(step1))),
}

def aplicarBinarizacion(ind, ds, bestSolutionBin, indBin):
    """
    Aplica la función de transferencia y binarización especificada.

    Args:
        ind (np.ndarray): Individuo continuo a transformar.
        ds (str): Identificador de función de transferencia y binarización (e.g., "S1-STD").
        bestSolutionBin (np.ndarray): Mejor solución binaria conocida.
        indBin (np.ndarray): Representación binaria actual del individuo.

    Returns:
        np.ndarray: Representación binaria transformada del individuo.

    Raises:
        ValueError: Si ocurre un error en la función de transferencia o binarización.
    """
    transferFunction, binarizationFunction = ds.split("-")
    
    ind = np.array(ind)
    bestSolutionBin = np.array(bestSolutionBin)
    indBin = np.array(indBin)
    
    try:
        step1 = TRANSFER_FUNCTIONS[transferFunction](ind)
    
    except Exception as e:
        raise ValueError(f"Error en la función de transferencia '{transferFunction}': {e}")

    try:
        if binarizationFunction == "GVP":
            individuoBin = BINARIZATION_FUNCTIONS[binarizationFunction](step1)
        
        else:
            individuoBin = BINARIZATION_FUNCTIONS[binarizationFunction](step1, bestSolutionBin, indBin)
        
    except Exception as e:
        raise ValueError(f"Error en la función de binarización '{binarizationFunction}': {e}")

    return individuoBin