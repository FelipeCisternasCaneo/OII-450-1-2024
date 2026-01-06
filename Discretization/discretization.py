import numpy as np
from scipy import special as scyesp

# === Funciones de transferencia ===

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

BINARIZATION_FUNCTIONS = {
    "STD": lambda step1, bestBin, indBin, chaotic_vals: np.where(
        step1 >= np.random.rand(len(step1)) if chaotic_vals is None else chaotic_vals[:len(step1)], 
        1, 0
    ),
    "COM": lambda step1, bestBin, indBin, chaotic_vals: np.where(
        step1 >= np.random.rand(len(step1)) if chaotic_vals is None else chaotic_vals[:len(step1)], 
        1 - np.array(indBin), 0
    ),
    "ELIT": lambda step1, bestBin, indBin, chaotic_vals: np.where(
        step1 >= np.random.rand(len(step1)) if chaotic_vals is None else chaotic_vals[:len(step1)], 
        np.array(bestBin), 0
    ),
    "PS": lambda step1, bestBin, indBin, chaotic_vals: np.where(
        step1 >= (1 / 2) * (1 + 1 / 3),
        1,
        np.where((step1 > 1 / 3) & (step1 <= (1 / 2) * (1 + 1 / 3)), indBin, 0),
    ),
}

def aplicarBinarizacion(ind, ds, bestSolutionBin, indBin, chaotic_map=None, chaotic_index=0):
    """
    Aplica la función de transferencia y binarización especificada.
    
    NUEVO: Soporta mapas caóticos para reemplazar valores aleatorios.

    Args:
        ind (np.ndarray): Individuo continuo a transformar.
        ds (str): Identificador de función de transferencia y binarización (e.g., "S1-STD").
        bestSolutionBin (np.ndarray): Mejor solución binaria conocida.
        indBin (np.ndarray): Representación binaria actual del individuo.
        chaotic_map (np.ndarray, optional): Secuencia de valores caóticos pregenerada.
        chaotic_index (int, optional): Índice de inicio en el mapa caótico.

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
        # Extraer valores caóticos si están disponibles
        chaotic_vals = None
        if chaotic_map is not None and len(chaotic_map) > 0:
            dim = len(ind)
            start_idx = chaotic_index % len(chaotic_map)
            end_idx = (start_idx + dim) % len(chaotic_map)
            
            if end_idx > start_idx:
                chaotic_vals = chaotic_map[start_idx:end_idx]
            else:
                # Wrap around
                chaotic_vals = np.concatenate([
                    chaotic_map[start_idx:],
                    chaotic_map[:end_idx]
                ])
        
        individuoBin = BINARIZATION_FUNCTIONS[binarizationFunction](step1, bestSolutionBin, indBin, chaotic_vals)
        
    except Exception as e:
        raise ValueError(f"Error en la función de binarización '{binarizationFunction}': {e}")

    return individuoBin