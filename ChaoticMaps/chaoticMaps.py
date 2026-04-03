"""\
Chaotic Maps for Metaheuristic Algorithms - Optimized Version

Implementación de mapas caóticos para mejorar la exploración y 
explotación en algoritmos de optimización metaheurística.

Optimizado con Numba JIT para rendimiento cercano a C en los loops
recurrentes (cada valor depende del anterior, no es vectorizable con NumPy).
"""

import numpy as np
from numba import njit

# ========= KERNELS JIT (funciones internas compiladas) =========

@njit(cache=True)
def _logistic_kernel(x0: float, quantity: int, mu: float) -> np.ndarray:
    seq = np.empty(quantity)
    x = x0
    for i in range(quantity):
        x = mu * x * (1.0 - x)
        seq[i] = x
    return seq

@njit(cache=True)
def _piecewise_kernel(x0: float, quantity: int, p: float) -> np.ndarray:
    seq = np.empty(quantity)
    x = x0
    for i in range(quantity):
        if x < p:
            x = x / p
        elif x < 0.5:
            x = (x - p) / (0.5 - p)
        elif x < (1.0 - p):
            x = (1.0 - p - x) / (0.5 - p)
        else:
            x = (1.0 - x) / p
        seq[i] = x
    return seq

@njit(cache=True)
def _sine_kernel(x0: float, quantity: int, a: float) -> np.ndarray:
    seq = np.empty(quantity)
    x = x0
    a_over_4 = a / 4.0
    for i in range(quantity):
        x = a_over_4 * np.sin(np.pi * x)
        seq[i] = abs(x)
    return seq

@njit(cache=True)
def _singer_kernel(x0: float, quantity: int, mu: float) -> np.ndarray:
    seq = np.empty(quantity)
    x = x0
    for i in range(quantity):
        x = mu * (7.86 * x - 23.31 * x * x + 28.75 * x * x * x - 13.302875 * x * x * x * x)
        # Clip manual (np.clip no soportado en njit con escalares en todas las versiones)
        if x < 0.0:
            x = 0.0
        elif x > 1.0:
            x = 1.0
        seq[i] = x
    return seq

@njit(cache=True)
def _sinusoidal_kernel(x0: float, quantity: int, a: float) -> np.ndarray:
    seq = np.empty(quantity)
    x = x0
    for i in range(quantity):
        x = a * x * x * np.sin(np.pi * x)
        x_abs = abs(x)
        x = x_abs - np.floor(x_abs)  # equivalente a abs(x) % 1.0
        seq[i] = x
    return seq

@njit(cache=True)
def _tent_kernel(x0: float, quantity: int, mu: float) -> np.ndarray:
    seq = np.empty(quantity)
    x = x0
    for i in range(quantity):
        if x < 0.5:
            x = mu * x
        else:
            x = mu * (1.0 - x)
        x = x - np.floor(x)  # equivalente a x % 1.0
        seq[i] = x
    return seq

@njit(cache=True)
def _circle_kernel(x0: float, quantity: int, a: float, b: float) -> np.ndarray:
    seq = np.empty(quantity)
    x = x0
    two_pi = 2.0 * np.pi
    a_over_two_pi = a / two_pi
    for i in range(quantity):
        x = x + b - a_over_two_pi * np.sin(two_pi * x)
        x = x - np.floor(x)  # equivalente a x % 1.0
        seq[i] = x
    return seq

@njit(cache=True)
def _chebyshev_kernel(x0: float, quantity: int) -> np.ndarray:
    seq = np.empty(quantity)
    x = x0
    for i in range(quantity):
        x = np.cos(x * (1.0 / np.cos(x)))
        seq[i] = (x + 1.0) / 2.0
    return seq

@njit(cache=True)
def _gauss_kernel(x0: float, quantity: int) -> np.ndarray:
    seq = np.empty(quantity)
    x = x0
    for i in range(quantity):
        if x == 0.0:
            x = 0.0
        else:
            x_mod = x - np.floor(x)  # x % 1.0
            if x_mod == 0.0:
                x = 0.0
            else:
                x = 1.0 / x_mod
        seq[i] = x - np.floor(x)  # x % 1.0
    return seq


# ========= API PÚBLICA (misma interfaz que antes) =========

def logisticMap(x0: float, quantity: int, mu: float = 4.0) -> np.ndarray:
    """Mapa Logístico: x_{n+1} = μ * x_n * (1 - x_n)"""
    if not (0 < x0 < 1):
        raise ValueError("x0 debe estar en el intervalo (0, 1)")
    return _logistic_kernel(x0, quantity, mu)

def piecewiseMap(x0: float, quantity: int, p: float = 0.4) -> np.ndarray:
    """Mapa Piecewise: Dividido en 4 tramos para maximizar la cobertura del espacio."""
    if not (0 < x0 < 1):
        raise ValueError("x0 debe estar en (0, 1)")
    return _piecewise_kernel(x0, quantity, p)

def sineMap(x0: float, quantity: int, a: float = 2.3) -> np.ndarray:
    """Mapa Seno: x_{n+1} = (a/4) * sin(π * x_n)"""
    if not (0 < x0 < 1):
        raise ValueError("x0 debe estar en el intervalo (0, 1)")
    return _sine_kernel(x0, quantity, a)

def singerMap(x0: float, quantity: int, mu: float = 1.07) -> np.ndarray:
    """Mapa de Singer: Basado en un polinomio de 4to grado."""
    if not (0 < x0 < 1):
        raise ValueError("x0 debe estar en el intervalo (0, 1)")
    return _singer_kernel(x0, quantity, mu)

def sinusoidalMap(x0: float, quantity: int, a: float = 2.3) -> np.ndarray:
    """Mapa Sinusoidal: x_{n+1} = a * x_n² * sin(π * x_n)"""
    if not (0 < x0 < 1):
        raise ValueError("x0 debe estar en el intervalo (0, 1)")
    return _sinusoidal_kernel(x0, quantity, a)

def tentMap(x0: float, quantity: int, mu: float = 2.0) -> np.ndarray:
    """Mapa Tent: x_{n+1} = μ*x si x<0.5, else μ*(1-x). Estándar μ=2.0."""
    if not (0 < x0 < 1):
        raise ValueError("x0 debe estar en el intervalo (0, 1)")
    return _tent_kernel(x0, quantity, mu)

def circleMap(x0: float, quantity: int, a: float = 0.5, b: float = 0.2) -> np.ndarray:
    """Mapa del Círculo: x_{n+1} = (x_n + b - (a/(2π)) * sin(2π * x_n)) mod 1"""
    if not (0 < x0 < 1):
        raise ValueError("x0 debe estar en el intervalo (0, 1)")
    return _circle_kernel(x0, quantity, a, b)

def chebyshevMap(x0: float, quantity: int) -> np.ndarray:
    """Mapa de Chebyshev: x_{n+1} = cos(x_n * (1 / cos(x_n)))"""
    if not (0 < x0 < 1):
        raise ValueError("x0 debe estar en el intervalo (0, 1)")
    return _chebyshev_kernel(x0, quantity)

def gaussMap(x0: float, quantity: int) -> np.ndarray:
    """Mapa de Gauss: x_{n+1} = 1 / (x_n mod 1)"""
    if not (0 < x0 < 1):
        raise ValueError("x0 debe estar en el intervalo (0, 1)")
    return _gauss_kernel(x0, quantity)


# ========= UTILIDADES =========

# Nombres legibles y alias para mostrar en logs
CHAOTIC_MAP_NAMES = {
    'LOG': 'Logistic Map',
    'PIECE': 'Piecewise Map',
    'SINE': 'Sine Map',
    'SINGER': 'Singer Map',
    'SINU': 'Sinusoidal Map',
    'TENT': 'Tent Map',
    'CIRCLE': 'Circle Map',
    'CHEB': 'Chebyshev Map',
    'GAUS': 'Gauss Map',
}


def get_chaotic_map(map_name: str):
    """Retorna la función del mapa caótico por su sigla (insensible a mayúsculas)."""
    map_functions = {
        'LOG': logisticMap,
        'PIECE': piecewiseMap,
        'SINE': sineMap,
        'SINGER': singerMap,
        'SINU': sinusoidalMap,
        'TENT': tentMap,
        'CIRCLE': circleMap,
        'CHEB': chebyshevMap,
        'GAUS': gaussMap,
    }
    
    if map_name not in map_functions:
        raise ValueError(f"Mapa caótico '{map_name}' no reconocido. Opciones válidas: {list(map_functions.keys())}")
    
    return map_functions[map_name]