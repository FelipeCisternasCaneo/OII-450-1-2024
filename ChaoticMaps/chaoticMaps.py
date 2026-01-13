"""
Chaotic Maps for Metaheuristic Algorithms - Unified Version

Implementación profesional de mapas caóticos para mejorar la exploración y 
explotación en algoritmos de optimización metaheurística.

Incluye todos los mapas del set original y las mejoras de validación del set nuevo.
"""

import numpy as np

# ========= MAPAS CAÓTICOS =========

def logisticMap(x0: float, quantity: int, mu: float = 4.0) -> np.ndarray:
    """Mapa Logístico: x_{n+1} = μ * x_n * (1 - x_n)"""
    if not (0 < x0 < 1): raise ValueError("x0 debe estar en (0, 1)")
    chaotic_seq = np.zeros(quantity)
    x = x0
    for i in range(quantity):
        x = mu * x * (1 - x)
        chaotic_seq[i] = x
    return chaotic_seq

def piecewiseMap(x0: float, quantity: int, p: float = 0.4) -> np.ndarray:
    """
    Mapa Piecewise: Dividido en 4 tramos para maximizar la cobertura del espacio.
    Se ha optimizado eliminando la recursión para mayor velocidad.
    """
    if not (0 < x0 < 1): raise ValueError("x0 debe estar en (0, 1)")
    chaotic_seq = np.zeros(quantity)
    x = x0
    for i in range(quantity):
        if 0 <= x < p:
            x = x / p
        elif p <= x < 0.5:
            x = (x - p) / (0.5 - p)
        elif 0.5 <= x < (1 - p):
            x = (1 - p - x) / (0.5 - p)
        else:
            x = (1 - x) / p
        chaotic_seq[i] = x
    return chaotic_seq

def sineMap(x0: float, quantity: int, a: float = 4.0) -> np.ndarray:
    """Mapa Seno: x_{n+1} = (a/4) * sin(π * x_n)"""
    if not (0 < x0 < 1): raise ValueError("x0 debe estar en (0, 1)")
    chaotic_seq = np.zeros(quantity)
    x = x0
    for i in range(quantity):
        x = (a / 4) * np.sin(np.pi * x)
        chaotic_seq[i] = abs(x)
    return chaotic_seq

def singerMap(x0: float, quantity: int, mu: float = 1.07) -> np.ndarray:
    """Mapa de Singer: Basado en un polinomio de 4to grado."""
    if not (0 < x0 < 1): raise ValueError("x0 debe estar en (0, 1)")
    chaotic_seq = np.zeros(quantity)
    x = x0
    for i in range(quantity):
        x = mu * (7.86 * x - 23.31 * x**2 + 28.75 * x**3 - 13.302875 * x**4)
        chaotic_seq[i] = np.clip(x, 0.0, 1.0)
    return chaotic_seq

def sinusoidalMap(x0: float, quantity: int, a: float = 2.3) -> np.ndarray:
    """Mapa Sinusoidal: x_{n+1} = a * x_n² * sin(π * x_n)"""
    if not (0 < x0 < 1): raise ValueError("x0 debe estar en (0, 1)")
    chaotic_seq = np.zeros(quantity)
    x = x0
    for i in range(quantity):
        x = a * x**2 * np.sin(np.pi * x)
        chaotic_seq[i] = abs(x) % 1.0
    return chaotic_seq

def tentMap(x0: float, quantity: int, mu: float = 2.0) -> np.ndarray:
    """Mapa Tent: x_{n+1} = μ*x si x<0.5, else μ*(1-x). Estándar μ=2.0."""
    if not (0 < x0 < 1): raise ValueError("x0 debe estar en (0, 1)")
    chaotic_seq = np.zeros(quantity)
    x = x0
    for i in range(quantity):
        x = mu * x if x < 0.5 else mu * (1 - x)
        chaotic_seq[i] = x % 1.0
    return chaotic_seq

def circleMap(x0: float, quantity: int, a: float = 0.5, b: float = 0.2) -> np.ndarray:
    """Mapa del Círculo: x_{n+1} = (x_n + b - (a/(2π)) * sin(2π * x_n)) mod 1"""
    if not (0 < x0 < 1): raise ValueError("x0 debe estar en (0, 1)")
    chaotic_seq = np.zeros(quantity)
    x = x0
    for i in range(quantity):
        x = (x + b - (a / (2 * np.pi)) * np.sin(2 * np.pi * x)) % 1.0
        chaotic_seq[i] = x
    return chaotic_seq

def chebyshevMap(x0: float, quantity: int) -> np.ndarray:
    """Mapa de Chebyshev: x_{n+1} = cos(x_n * (1 / cos(x_n)))"""
    if not (0 < x0 < 1): raise ValueError("x0 debe estar en (0, 1)")
    chaotic_seq = np.zeros(quantity)
    x = x0
    for i in range(quantity):
        x = np.cos(x * (1 / np.cos(x)))
        chaotic_seq[i] = (x + 1) / 2  # Normalizado a [0, 1]
    return chaotic_seq

def gaussMap(x0: float, quantity: int) -> np.ndarray:
    """Mapa de Gauss: x_{n+1} = 1 / (x_n mod 1)"""
    if not (0 < x0 < 1): raise ValueError("x0 debe estar en (0, 1)")
    chaotic_seq = np.zeros(quantity)
    x = x0
    for i in range(quantity):
        x = 0 if x == 0 else (1 / (x % 1.0))
        chaotic_seq[i] = x % 1.0
    return chaotic_seq

# ========= UTILIDADES =========

# Nombres legibles y alias para mostrar en logs
CHAOTIC_MAP_NAMES = {
    'LOG': 'Logistic',
    'PIECE': 'Piecewise',
    'SINE': 'Sine',
    'SINGER': 'Singer',
    'SINU': 'Sinusoidal',
    'TENT': 'Tent',
    'CIRCLE': 'Circle',
    'CHEB': 'Chebyshev',
    'GAUS': 'Gauss',
}

# Funciones del mapa caótico (acepta alias largos y cortos)
CHAOTIC_MAP_FUNCTIONS = {
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

def get_chaotic_map(map_name: str):
    """Retorna la función del mapa caótico por su sigla (insensible a mayúsculas)."""
    key = map_name.upper()
    if key not in CHAOTIC_MAP_FUNCTIONS:
        validas = list(CHAOTIC_MAP_FUNCTIONS.keys())
        raise ValueError(f"Mapa '{map_name}' no válido. Opciones: {validas}")
    return CHAOTIC_MAP_FUNCTIONS[key]