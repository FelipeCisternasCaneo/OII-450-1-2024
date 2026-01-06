"""
Chaotic Maps for Metaheuristic Algorithms

Implementación de mapas caóticos para mejorar la exploración en metaheurísticas.
Cada mapa genera una secuencia pseudoaleatoria con propiedades caóticas.

Referencias:
- Tizhoosh, H. R. (2005). Opposition-based learning: A new scheme for machine intelligence.
- Gandomi, A. H., et al. (2013). Chaotic bat algorithm. Journal of Computational Science.
"""

import numpy as np

# ========= MAPAS CAÓTICOS =========

def logisticMap(x0: float, quantity: int, mu: float = 4.0) -> np.ndarray:
    """
    Mapa Logístico: x_{n+1} = μ * x_n * (1 - x_n)
    
    Args:
        x0: Valor inicial en [0, 1]
        quantity: Cantidad de valores a generar
        mu: Parámetro de control (típicamente 4.0 para caos completo)
    
    Returns:
        Array de valores caóticos en [0, 1]
    """
    if not (0 < x0 < 1):
        raise ValueError("x0 debe estar en el intervalo (0, 1)")
    
    chaotic_seq = np.zeros(quantity)
    x = x0
    
    for i in range(quantity):
        x = mu * x * (1 - x)
        chaotic_seq[i] = x
    
    return chaotic_seq


def piecewiseMap(x0: float, quantity: int, p: float = 0.4) -> np.ndarray:
    """
    Mapa Piecewise (por tramos):
    
    x_{n+1} = { x_n / p,                if x_n ∈ [0, p)
              { (x_n - p) / (0.5 - p),  if x_n ∈ [p, 0.5)
              { piecewise(1 - x_n),     if x_n ∈ [0.5, 1]
    
    Args:
        x0: Valor inicial en [0, 1]
        quantity: Cantidad de valores a generar
        p: Parámetro de control en (0, 0.5)
    
    Returns:
        Array de valores caóticos en [0, 1]
    """
    if not (0 < x0 < 1):
        raise ValueError("x0 debe estar en el intervalo (0, 1)")
    if not (0 < p < 0.5):
        raise ValueError("p debe estar en el intervalo (0, 0.5)")
    
    chaotic_seq = np.zeros(quantity)
    x = x0
    
    for i in range(quantity):
        if 0 <= x < p:
            x = x / p
        elif p <= x < 0.5:
            x = (x - p) / (0.5 - p)
        else:
            x = piecewiseMap(1 - x, 1, p)[0]
        
        chaotic_seq[i] = x
    
    return chaotic_seq


def sineMap(x0: float, quantity: int, a: float = 2.3) -> np.ndarray:
    """
    Mapa Seno: x_{n+1} = (a/4) * sin(π * x_n)
    
    Args:
        x0: Valor inicial en [0, 1]
        quantity: Cantidad de valores a generar
        a: Parámetro de control (típicamente entre 0 y 4)
    
    Returns:
        Array de valores caóticos en [0, 1]
    """
    if not (0 < x0 < 1):
        raise ValueError("x0 debe estar en el intervalo (0, 1)")
    
    chaotic_seq = np.zeros(quantity)
    x = x0
    
    for i in range(quantity):
        x = (a / 4) * np.sin(np.pi * x)
        # Normalizar a [0, 1]
        x = abs(x)
        chaotic_seq[i] = x
    
    return chaotic_seq


def singerMap(x0: float, quantity: int, mu: float = 1.07) -> np.ndarray:
    """
    Mapa de Singer: x_{n+1} = μ * (7.86 * x_n - 23.31 * x_n² + 28.75 * x_n³ - 13.302875 * x_n⁴)
    
    Args:
        x0: Valor inicial en [0.0, 1.0]
        quantity: Cantidad de valores a generar
        mu: Parámetro de control (típicamente 1.07)
    
    Returns:
        Array de valores caóticos en [0, 1]
    """
    if not (0 < x0 < 1):
        raise ValueError("x0 debe estar en el intervalo (0, 1)")
    
    chaotic_seq = np.zeros(quantity)
    x = x0
    
    for i in range(quantity):
        x = mu * (7.86 * x - 23.31 * x**2 + 28.75 * x**3 - 13.302875 * x**4)
        # Asegurar que está en [0, 1]
        x = np.clip(x, 0.0, 1.0)
        chaotic_seq[i] = x
    
    return chaotic_seq


def sinusoidalMap(x0: float, quantity: int, a: float = 2.3) -> np.ndarray:
    """
    Mapa Sinusoidal: x_{n+1} = a * x_n² * sin(π * x_n)
    
    Args:
        x0: Valor inicial en [0, 1]
        quantity: Cantidad de valores a generar
        a: Parámetro de control
    
    Returns:
        Array de valores caóticos en [0, 1]
    """
    if not (0 < x0 < 1):
        raise ValueError("x0 debe estar en el intervalo (0, 1)")
    
    chaotic_seq = np.zeros(quantity)
    x = x0
    
    for i in range(quantity):
        x = a * x**2 * np.sin(np.pi * x)
        # Normalizar a [0, 1]
        x = abs(x) % 1.0
        chaotic_seq[i] = x
    
    return chaotic_seq


def tentMap(x0: float, quantity: int, mu: float = 2.0) -> np.ndarray:
    """
    Mapa Tent (Tienda de campaña):
    
    x_{n+1} = { μ * x_n,         if x_n < 0.5
              { μ * (1 - x_n),   if x_n ≥ 0.5
    
    Args:
        x0: Valor inicial en [0, 1]
        quantity: Cantidad de valores a generar
        mu: Parámetro de control (típicamente 2.0)
    
    Returns:
        Array de valores caóticos en [0, 1]
    """
    if not (0 < x0 < 1):
        raise ValueError("x0 debe estar en el intervalo (0, 1)")
    
    chaotic_seq = np.zeros(quantity)
    x = x0
    
    for i in range(quantity):
        if x < 0.5:
            x = mu * x
        else:
            x = mu * (1 - x)
        
        # Asegurar que está en [0, 1]
        x = x % 1.0
        chaotic_seq[i] = x
    
    return chaotic_seq


def circleMap(x0: float, quantity: int, a: float = 0.5, b: float = 0.2) -> np.ndarray:
    """
    Mapa del Círculo: x_{n+1} = (x_n + b - (a/(2π)) * sin(2π * x_n)) mod 1
    
    Args:
        x0: Valor inicial en [0, 1]
        quantity: Cantidad de valores a generar
        a: Parámetro de control (típicamente 0.5)
        b: Parámetro de rotación (típicamente 0.2)
    
    Returns:
        Array de valores caóticos en [0, 1]
    """
    if not (0 < x0 < 1):
        raise ValueError("x0 debe estar en el intervalo (0, 1)")
    
    chaotic_seq = np.zeros(quantity)
    x = x0
    
    for i in range(quantity):
        x = (x + b - (a / (2 * np.pi)) * np.sin(2 * np.pi * x)) % 1.0
        chaotic_seq[i] = x
    
    return chaotic_seq


# ========= UTILIDADES =========

CHAOTIC_MAP_NAMES = {
    'LOG': 'Logistic Map',
    'PIECE': 'Piecewise Map',
    'SINE': 'Sine Map',
    'SINGER': 'Singer Map',
    'SINU': 'Sinusoidal Map',
    'TENT': 'Tent Map',
    'CIRCLE': 'Circle Map'
}


def get_chaotic_map(map_name: str):
    """
    Retorna la función del mapa caótico correspondiente.
    
    Args:
        map_name: Nombre del mapa ('LOG', 'PIECE', 'SINE', etc.)
    
    Returns:
        Función del mapa caótico
    
    Raises:
        ValueError: Si el nombre del mapa no es válido
    """
    map_functions = {
        'LOG': logisticMap,
        'PIECE': piecewiseMap,
        'SINE': sineMap,
        'SINGER': singerMap,
        'SINU': sinusoidalMap,
        'TENT': tentMap,
        'CIRCLE': circleMap
    }
    
    if map_name not in map_functions:
        raise ValueError(f"Mapa caótico '{map_name}' no reconocido. Opciones válidas: {list(map_functions.keys())}")
    
    return map_functions[map_name]