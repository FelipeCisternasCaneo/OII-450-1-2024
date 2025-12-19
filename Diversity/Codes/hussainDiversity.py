import numpy as np

def diversidadHussain(matriz):
    """
    Calcula la diversidad de Hussain de forma vectorizada.
    
    Fórmula: (1 / (l * n)) * sum(abs(matriz[i][d] - media[d]))
    donde media[d] es el promedio de la dimensión d.
    """
    # Asegurar que sea array de NumPy
    matriz = np.asarray(matriz)
    
    n, l = matriz.shape  # n = filas (población), l = columnas (dimensiones)
    
    # Calcular medias por columna (vectorizado)
    medianas = np.mean(matriz, axis=0)
    
    # Calcular suma de diferencias absolutas (vectorizado)
    diversidad = np.sum(np.abs(matriz - medianas))
    
    # Aplicar fórmula final
    return round((diversidad / (l * n)), 3)