import numpy as np

def diversity_per_dimension(population: np.ndarray):
    """
    Computes per-dimension diversity based on median absolute deviation.
    Div_j = mean_i |median(x_j) - x_{i,j}|
    Returns:
        divj_vec: np.ndarray (D,)  -> diversity per dimension
        divj_mean, divj_min, divj_max: float
    """
    median = np.median(population, axis=0)
    divj_vec = np.mean(np.abs(population - median), axis=0)
    return divj_vec, float(np.mean(divj_vec)), float(np.min(divj_vec)), float(np.max(divj_vec))