import numpy as np

# Grey Wolf Optimizer (GWO)
# https://doi.org/10.1016/j.advengsoft.2013.12.007

def iterarGWO(maxIter, iter, dim, population, fitness, objective_type):
    """
    Grey Wolf Optimizer (GWO).
    Optimized to work with numpy arrays.
    """
    population = np.array(population)
    fitness = np.array(fitness)

    a = 2 - iter * ((2) / maxIter)  # 'a' decreases linearly from 2 to 0

    # Sort positions based on fitness
    if objective_type == "MIN":
        sorted_indices = np.argsort(fitness)
    elif objective_type == "MAX":
        sorted_indices = np.argsort(fitness)[::-1]
    else:
        raise ValueError("typeProblem must be 'MIN' or 'MAX'.")

    # Extract alpha, beta, and delta wolves
    Xalfa = population[sorted_indices[0]]
    Xbeta = population[sorted_indices[1]]
    Xdelta = population[sorted_indices[2]]

    # Generate random values for all calculations
    r1 = np.random.uniform(0.0, 1.0, (population.shape[0], dim, 3))  # 3 sets of r1
    r2 = np.random.uniform(0.0, 1.0, (population.shape[0], dim, 3))  # 3 sets of r2

    # Calculate A and C for all wolves
    A = 2 * a * r1 - a  # Shape: (pop, dim, 3)
    C = 2 * r2  # Shape: (pop, dim, 3)

    # Calculate distances
    d_alfa = np.abs(C[:, :, 0] * Xalfa - population)
    d_beta = np.abs(C[:, :, 1] * Xbeta - population)
    d_delta = np.abs(C[:, :, 2] * Xdelta - population)

    # Update positions using GWO equations
    X1 = Xalfa - A[:, :, 0] * d_alfa
    X2 = Xbeta - A[:, :, 1] * d_beta
    X3 = Xdelta - A[:, :, 2] * d_delta

    # Average the contributions
    population = (X1 + X2 + X3) / 3

    return population