import numpy as np

# Sine Cosine Algorithm (SCA)
# https://doi.org/10.1016/j.knosys.2015.12.022

def iterarSCA(maxIter, iter, population, best):
    """
    Optimized Sine Cosine Algorithm (SCA) using vectorized operations.
    Args:
        maxIter (int): Maximum number of iterations.
        t (int): Current iteration.
        dimension (int): Number of dimensions.
        population (numpy.ndarray): Current population of solutions (pop_size x dimension).
        bestSolution (numpy.ndarray): Best solution found so far (1 x dimension).

    Returns:
        numpy.ndarray: Updated population after one iteration of SCA.
    """

    population = np.array(population)
    best = np.array(best)

    # a is a constant number, recommended value is 2
    a = 2
    # Compute r1 using equation 3.4
    r1 = a - (iter * (a / maxIter))
    
    # Vectorized random values
    rand1 = np.random.uniform(0.0, 1.0, population.shape)
    r2 = 2 * np.pi * rand1
    rand2 = np.random.uniform(0.0, 1.0, population.shape)
    r3 = 2 * rand2
    r4 = np.random.uniform(0.0, 1.0, population.shape)
    
    # Compute the sine and cosine components
    sin_component = r1 * np.sin(r2) * np.abs(r3 * best - population)
    cos_component = r1 * np.cos(r2) * np.abs(r3 * best - population)
    
    # Update population based on r4
    population = np.where(r4 < 0.5, population + sin_component, population + cos_component)
    
    return population