import numpy as np

def iterarPSO(maxIter, iter, dim, population, best, pBest, vel, ub0):
    """
    Particle Swarm Optimization (PSO) - Optimized with NumPy.
    """
    
    Vmax = ub0 * 0.1
    wMax = 0.9
    wMin = 0.1
    c1 = 2
    c2 = 2

    # Update the inertia weight (w)
    w = wMax - iter * ((wMax - wMin) / maxIter)

    # Generate random values r1 and r2 for all particles and dimensions
    r1 = np.random.rand(population.shape[0], dim)
    r2 = np.random.rand(population.shape[0], dim)

    # Update velocities
    vel = (
        w * vel
        + c1 * r1 * (pBest - population)
        + c2 * r2 * (best - population)
    )

    # Clip velocities to the maximum range
    vel = np.clip(vel, -Vmax, Vmax)

    # Update positions
    population = population + vel

    return population, vel
