import math
import random
import numpy as np

# Whale Optimization Algorithm (WOA)
# https://doi.org/10.1016/j.advengsoft.2016.01.008

def iterarWOA(maxIter, t, dimension, population, bestSolution):
    """
    Whale Optimization Algorithm (WOA)
    Args:
        maxIter (int): Maximum number of iterations.
        t (int): Current iteration.
        dimension (int): Number of dimensions.
        population (list of lists): Current population of solutions.
        bestSolution (list): Best solution found so far.

    Returns:
        list of lists: Updated population after one iteration of WOA.
    """
    
    a = 2 - (2 * t / maxIter)
    b = 1  # Constant for logarithmic spiral

    new_population = []

    for individual in population:
        # Random numbers for conditions
        p = random.uniform(0, 1)
        r = random.uniform(0, 1)
        l = random.uniform(-1, 1)

        # Calculate A and C (Equations 2.3 and 2.4)
        A = 2 * a * r - a
        C = 2 * random.uniform(0, 1)

        if p < 0.5:  # Exploitation phase
            if abs(A) < 1:  # Encircling prey (Equation 2.1)
                D = [abs(C * bestSolution[j] - individual[j]) for j in range(dimension)]
                new_individual = [bestSolution[j] - A * D[j] for j in range(dimension)]
            else:  # Search for prey (Equation 2.7)
                random_index = random.randint(0, len(population) - 1)
                random_individual = population[random_index]
                D = [abs(C * random_individual[j] - individual[j]) for j in range(dimension)]
                new_individual = [random_individual[j] - A * D[j] for j in range(dimension)]
        else:  # Spiral updating (Equation 2.5)
            D_prime = [bestSolution[j] - individual[j] for j in range(dimension)]
            spiral_component = math.exp(b * l) * math.cos(2 * math.pi * l)
            new_individual = [
                D_prime[j] * spiral_component + bestSolution[j]
                for j in range(dimension)
            ]

        new_population.append(new_individual)

    return np.array(new_population)