import numpy as np
import time

population = np.random.uniform(-10, 10, (10000, 30))

def fitness(x):
    return np.sum(x**2)

start = time.time()
fitness_values = [fitness(ind) for ind in population]
end = time.time()

print(f"Tiempo con bucles: {end - start:.5f} segundos")