import numpy as np
import time

population = np.random.uniform(-10, 10, (10000, 30))

def fitness(x):
    return np.sum(x**2)

start = time.time()
fitness_values = np.sum(population**2, axis=1)
end = time.time()

print(f"Tiempo sin bucles: {end - start:.5f} segundos")