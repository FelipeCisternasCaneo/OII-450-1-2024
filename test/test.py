import numpy as np
from Problem.Benchmark.Problem import fitness as f

# Crear una población de prueba
test_population = np.random.uniform(-100, 100, (10, 30))  # 10 individuos, 30 dimensiones
test_function = 'F8'  # Reemplaza con una función que uses en tus experimentos

# Probar si `f` soporta una entrada matricial
try:
    test_fitness = f(test_function, test_population)
    print("La función `f` es vectorizable. Resultados del fitness para la población:")
    print(test_fitness)
except Exception as e:
    print("La función `f` no es vectorizable. Error encontrado:")
    print(e)