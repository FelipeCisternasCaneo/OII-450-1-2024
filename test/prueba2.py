import numpy as np
import matplotlib.pyplot as plt

# Configuración inicial
np.random.seed(42)
num_particles = 50  # Número de partículas
iterations = 500  # Iteraciones del algoritmo
radius = 10  # Radio del espacio de búsqueda
center = np.array([0, 0])  # Centro (óptimo global)

# Inicializar partículas aleatoriamente en un espacio 2D
particles = np.random.uniform(-radius, radius, (num_particles, 2))

# Función para actualizar las partículas (simula convergencia)
def update_particles(particles, center, step_size=0.2):
    for i in range(len(particles)):
        # Movimiento hacia el centro (óptimo global)
        direction = center - particles[i]
        particles[i] += step_size * direction
    return particles

# Visualización de la simulación
plt.figure(figsize=(8, 8))
circle = plt.Circle((0, 0), radius, color='lightblue', fill=False, linestyle='--')
plt.gca().add_artist(circle)
plt.scatter(center[0], center[1], color='red', label='Óptimo Global', zorder=5)

# Simular iteraciones
for iteration in range(iterations):
    if iteration % 10 == 0 or iteration == iterations - 1:  # Graficar cada 10 iteraciones
        plt.scatter(particles[:, 0], particles[:, 1], label=f'Iteración {iteration}')
    particles = update_particles(particles, center)

# Detalles del gráfico
plt.xlim(-radius-2, radius+2)
plt.ylim(-radius-2, radius+2)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.title('Convergencia de Partículas hacia el Óptimo')
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.show()