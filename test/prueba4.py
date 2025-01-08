import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# probar en un futuro

# Definir la función objetivo
def f1(x, y):
    return x**2 + y**2

# Crear el espacio de búsqueda
x = np.linspace(-100, 100, 200)
y = np.linspace(-100, 100, 200)
X, Y = np.meshgrid(x, y)
Z = f1(X, Y)

# Inicializar partículas
np.random.seed(42)
num_agents = 50
particles = np.random.uniform(-100, 100, (num_agents, 2))

# Simular iteraciones
iterations = [1, 100, 300, 400, 500]
particle_positions = []

for it in range(max(iterations) + 1):
    # Simula movimiento hacia el óptimo (0, 0)
    particles -= 0.1 * particles  # Simplificación del movimiento
    if it in iterations:
        particle_positions.append(particles.copy())

# Crear la figura
fig = plt.figure(figsize=(16, 8))

# Gráfico 3D de la función objetivo
ax1 = fig.add_subplot(1, len(iterations) + 1, 1, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.contour(X, Y, Z, levels=15, offset=0, cmap='viridis', linestyles="solid")
ax1.set_title('Función Objetivo (F1)')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('F(x1, x2)')

# Gráficos de distribución de partículas en iteraciones seleccionadas
for i, (it, positions) in enumerate(zip(iterations, particle_positions), start=2):
    ax = fig.add_subplot(1, len(iterations) + 1, i)
    contour = ax.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.8)
    ax.scatter(positions[:, 0], positions[:, 1], color='red', label=f'Iteración {it}', alpha=0.7)
    ax.set_title(f'Iteración {it}')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()

plt.tight_layout()
plt.show()