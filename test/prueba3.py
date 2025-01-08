import numpy as np
import matplotlib.pyplot as plt

# Función objetivo
def objective_function(x, y):
    return x**2 + y**2

x = np.linspace(-100, 100, 200)
y = np.linspace(-100, 100, 200)
X, Y = np.meshgrid(x, y)
Z = objective_function(X, Y)

# Gráfico 3D del paisaje de búsqueda
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.set_title("Paisaje de la Función Objetivo")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("F(x1, x2)")
plt.show()

np.random.seed(42)
num_particles = 50
iterations = 100
particles = np.random.uniform(-100, 100, (num_particles, 2))
best_positions = []

for _ in range(iterations):
    particles -= 0.1 * particles  # Simular movimiento hacia el origen
    best_positions.append(particles.mean(axis=0))

plt.figure(figsize=(8, 6))
plt.scatter(particles[:, 0], particles[:, 1], label='Posiciones Finales', alpha=0.7)
plt.scatter(0, 0, color='red', label='Óptimo Global')
plt.title("Historial de Búsqueda")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid()
plt.show()

best_scores = [np.linalg.norm(pos) for pos in best_positions]
plt.figure(figsize=(8, 6))
plt.plot(range(iterations), best_scores)
plt.title("Curva de Convergencia")
plt.xlabel("Iteraciones")
plt.ylabel("Mejor Aptitud Alcanzada")
plt.grid()
plt.show()