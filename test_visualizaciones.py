"""
Script de prueba para visualizaciones de trayectoria y exploración espacial.
Versión corregida: 
- Search history completo (todos los puntos)
- Óptimos correctos para cada función
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Problem.Benchmark.Problem import fitness as f
from Problem.Benchmark.CEC.cec2017 import functions as cec2017
from Util.visualizaciones import (
    graficar_trayectoria_dimensional,
    graficar_search_history_2d,
    graficar_search_history_2d_completo,
    graficar_search_history_slice,
    obtener_optimo
)


def simular_ejecucion_2d(mh, funcion, maxIter=100, pop_size=30):
    """Simula una ejecución en 2D y retorna datos."""
    print(f"\n🔄 Simulando ejecución de {mh} en {funcion} (2D)...")
    
    # Obtener óptimo real de la función
    optimo_fitness = obtener_optimo(funcion)
    
    # Para F8 (Schwefel), el óptimo está en 420.9687 en cada dimensión
    if funcion == 'F8':
        optimo_pos = np.array([420.9687, 420.9687])
        lb = np.array([0, 0])  # Schwefel típicamente en [0, 500]
        ub = np.array([500, 500])
    else:
        optimo_pos = np.array([0, 0])  # La mayoría están centradas en 0
        lb = np.array([-100, -100])
        ub = np.array([100, 100])
    
    dim = 2
    
    # Población inicial
    population = np.random.uniform(lb, ub, (pop_size, dim))
    
    # Trayectoria del mejor
    trajectory = []
    fitness_traj = []
    search_history = []
    all_positions = []  # TODOS los puntos visitados
    
    # Punto inicial (mejor de la población)
    fitness_pop = np.array([f(funcion, ind) for ind in population])
    best_idx = np.argmin(fitness_pop)
    current_best = population[best_idx].copy()
    
    sample_iterations = [0, maxIter//5, 2*maxIter//5, 3*maxIter//5, 4*maxIter//5, maxIter-1]
    
    for iter in range(maxIter):
        # Guardar mejor solución
        trajectory.append(current_best.copy())
        fitness_traj.append(f(funcion, current_best))
        
        # Guardar TODAS las posiciones de esta iteración
        all_positions.extend(population.copy())
        
        # Guardar snapshot de población
        if iter in sample_iterations:
            search_history.append({
                'iter': iter,
                'population': population.copy()
            })
        
        # Simular evolución hacia el óptimo correcto
        noise_factor = (1 - iter/maxIter)
        
        for i in range(pop_size):
            noise = np.random.randn(dim) * noise_factor * 20
            # Converger hacia el óptimo real
            population[i] = population[i] * 0.90 + optimo_pos * 0.10 + noise
            population[i] = np.clip(population[i], lb, ub)
        
        # Actualizar mejor
        fitness_pop = np.array([f(funcion, ind) for ind in population])
        best_idx = np.argmin(fitness_pop)
        current_best = population[best_idx].copy()
    
    all_positions = np.array(all_positions)
    
    return (np.array(trajectory), np.array(fitness_traj), 
            search_history, all_positions, lb, ub, dim)


def simular_ejecucion_nd(mh, funcion, dim=30, maxIter=200, pop_size=50):
    """Simula una ejecución en alta dimensión."""
    print(f"\n🔄 Simulando ejecución de {mh} en {funcion} ({dim}D)...")
    
    # Obtener óptimo real
    optimo_fitness = obtener_optimo(funcion)
    
    if funcion == 'F8':
        optimo_pos = np.array([420.9687] * dim)
        lb = np.array([0] * dim)
        ub = np.array([500] * dim)
    else:
        optimo_pos = np.zeros(dim)
        lb = np.array([-100] * dim)
        ub = np.array([100] * dim)
    
    # Población inicial
    population = np.random.uniform(lb, ub, (pop_size, dim))
    
    trajectory = []
    fitness_traj = []
    search_history = []
    
    # Punto inicial
    fitness_pop = np.array([f(funcion, ind) for ind in population])
    best_idx = np.argmin(fitness_pop)
    current_best = population[best_idx].copy()
    
    sample_iterations = [0, maxIter//5, 2*maxIter//5, 3*maxIter//5, 4*maxIter//5, maxIter-1]
    
    for iter in range(maxIter):
        trajectory.append(current_best.copy())
        fitness_traj.append(f(funcion, current_best))
        
        if iter in sample_iterations:
            search_history.append({
                'iter': iter,
                'population': population.copy()
            })
        
        # Evolución hacia el óptimo correcto
        noise_factor = (1 - iter/maxIter)
        
        for i in range(pop_size):
            noise = np.random.randn(dim) * noise_factor * 20
            population[i] = population[i] * 0.90 + optimo_pos * 0.10 + noise
            population[i] = np.clip(population[i], lb, ub)
        
        fitness_pop = np.array([f(funcion, ind) for ind in population])
        best_idx = np.argmin(fitness_pop)
        current_best = population[best_idx].copy()
    
    return (np.array(trajectory), np.array(fitness_traj), 
            search_history, lb, ub, dim)


def test_search_history_completo():
    """
    Test: Search History Completo (como en papers)
    Muestra TODOS los puntos visitados en una sola imagen.
    """
    print("\n" + "="*70)
    print("🧪 TEST: Search History Completo (Todos los Puntos)")
    print("="*70)
    
    funciones = ['F1', 'F8', 'F9']  # Sphere, Schwefel, Rastrigin
    mh = 'PSO'
    
    for funcion in funciones:
        traj, fit, hist, all_pos, lb, ub, dim = simular_ejecucion_2d(
            mh, funcion, maxIter=150, pop_size=30
        )
        
        # Gráfico completo (todos los puntos)
        graficar_search_history_2d_completo(mh, funcion, all_pos, lb, ub)
        
        # Gráfico con snapshots (para comparar)
        graficar_search_history_2d(mh, funcion, hist, lb, ub)
    
    print("\n✅ Test Search History Completo terminado")


def test_trayectoria_dimensional():
    """Test: Trayectoria Dimensional"""
    print("\n" + "="*70)
    print("🧪 TEST: Trayectoria Dimensional")
    print("="*70)
    
    casos = [
        ('GWO', 'F1', 10),   # Sphere
        ('PSO', 'F8', 30),   # Schwefel (óptimo != 0)
        ('WOA', 'F9', 5),    # Rastrigin
    ]
    
    for mh, funcion, dim in casos:
        trajectory, fitness_traj, search_history, lb, ub, _ = simular_ejecucion_nd(
            mh, funcion, dim=dim, maxIter=150
        )
        graficar_trayectoria_dimensional(mh, funcion, trajectory, dim)
        print(f"   Óptimo teórico de {funcion}: {obtener_optimo(funcion)}")
    
    print("\n✅ Test Trayectoria Dimensional completado")


def test_completo():
    """Test completo con todas las visualizaciones."""
    print("\n" + "="*70)
    print("🧪 TEST COMPLETO: Todas las Visualizaciones")
    print("="*70)
    
    # Caso 2D con Schwefel (óptimo != 0)
    print("\n--- Función F8 (Schwefel 2D) ---")
    mh = 'GWO'
    funcion = 'F8'
    
    traj, fit, hist, all_pos, lb, ub, dim = simular_ejecucion_2d(mh, funcion, maxIter=150)
    
    print(f"   Óptimo teórico: {obtener_optimo(funcion)}")
    print(f"   Fitness final alcanzado: {fit[-1]:.4f}")
    
    graficar_trayectoria_dimensional(mh, funcion, traj, dim)
    graficar_search_history_2d_completo(mh, funcion, all_pos, lb, ub)
    graficar_search_history_2d(mh, funcion, hist, lb, ub)
    
    # Caso Alta Dimensión
    print("\n--- Función F9 (Rastrigin 30D) ---")
    mh_nd = 'PSO'
    func_nd = 'F9'
    dim_nd = 30
    
    traj, fit, hist, lb, ub, _ = simular_ejecucion_nd(mh_nd, func_nd, dim=dim_nd, maxIter=200)
    
    print(f"   Óptimo teórico: {obtener_optimo(func_nd)}")
    print(f"   Fitness final alcanzado: {fit[-1]:.4f}")
    
    graficar_trayectoria_dimensional(mh_nd, func_nd, traj, dim_nd)
    graficar_search_history_slice(mh_nd, func_nd, hist, 0, 1, lb, ub, dim_nd)
    
    print("\n✅ Test Completo terminado")


def simular_ejecucion_cec2017_2d(mh, funcion_cec, maxIter=100, pop_size=30):
    """
    Simula una ejecución con funciones CEC2017 en 2D.
    
    Args:
        mh: nombre de la metaheurística
        funcion_cec: función CEC2017 (ej: cec2017.f1, cec2017.f3)
        maxIter: número de iteraciones
        pop_size: tamaño de población
    
    Returns:
        Tupla con (trajectory, fitness_traj, search_history, all_positions, lb, ub, dim)
    """
    print(f"\n🔄 Simulando ejecución de {mh} en CEC2017-{funcion_cec.__name__} (2D)...")
    
    dim = 2
    lb = np.array([-100] * dim)
    ub = np.array([100] * dim)
    
    # El óptimo de CEC2017 es siempre función_number * 100
    func_num = int(funcion_cec.__name__[1:])  # f1 -> 1, f3 -> 3, etc.
    optimo_fitness = func_num * 100
    
    # Población inicial
    population = np.random.uniform(lb, ub, (pop_size, dim))
    
    trajectory = []
    fitness_traj = []
    search_history = []
    all_positions = []
    
    # Evaluar con CEC2017 (espera [n_samples, dim])
    fitness_pop = funcion_cec(population)
    best_idx = np.argmin(fitness_pop)
    current_best = population[best_idx].copy()
    
    sample_iterations = [0, maxIter//5, 2*maxIter//5, 3*maxIter//5, 4*maxIter//5, maxIter-1]
    
    for iter in range(maxIter):
        trajectory.append(current_best.copy())
        current_fitness = funcion_cec(current_best.reshape(1, -1))[0]
        fitness_traj.append(current_fitness)
        
        # Guardar todas las posiciones
        all_positions.extend(population.copy())
        
        if iter in sample_iterations:
            search_history.append({
                'iter': iter,
                'population': population.copy()
            })
        
        # Evolución: exploración decrece con iteraciones
        noise_factor = (1 - iter/maxIter)
        
        for i in range(pop_size):
            # Movimiento exploratorio
            noise = np.random.randn(dim) * noise_factor * 40
            direction = current_best - population[i]
            population[i] = population[i] + 0.5 * direction + noise
            population[i] = np.clip(population[i], lb, ub)
        
        fitness_pop = funcion_cec(population)
        best_idx = np.argmin(fitness_pop)
        current_best = population[best_idx].copy()
    
    all_positions = np.array(all_positions)
    print(f"   Total de puntos evaluados: {len(all_positions)}")
    print(f"   Fitness final: {fitness_traj[-1]:.2f} (óptimo teórico: {optimo_fitness})")
    
    return (np.array(trajectory), np.array(fitness_traj), 
            search_history, all_positions, lb, ub, dim)


def test_cec2017():
    """Test con funciones CEC2017."""
    print("\n" + "="*70)
    print("🧪 TEST: Visualizaciones con CEC2017")
    print("="*70)
    
    # Probar con funciones simples y híbridas (algunas no están definidas para D=2)
    funciones_test = [
        (cec2017.f1, "Simple"),      # Shifted and Rotated Bent Cigar
        (cec2017.f3, "Simple"),      # Shifted and Rotated Rosenbrock
        (cec2017.f5, "Simple"),      # Shifted and Rotated Rastrigin
        (cec2017.f10, "Hybrid"),     # Hybrid Function 1
    ]
    
    mh = 'GWO'
    
    for funcion, tipo in funciones_test:
        print(f"\n--- {funcion.__name__} ({tipo}) ---")
        
        traj, fit, hist, all_pos, lb, ub, dim = simular_ejecucion_cec2017_2d(
            mh, funcion, maxIter=150, pop_size=30
        )
        
        # Usar el nombre de la función CEC como identificador
        nombre_func = f"CEC2017-{funcion.__name__}"
        
        # Visualizaciones sin evaluar función (solo usar datos generados)
        graficar_trayectoria_dimensional(mh, nombre_func, traj, dim)
        
        # Visualizaciones 2D con paisaje de fondo
        print(f"📊 Graficando trayectoria histórica 2D para {nombre_func}...")
        print("   Evaluando función en grid 2D...")
        
        # Crear grid para el paisaje
        x = np.linspace(lb[0], ub[0], 200)
        y = np.linspace(lb[1], ub[1], 200)
        X, Y = np.meshgrid(x, y)
        
        # Evaluar función CEC2017 en el grid
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                punto = np.array([[X[i, j], Y[i, j]]])
                Z[i, j] = funcion(punto)[0]
        
        # Crear figura para search history manual
        os.makedirs("Graficos/SearchHistory", exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Paisaje de fondo
        contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
        
        # Graficar todas las posiciones con gradiente temporal
        n_points = len(all_pos)
        
        scatter = ax.scatter(all_pos[:, 0], all_pos[:, 1], c=np.linspace(0, 1, n_points), 
                            cmap='plasma', s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.scatter(traj[0, 0], traj[0, 1], c='green', s=200, marker='*', 
                  edgecolors='black', linewidths=2, label='Inicio', zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=200, marker='*', 
                  edgecolors='black', linewidths=2, label='Final', zorder=5)
        
        ax.set_xlabel('Dimensión 1', fontsize=12)
        ax.set_ylabel('Dimensión 2', fontsize=12)
        ax.set_title(f'Search History Completo: {mh} en {nombre_func}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lb[0], ub[0])
        ax.set_ylim(lb[1], ub[1])
        ax.set_aspect('equal', adjustable='box')
        
        # Agregar colorbars más pequeñas
        cbar1 = plt.colorbar(contour, ax=ax, label='Fitness', shrink=0.6, pad=0.02)
        cbar2 = plt.colorbar(scatter, ax=ax, label='Progreso Temporal', shrink=0.6, pad=0.08)
        
        filename = f"Graficos/SearchHistory/{mh}_{nombre_func}_completo.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Guardado: {filename}")
    
    print("\n✅ Test CEC2017 completado")
    print("📁 Gráficos guardados en ./Graficos/")


def simular_ejecucion_cec2017_30d(mh, funcion_cec, maxIter=100, pop_size=30, dim=30):
    """
    Simula una ejecución con funciones CEC2017 en 30D.
    
    Args:
        mh: nombre de la metaheurística
        funcion_cec: función CEC2017
        maxIter: número de iteraciones
        pop_size: tamaño de población
        dim: dimensionalidad
    
    Returns:
        Tupla con (trajectory, fitness_traj, all_positions, best_solution, lb, ub, dim)
    """
    print(f"\n🔄 Simulando ejecución de {mh} en CEC2017-{funcion_cec.__name__} ({dim}D)...")
    
    lb = np.array([-100] * dim)
    ub = np.array([100] * dim)
    
    # El óptimo de CEC2017 es siempre función_number * 100
    func_num = int(funcion_cec.__name__[1:])
    optimo_fitness = func_num * 100
    
    # Población inicial
    population = np.random.uniform(lb, ub, (pop_size, dim))
    
    trajectory = []
    fitness_traj = []
    all_positions = []
    
    # Evaluar con CEC2017
    fitness_pop = funcion_cec(population)
    best_idx = np.argmin(fitness_pop)
    current_best = population[best_idx].copy()
    
    for iter in range(maxIter):
        trajectory.append(current_best.copy())
        current_fitness = funcion_cec(current_best.reshape(1, -1))[0]
        fitness_traj.append(current_fitness)
        
        # Guardar todas las posiciones
        all_positions.extend(population.copy())
        
        # Evolución hacia convergencia
        noise_factor = (1 - iter/maxIter)
        
        for i in range(pop_size):
            # Movimiento exploratorio
            noise = np.random.randn(dim) * noise_factor * 40
            direction = current_best - population[i]
            population[i] = population[i] + 0.5 * direction + noise
            population[i] = np.clip(population[i], lb, ub)
        
        fitness_pop = funcion_cec(population)
        best_idx = np.argmin(fitness_pop)
        current_best = population[best_idx].copy()
    
    all_positions = np.array(all_positions)
    print(f"   Total de puntos evaluados: {len(all_positions)}")
    print(f"   Fitness final: {fitness_traj[-1]:.2f} (óptimo teórico: {optimo_fitness})")
    
    return (np.array(trajectory), np.array(fitness_traj), 
            all_positions, current_best, lb, ub, dim)


def test_cec2017_30d():
    """Test con funciones CEC2017 en 30D."""
    print("\n" + "="*70)
    print("🧪 TEST: Visualizaciones con CEC2017 (30D)")
    print("="*70)
    
    mh = 'PSO'
    
    # Probar varias funciones CEC2017
    funciones_test = [
        # Funciones Simples (f1-f10)
        (cec2017.f1, "Shifted and Rotated Bent Cigar Function"),
        (cec2017.f3, "Shifted and Rotated Zakharov Function"),
        (cec2017.f4, "Shifted and Rotated Rosenbrock's Function"),
        (cec2017.f5, "Shifted and Rotated Rastrigin's Function"),
        (cec2017.f6, "Shifted and Rotated Expanded Scaffer's F6 Function"),
        (cec2017.f7, "Shifted and Rotated Lunacek Bi-Rastrigin Function"),
        (cec2017.f9, "Shifted and Rotated Levy Function"),
        (cec2017.f10, "Shifted and Rotated Schwefel's Function"),
        # Funciones Híbridas (f11-f20)
        (cec2017.f11, "Hybrid Function 1 (N=3)"),
        (cec2017.f14, "Hybrid Function 4 (N=4)"),
        (cec2017.f17, "Hybrid Function 7 (N=5)"),
        # Funciones de Composición (f21-f30)
        (cec2017.f21, "Composition Function 1 (N=3)"),
        (cec2017.f23, "Composition Function 3 (N=4)"),
        (cec2017.f29, "Composition Function 9 (N=3)"),
    ]
    
    for funcion, descripcion in funciones_test:
        print(f"\n--- {funcion.__name__} ({descripcion}, 30D) ---")
    for funcion, descripcion in funciones_test:
        print(f"\n--- {funcion.__name__} ({descripcion}, 30D) ---")
    
        traj, fit, all_pos, best_sol, lb, ub, dim = simular_ejecucion_cec2017_30d(
            mh, funcion, maxIter=150, pop_size=30, dim=30
        )
        
        nombre_func = f"CEC2017-{funcion.__name__}"
        
        # Visualizaciones
        graficar_trayectoria_dimensional(mh, nombre_func, traj, dim)
        
        # Search history con slice 2D usando la función CEC directamente
        print(f"📊 Graficando search history completo 2D para {mh} en {nombre_func} ({dim}D)...")
        print(f"   Total de puntos: {len(all_pos)}")
        print("   Evaluando función en grid 2D puro (sin dimensiones extras)...")
        
        # Crear grid para el paisaje (dims 0 y 1)
        x = np.linspace(lb[0], ub[0], 200)
        y = np.linspace(lb[1], ub[1], 200)
        X, Y = np.meshgrid(x, y)
        
        # Evaluar función en el grid usando SOLO 2D (como graficar_cec2017py.py)
        # Para ver el paisaje canónico de la función, no un slice de 30D
        Z = np.zeros_like(X)
        
        # Determinar dimensión mínima requerida (según graficar_cec2017py.py)
        func_num = int(funcion.__name__[1:])
        # Solo funciones híbridas (f11-f20) y f29, f30 necesitan 10D
        # Las demás (f1-f10, f21-f28) se evalúan en 2D puro
        if func_num in [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 29, 30]:
            dim_eval = 10
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    punto = np.zeros((1, dim_eval))
                    punto[0, 0] = X[i, j]
                    punto[0, 1] = Y[i, j]
                    Z[i, j] = funcion(punto)[0]
        else:  # f1-f10 (simples) y f21-f28 (composición) usan 2D puro
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    punto_2d = np.array([[X[i, j], Y[i, j]]])
                    Z[i, j] = funcion(punto_2d)[0]
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Paisaje de fondo
        contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
        
        # Proyectar puntos a 2D
        positions_2d = all_pos[:, :2]
        
        scatter = ax.scatter(positions_2d[:, 0], positions_2d[:, 1], 
                            c=np.linspace(0, 1, len(positions_2d)), 
                            cmap='plasma', s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.scatter(traj[0, 0], traj[0, 1], c='green', s=200, marker='*', 
                  edgecolors='black', linewidths=2, label='Inicio', zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=200, marker='*', 
                  edgecolors='black', linewidths=2, label='Final', zorder=5)
        
        ax.set_xlabel('Dimensión 1', fontsize=12)
        ax.set_ylabel('Dimensión 2', fontsize=12)
        ax.set_title(f'Search History Completo: {mh} en {nombre_func}\n({len(all_pos)} evaluaciones, proyección dims 0-1 de {dim}D)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lb[0], ub[0])
        ax.set_ylim(lb[1], ub[1])
        ax.set_aspect('equal', adjustable='box')
        
        # Agregar colorbars
        cbar1 = plt.colorbar(contour, ax=ax, label='Fitness', shrink=0.6, pad=0.02)
        cbar2 = plt.colorbar(scatter, ax=ax, label='Progreso Temporal', shrink=0.6, pad=0.08)
        
        os.makedirs("Graficos/SearchHistory", exist_ok=True)
        filename = f"Graficos/SearchHistory/{mh}_{nombre_func}_completo_30D.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Guardado: {filename}")
    
    print("\n✅ Test CEC2017 30D completado")
    print("📁 Gráficos guardados en ./Graficos/")


if __name__ == "__main__":
    print("="*70)
    print("🚀 PRUEBAS DE VISUALIZACIONES (VERSIÓN CORREGIDA V2)")
    print("="*70)
    print("\nCORRECCIONES:")
    print("  ✅ Search history completo (todos los puntos)")
    print("  ✅ Óptimos correctos (F8 = -418.9829*dim, no 0)")
    print("  ✅ Convergencia hacia óptimos reales")
    print("  ✅ Soporte para CEC2017")
    print("="*70)
    
    try:
        # Test principal: Search history completo
        # test_search_history_completo()
        
        # Test trayectoria con óptimos correctos
        # test_trayectoria_dimensional()
        
        # Test completo
        # test_completo()
        
        # Test CEC2017 2D
        # test_cec2017()
        
        # Test CEC2017 30D
        test_cec2017_30d()
        
        print("\n" + "="*70)
        print("✅ TODAS LAS PRUEBAS COMPLETADAS")
        print("="*70)
        print("\n📁 Gráficos guardados en:")
        print("   - ./Graficos/Trayectorias/")
        print("   - ./Graficos/SearchHistory/")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()