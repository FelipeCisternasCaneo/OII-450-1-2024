"""
Funciones de visualización para análisis de metaheurísticas.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from Problem.Benchmark.Problem import fitness as f


def obtener_optimo(funcion):
    """
    Retorna el óptimo teórico de cada función.
    
    Returns:
        np.array: Vector del óptimo global
    """
    # Funciones clásicas (F1-F23)
    optimos_clasicas = {
        'F1': 0,      # Sphere
        'F2': 0,      # Schwefel 2.22
        'F3': 0,      # Schwefel 1.2
        'F4': 0,      # Schwefel 2.21
        'F5': 0,      # Rosenbrock
        'F6': 0,      # Step
        'F7': 0,      # Quartic (con ruido)
        'F8': -418.9829 * 2,  # Schwefel 2.26 (dim=2: -837.9658)
        'F9': 0,      # Rastrigin
        'F10': 0,     # Ackley
        'F11': 0,     # Griewank
        'F12': 0,     # Penalized 1
        'F13': 0,     # Penalized 2
        'F14': 0.998, # Shekel's Foxholes
        'F15': 0.0003075,  # Kowalik
        'F16': -1.0316,    # Six-Hump Camel-Back
        'F17': 0.398,      # Branin
        'F18': 3,          # Goldstein-Price
        'F19': -3.86,      # Hartman 3
        'F20': -3.32,      # Hartman 6
        'F21': -10.1532,   # Shekel 5
        'F22': -10.4028,   # Shekel 7
        'F23': -10.5363,   # Shekel 10
    }
    
    if funcion in optimos_clasicas:
        return optimos_clasicas[funcion]
    
    # CEC2017: todos tienen fitness óptimo en 100*i (i = número de función)
    if funcion.startswith('CEC2017_F'):
        func_num = int(funcion.split('_F')[1])
        return 100 * func_num
    
    return 0  # Default


def graficar_trayectoria_dimensional(mh, funcion, trajectory, dim):
    """
    Grafica la evolución de cada dimensión del mejor individuo a lo largo del tiempo.
    
    Args:
        mh: Nombre de la metaheurística
        funcion: Nombre de la función objetivo
        trajectory: array [n_iterations, dim] - Mejor solución en cada iteración
        dim: Número de dimensiones
    """
    print(f" Graficando trayectoria dimensional para {mh} en {funcion} ({dim}D)...")
    
    iterations = np.arange(len(trajectory))
    
    # Graficar solo la primera dimensión
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(iterations, trajectory[:, 0], linewidth=1.5, color='C0')
    ax.set_ylabel('Dimensión 1', fontsize=10)
    ax.set_xlabel('Iteración', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    ax.set_title(f'Trayectoria Dimensional - {mh} en {funcion} (Dim 1)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    os.makedirs('./Graficos/Trayectorias', exist_ok=True)
    plt.savefig(f'./Graficos/Trayectorias/{mh}_{funcion}_trajectory_dims.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Guardado: ./Graficos/Trayectorias/{mh}_{funcion}_trajectory_dims.png")


def graficar_search_history_2d_completo(mh, funcion, all_positions, lb, ub, best_solution=None, dim=2):
    """
    Grafica TODOS los puntos visitados por la población a lo largo de toda la ejecución.
    Versión papers: muestra el historial completo de exploración.
    Soporta alta dimensión mediante slice 2D.
    
    Args:
        mh: Nombre de la metaheurística
        funcion: Nombre de la función objetivo
        all_positions: array [total_evaluations, dim] - TODAS las posiciones evaluadas
        lb, ub: Límites de la función (arrays de tamaño dim)
        best_solution: array [dim] - Mejor solución (usado como base para slice en alta dim)
        dim: Número de dimensiones
    """
    print(f" Graficando search history completo 2D para {mh} en {funcion} ({dim}D)...")
    print(f"   Total de puntos: {len(all_positions)}")
    
    # Crear grid para el paisaje (siempre en dimensiones 0 y 1)
    x = np.linspace(lb[0], ub[0], 200)
    y = np.linspace(lb[1], ub[1], 200)
    X, Y = np.meshgrid(x, y)
    
    # Evaluar función en el grid
    print("   Evaluando función en grid 2D...")
    Z = np.zeros_like(X)
    
    if dim == 2:
        # Caso 2D normal
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = f(funcion, np.array([X[i, j], Y[i, j]]))
    else:
        # Caso alta dimensión: fijar dimensiones extras en 0 (como en graficar_cec2017py.py)
        # Esto permite que la función CEC2017 aplique su propia transformación (shift + rotación)
        base_point = np.zeros(dim)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                base_point[0] = X[i, j]  # Variar dimensión 0
                base_point[1] = Y[i, j]  # Variar dimensión 1
                # Dimensiones 2...dim-1 se mantienen fijas en 0
                Z[i, j] = f(funcion, base_point)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Paisaje de fondo
    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
    cbar = plt.colorbar(contour, ax=ax, label='Fitness')
    
    # Proyectar puntos a 2D (solo dimensiones 0 y 1)
    positions_2d = all_positions[:, :2]
    
    # TODOS los puntos visitados
    # Usar colormap para mostrar orden temporal (del más antiguo al más reciente)
    scatter = ax.scatter(positions_2d[:, 0], positions_2d[:, 1], 
                        c=np.arange(len(positions_2d)),  # Color por orden temporal
                        cmap='plasma', s=20, alpha=0.6, edgecolors='black', 
                        linewidth=0.5, zorder=5)
    
    cbar2 = plt.colorbar(scatter, ax=ax, label='Orden Temporal')
    
    titulo = f'Search History Completo - {mh} en {funcion}\n({len(all_positions)} evaluaciones'
    if dim > 2:
        titulo += f', proyección dims 0-1 de {dim}D'
    titulo += ')'
    
    ax.set_xlabel('Dimensión 1', fontsize=12)
    ax.set_ylabel('Dimensión 2', fontsize=12)
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    ax.set_xlim(lb[0], ub[0])
    ax.set_ylim(lb[1], ub[1])
    
    plt.tight_layout()
    
    os.makedirs('./Graficos/SearchHistory', exist_ok=True)
    plt.savefig(f'./Graficos/SearchHistory/{mh}_{funcion}_search_history_complete.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Guardado: ./Graficos/SearchHistory/{mh}_{funcion}_search_history_complete.png")


def graficar_search_history_2d(mh, funcion, search_history, lb, ub):
    """
    Grafica el historial de búsqueda de TODA la población en función 2D.
    Muestra snapshots de la población en diferentes iteraciones sobre el paisaje.
    
    Args:
        mh: Nombre de la metaheurística
        funcion: Nombre de la función objetivo
        search_history: Lista de dicts con {'iter': int, 'population': array[pop_size, 2]}
        lb, ub: Límites de la función
    """
    print(f" Graficando search history 2D (snapshots) para {mh} en {funcion}...")
    
    # Crear grid para el paisaje
    x = np.linspace(lb[0], ub[0], 200)
    y = np.linspace(lb[1], ub[1], 200)
    X, Y = np.meshgrid(x, y)
    
    # Evaluar función en el grid
    print("   Evaluando función en grid 2D...")
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(funcion, np.array([X[i, j], Y[i, j]]))
    
    # Determinar número de subplots (máximo 6)
    n_snapshots = min(len(search_history), 6)
    rows = 2
    cols = 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx in range(n_snapshots):
        ax = axes[idx]
        snapshot = search_history[idx]
        
        # Paisaje de fondo
        contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
        
        # Población en esta iteración
        pop = snapshot['population']
        ax.scatter(pop[:, 0], pop[:, 1], 
                  c='red', s=60, alpha=0.8, edgecolors='black', linewidth=1, zorder=5)
        
        ax.set_title(f'Iteración {snapshot["iter"]}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Dimensión 1', fontsize=10)
        ax.set_ylabel('Dimensión 2', fontsize=10)
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
        ax.set_xlim(lb[0], ub[0])
        ax.set_ylim(lb[1], ub[1])
    
    # Ocultar subplots vacíos
    for idx in range(n_snapshots, len(axes)):
        axes[idx].axis('off')
    
    # Colorbar compartido
    fig.colorbar(contour, ax=axes, label='Fitness', fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Search History (Snapshots) - {mh} en {funcion}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    os.makedirs('./Graficos/SearchHistory', exist_ok=True)
    plt.savefig(f'./Graficos/SearchHistory/{mh}_{funcion}_search_history_snapshots.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Guardado: ./Graficos/SearchHistory/{mh}_{funcion}_search_history_snapshots.png")


def graficar_search_history_slice(mh, funcion, search_history, dim1, dim2, lb, ub, dim):
    """
    Grafica el historial de búsqueda proyectado en 2 dimensiones específicas.
    Para funciones de alta dimensión.
    
    Args:
        dim1, dim2: Índices de las dimensiones a visualizar
        Resto de dimensiones se fijan en el valor promedio del último snapshot
    """
    print(f" Graficando search history slice [{dim1}, {dim2}] para {mh} en {funcion}...")
    
    # Valores fijos para otras dimensiones (usar población final promedio)
    last_pop = search_history[-1]['population']
    fixed_values = np.mean(last_pop, axis=0)
    
    # Crear grid solo para dim1 y dim2
    x = np.linspace(lb[dim1], ub[dim1], 150)
    y = np.linspace(lb[dim2], ub[dim2], 150)
    X, Y = np.meshgrid(x, y)
    
    # Evaluar función en el grid (fijando otras dimensiones)
    print(f"   Evaluando función en slice [dim{dim1}, dim{dim2}]...")
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            punto = fixed_values.copy()
            punto[dim1] = X[i, j]
            punto[dim2] = Y[i, j]
            Z[i, j] = f(funcion, punto)
    
    # Determinar número de subplots
    n_snapshots = min(len(search_history), 6)
    rows = 2
    cols = 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx in range(n_snapshots):
        ax = axes[idx]
        snapshot = search_history[idx]
        
        # Paisaje de fondo
        contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
        
        # Población proyectada a las 2 dimensiones
        pop = snapshot['population']
        ax.scatter(pop[:, dim1], pop[:, dim2], 
                  c='red', s=60, alpha=0.8, edgecolors='black', linewidth=1, zorder=5)
        
        ax.set_title(f'Iteración {snapshot["iter"]}', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'Dimensión {dim1+1}', fontsize=10)
        ax.set_ylabel(f'Dimensión {dim2+1}', fontsize=10)
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
        ax.set_xlim(lb[dim1], ub[dim1])
        ax.set_ylim(lb[dim2], ub[dim2])
    
    # Ocultar subplots vacíos
    for idx in range(n_snapshots, len(axes)):
        axes[idx].axis('off')
    
    # Colorbar compartido
    fig.colorbar(contour, ax=axes, label='Fitness', fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Search History Slice [dim{dim1+1}, dim{dim2+1}] - {mh} en {funcion} ({dim}D)', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    os.makedirs('./Graficos/SearchHistory', exist_ok=True)
    plt.savefig(f'./Graficos/SearchHistory/{mh}_{funcion}_search_slice_{dim1}_{dim2}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Guardado: ./Graficos/SearchHistory/{mh}_{funcion}_search_slice_{dim1}_{dim2}.png")
