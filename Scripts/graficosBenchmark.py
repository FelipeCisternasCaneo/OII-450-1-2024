import sys
import os

# Agregar el directorio raíz al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed
from Problem.Benchmark.Problem import fitness as f

# Directorio de salida
OUTPUT_DIR = './Graficos_Benchmark/Classical/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Límites de las variables por función
LIMITES = {
    'F1': (-100, 100), 'F2': (-10, 10), 'F3': (-100, 100), 'F4': (-100, 100),
    'F5': (-20, 20), 'F6': (-100, 100), 'F7': (-1, 1), 'F8': (-500, 500),
    'F9': (-5, 5), 'F10': (-20, 20), 'F11': (-600, 600), 'F12': (-10, 10),
    'F13': (-5, 5), 'F14': (-50, 50), 'F15': (-5, 5), 'F16': (-1, 1),
    'F17': (-15, 15), 'F18': (-5, 5), 'F19': (-5, 5), 'F20': (-5, 5),
    'F21': (-5, 5), 'F22': (-5, 5), 'F23': (-5, 5)
}

# Nombres completos de las funciones
NOMBRES_FUNCIONES = {
    'F1': 'Sphere Function',
    'F2': 'Schwefel 2.22 Function',
    'F3': 'Schwefel 1.2 Function',
    'F4': 'Schwefel 2.21 Function',
    'F5': 'Rosenbrock Function',
    'F6': 'Step Function',
    'F7': 'Quartic Function with Noise',
    'F8': 'Schwefel Function',
    'F9': 'Rastrigin Function',
    'F10': 'Ackley Function',
    'F11': 'Griewank Function',
    'F12': 'Generalized Penalized Function 1',
    'F13': 'Generalized Penalized Function 2',
    'F14': 'Shekel\'s Foxholes Function',
    'F15': 'Kowalik Function',
    'F16': 'Six-Hump Camel-Back Function',
    'F17': 'Branin Function',
    'F18': 'Goldstein-Price Function',
    'F19': 'Hartman Function 3',
    'F20': 'Hartman Function 6',
    'F21': 'Shekel Function 5',
    'F22': 'Shekel Function 7',
    'F23': 'Shekel Function 10'
}

# Configuración de dimensiones por función
DIMENSIONES_EXTRA = {
    'F15': 4,  # Requiere 4 dimensiones
    'F19': 3,  # Requiere 3 dimensiones
    'F20': 6,  # Requiere 6 dimensiones
    'F21': 4,  # Requiere 4 dimensiones
    'F22': 4,  # Requiere 4 dimensiones
    'F23': 4   # Requiere 4 dimensiones
}

# Óptimos globales conocidos de las funciones
OPTIMOS_GLOBALES = {
    'F1': 0.0,           # Sphere
    'F2': 0.0,           # Schwefel 2.22
    'F3': 0.0,           # Schwefel 1.2
    'F4': 0.0,           # Schwefel 2.21
    'F5': 0.0,           # Rosenbrock
    'F6': 0.0,           # Step
    'F7': 0.0,           # Quartic
    'F8': -418.9829 * 2, # Schwefel (2D)
    'F9': 0.0,           # Rastrigin
    'F10': 0.0,          # Ackley
    'F11': 0.0,          # Griewank
    'F12': 0.0,          # Penalized 1
    'F13': 0.0,          # Penalized 2
    'F14': 1.0,          # Shekel's Foxholes
    'F15': 0.0003075,    # Kowalik
    'F16': -1.0316,      # Six-Hump Camel
    'F17': 0.398,        # Branin
    'F18': 3.0,          # Goldstein-Price
    'F19': -3.86,        # Hartman 3
    'F20': -3.32,        # Hartman 6
    'F21': -10.1532,     # Shekel 5
    'F22': -10.4028,     # Shekel 7
    'F23': -10.5363      # Shekel 10
}

def graficar_funcion_benchmark(funcion, resolucion=200):
    """
    Genera gráficos 2D y 3D para una función benchmark.
    
    Args:
        funcion: Nombre de la función (e.g., 'F1', 'F2')
        resolucion: Número de puntos por dimensión
    """
    nombre_completo = NOMBRES_FUNCIONES.get(funcion, funcion)
    
    file_2d = os.path.join(OUTPUT_DIR, f'{funcion}_2D.pdf')
    file_3d = os.path.join(OUTPUT_DIR, f'{funcion}_3D.pdf')
    
    # Verificar si ya existen
    if os.path.exists(file_2d) and os.path.exists(file_3d):
        return f'[SKIP] {funcion}: Ya existe'
    
    try:
        # Obtener límites
        lb, ub = LIMITES[funcion]
        
        # Crear malla de puntos 2D
        x1 = np.linspace(lb, ub, resolucion)
        x2 = np.linspace(lb, ub, resolucion)
        X1, X2 = np.meshgrid(x1, x2)
        
        # Evaluar la función
        Z = np.zeros_like(X1)
        dim_total = DIMENSIONES_EXTRA.get(funcion, 2)
        
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                # Crear vector con dimensión apropiada
                if dim_total == 2:
                    x = np.array([X1[i, j], X2[i, j]])
                else:
                    x = np.zeros(dim_total)
                    x[0] = X1[i, j]
                    x[1] = X2[i, j]
                
                Z[i, j] = f(funcion, x)
        
        # Limpiar valores infinitos/NaN
        Z_valid = Z[np.isfinite(Z)]
        if len(Z_valid) == 0:
            return f'[ERROR] {funcion}: Todos los valores son infinitos/NaN'
        
        Z_mean = np.mean(Z_valid)
        Z_max = np.percentile(Z_valid, 99)
        Z_min = np.percentile(Z_valid, 1)
        Z = np.nan_to_num(Z, nan=Z_mean, posinf=Z_max, neginf=Z_min)
        
        # ============================================
        # GRÁFICO 2D (Contorno)
        # ============================================
        if not os.path.exists(file_2d):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Subplot 1: Contorno relleno
            contour = ax1.contourf(X1, X2, Z, levels=30, cmap='viridis')
            ax1.contour(X1, X2, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
            ax1.set_xlabel('x₁', fontsize=12)
            ax1.set_ylabel('x₂', fontsize=12)
            ax1.set_title('Curvas de nivel', fontsize=13)
            ax1.grid(True, alpha=0.3)
            fig.colorbar(contour, ax=ax1)
            
            # Subplot 2: Heatmap
            heatmap = ax2.imshow(Z, extent=[lb, ub, lb, ub], 
                                 origin='lower', cmap='viridis', 
                                 aspect='auto', interpolation='bilinear')
            ax2.set_xlabel('x₁', fontsize=12)
            ax2.set_ylabel('x₂', fontsize=12)
            ax2.set_title('Mapa de calor', fontsize=13)
            fig.colorbar(heatmap, ax=ax2)
            
            # Título general
            titulo = f'{funcion}: {nombre_completo}'
            fig.suptitle(titulo, fontsize=13, fontweight='bold')
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(file_2d, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        # ============================================
        # GRÁFICO 3D
        # ============================================
        if not os.path.exists(file_3d):
            fig = plt.figure(figsize=(14, 10))
            
            # Vista 3D principal
            ax1 = fig.add_subplot(221, projection='3d')
            surf = ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.9,
                                   edgecolor='none', antialiased=True)
            
            # Proyección en plano XY
            z_offset = np.min(Z) - (np.max(Z) - np.min(Z)) * 0.1
            ax1.contourf(X1, X2, Z, zdir='z', offset=z_offset, 
                        cmap='viridis', alpha=0.6, levels=20)
            
            ax1.set_xlabel('x₁', fontsize=10)
            ax1.set_ylabel('x₂', fontsize=10)
            ax1.set_zlabel('f(x)', fontsize=10)
            ax1.set_title('Vista 3D con proyección', fontsize=11)
            ax1.set_zlim(z_offset, np.max(Z))
            fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
            
            # Vista desde arriba
            ax2 = fig.add_subplot(222, projection='3d')
            ax2.plot_surface(X1, X2, Z, cmap='plasma', alpha=0.9)
            ax2.view_init(elev=90, azim=0)
            ax2.set_xlabel('x₁', fontsize=10)
            ax2.set_ylabel('x₂', fontsize=10)
            ax2.set_title('Vista superior (XY)', fontsize=11)
            
            # Vista lateral
            ax3 = fig.add_subplot(223, projection='3d')
            ax3.plot_surface(X1, X2, Z, cmap='coolwarm', alpha=0.9)
            ax3.view_init(elev=0, azim=0)
            ax3.set_xlabel('x₁', fontsize=10)
            ax3.set_zlabel('f(x)', fontsize=10)
            ax3.set_title('Vista lateral (XZ)', fontsize=11)
            
            # Información estadística
            ax4 = fig.add_subplot(224)
            ax4.axis('off')
            
            dim_text = f"{dim_total}D (visualizando 2D slice)" if dim_total > 2 else "2D"
            optimo_global = OPTIMOS_GLOBALES.get(funcion, 'Desconocido')
            optimo_str = f"{optimo_global:>12.4e}" if isinstance(optimo_global, (int, float)) else optimo_global
            
            info_text = f"""
{funcion}

Nombre:
  {nombre_completo}

Parámetros:
  • Rango: [{lb}, {ub}]
  • Resolución: {resolucion}×{resolucion}
  • Dimensión: {dim_text}
  • Óptimo global: {optimo_str}

Estadísticas (slice 2D evaluado):
  • Mínimo:     {np.min(Z):>12.4e}
  • Máximo:     {np.max(Z):>12.4e}
  • Media:      {np.mean(Z):>12.4e}
  • Desv. std:  {np.std(Z):>12.4e}
  • Mediana:    {np.median(Z):>12.4e}
  • Rango:      {np.max(Z) - np.min(Z):>12.4e}
            """
            
            ax4.text(0.05, 0.5, info_text.strip(), fontsize=9,
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            # Título general
            titulo = f'{funcion}: {nombre_completo}'
            fig.suptitle(titulo, fontsize=13, fontweight='bold')
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(file_3d, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        return f'[OK] {funcion}'
        
    except Exception as e:
        plt.close('all')
        return f'[ERROR] {funcion}: {str(e)}'

def generar_todos_graficos(n_jobs=-1, resolucion=200):
    """
    Genera gráficos para todas las funciones benchmark.
    
    Args:
        n_jobs: Número de trabajos paralelos (-1 = todos los cores)
        resolucion: Calidad de los gráficos (150-300)
    """
    print("=" * 80)
    print("GENERADOR DE GRÁFICOS BENCHMARK CLÁSICOS")
    print("=" * 80)
    
    funciones = list(NOMBRES_FUNCIONES.keys())
    
    print(f"\n[INFO] Total de funciones: {len(funciones)}")
    print(f"[INFO] Resolución: {resolucion}x{resolucion} puntos")
    print(f"[INFO] Directorio de salida: {OUTPUT_DIR}")
    
    # Listar funciones con nombres completos
    print(f"\n[INFO] Funciones a graficar:")
    for i, func in enumerate(funciones, 1):
        nombre_completo = NOMBRES_FUNCIONES[func]
        lb, ub = LIMITES[func]
        print(f"  {i:2d}. {func:4s} [{lb:>4}, {ub:>4}] - {nombre_completo}")
    
    print("\n" + "-" * 80)
    print("Generando gráficos...")
    print("-" * 80 + "\n")
    
    # Generar en paralelo
    resultados = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(graficar_funcion_benchmark)(func, resolucion) 
        for func in funciones
    )
    
    # Resumen
    print("\n" + "=" * 80)
    print("RESUMEN DE GENERACIÓN")
    print("=" * 80)
    
    exitosos = sum(1 for r in resultados if '[OK]' in r)
    saltados = sum(1 for r in resultados if '[SKIP]' in r)
    errores = sum(1 for r in resultados if '[ERROR]' in r)
    
    print(f"\n✓ Gráficos creados: {exitosos}")
    print(f"⊘ Saltados (ya existen): {saltados}")
    print(f"✗ Errores: {errores}")
    
    if errores > 0:
        print("\n[ERRORES ENCONTRADOS]:")
        for r in resultados:
            if '[ERROR]' in r:
                print(f"  {r}")
    
    print(f"\n[INFO] Total de archivos: {len(funciones) * 2} (2D + 3D por función)")
    print(f"[INFO] Ubicación: {os.path.abspath(OUTPUT_DIR)}")
    print("\n" + "=" * 80)

def generar_secuencial(resolucion=200):
    """Modo secuencial para debugging."""
    print("[INFO] Modo secuencial (debugging)\n")
    funciones = list(NOMBRES_FUNCIONES.keys())
    
    for i, func in enumerate(funciones, 1):
        nombre_completo = NOMBRES_FUNCIONES[func]
        print(f"[{i}/{len(funciones)}] {func} - {nombre_completo}")
        resultado = graficar_funcion_benchmark(func, resolucion)
        print(f"  → {resultado}\n")

def main():
    """Punto de entrada principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generador de gráficos para funciones benchmark clásicas',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--secuencial', action='store_true',
                       help='Ejecutar en modo secuencial (para debugging)')
    parser.add_argument('--jobs', type=int, default=-1,
                       help='Número de trabajos paralelos (default: -1 = todos los cores)')
    parser.add_argument('--resolucion', type=int, default=200,
                       help='Resolución de la malla (default: 200)')
    
    args = parser.parse_args()
    
    if args.secuencial:
        generar_secuencial(resolucion=args.resolucion)
    else:
        generar_todos_graficos(n_jobs=args.jobs, resolucion=args.resolucion)

if __name__ == '__main__':
    main()
