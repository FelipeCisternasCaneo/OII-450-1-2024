import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from joblib import Parallel, delayed
from Problem.Benchmark.CEC.cec2017.cec2017.functions import all_functions

# Directorio de salida
OUTPUT_DIR = './Graficos_Benchmark/CEC2017/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mapeo de nombres de funciones CEC2017
NOMBRES_CEC2017 = {
    'f1': 'Shifted and Rotated Bent Cigar Function',
    'f2': 'Shifted and Rotated Sum of Different Power Function',
    'f3': 'Shifted and Rotated Zakharov Function',
    'f4': 'Shifted and Rotated Rosenbrock\'s Function',
    'f5': 'Shifted and Rotated Rastrigin\'s Function',
    'f6': 'Shifted and Rotated Expanded Scaffer\'s F6 Function',
    'f7': 'Shifted and Rotated Lunacek Bi-Rastrigin Function',
    'f8': 'Shifted and Rotated Non-Continuous Rastrigin\'s Function',
    'f9': 'Shifted and Rotated Levy Function',
    'f10': 'Shifted and Rotated Schwefel\'s Function',
    'f11': 'Hybrid Function 1 (N=3)',
    'f12': 'Hybrid Function 2 (N=3)',
    'f13': 'Hybrid Function 3 (N=3)',
    'f14': 'Hybrid Function 4 (N=4)',
    'f15': 'Hybrid Function 5 (N=4)',
    'f16': 'Hybrid Function 6 (N=4)',
    'f17': 'Hybrid Function 7 (N=5)',
    'f18': 'Hybrid Function 8 (N=5)',
    'f19': 'Hybrid Function 9 (N=5)',
    'f20': 'Hybrid Function 10 (N=6)',
    'f21': 'Composition Function 1 (N=3)',
    'f22': 'Composition Function 2 (N=3)',
    'f23': 'Composition Function 3 (N=4)',
    'f24': 'Composition Function 4 (N=4)',
    'f25': 'Composition Function 5 (N=5)',
    'f26': 'Composition Function 6 (N=5)',
    'f27': 'Composition Function 7 (N=6)',
    'f28': 'Composition Function 8 (N=6)',
    'f29': 'Composition Function 9 (N=3)',
    'f30': 'Composition Function 10 (N=3)',
}

# Óptimos globales conocidos de las funciones CEC2017
# Para CEC2017, el óptimo es f* = f_i(x*) = i * 100, donde i es el número de función
OPTIMOS_GLOBALES_CEC2017 = {
    'f1': 100,      # Bent Cigar
    'f2': 200,      # Sum of Different Power
    'f3': 300,      # Zakharov
    'f4': 400,      # Rosenbrock
    'f5': 500,      # Rastrigin
    'f6': 600,      # Expanded Scaffer's F6
    'f7': 700,      # Lunacek Bi-Rastrigin
    'f8': 800,      # Non-Continuous Rastrigin
    'f9': 900,      # Levy
    'f10': 1000,    # Schwefel
    'f11': 1100,    # Hybrid 1
    'f12': 1200,    # Hybrid 2
    'f13': 1300,    # Hybrid 3
    'f14': 1400,    # Hybrid 4
    'f15': 1500,    # Hybrid 5
    'f16': 1600,    # Hybrid 6
    'f17': 1700,    # Hybrid 7
    'f18': 1800,    # Hybrid 8
    'f19': 1900,    # Hybrid 9
    'f20': 2000,    # Hybrid 10
    'f21': 2100,    # Composition 1
    'f22': 2200,    # Composition 2
    'f23': 2300,    # Composition 3
    'f24': 2400,    # Composition 4
    'f25': 2500,    # Composition 5
    'f26': 2600,    # Composition 6
    'f27': 2700,    # Composition 7
    'f28': 2800,    # Composition 8
    'f29': 2900,    # Composition 9
    'f30': 3000,    # Composition 10
}

def graficar_funcion_cec2017(func, resolucion=200):
    """
    Genera gráficos 2D y 3D para una función CEC2017.
    
    Args:
        func: Función de cec2017py
        resolucion: Número de puntos por dimensión
    """
    nombre = func.__name__
    nombre_completo = NOMBRES_CEC2017.get(nombre, nombre)
    
    file_2d = os.path.join(OUTPUT_DIR, f'{nombre}_2D.pdf')
    file_3d = os.path.join(OUTPUT_DIR, f'{nombre}_3D.pdf')
    
    # Verificar si ya existen
    if os.path.exists(file_2d) and os.path.exists(file_3d):
        return f'[SKIP] {nombre}: Ya existe'
    
    try:
        # CEC2017 tiene rango [-100, 100]
        lb, ub = -100, 100
        
        # CAMBIO: Determinar dimensión mínima requerida
        # Funciones híbridas y de composición necesitan más dimensiones
        if nombre in ['f11', 'f12', 'f13', 'f14', 'f15', 'f16', 
                      'f17', 'f18', 'f19', 'f20', 'f29', 'f30']:
            dimension_total = 10  # Usar 10 dimensiones
        else:
            dimension_total = 2   # Usar 2 dimensiones para las demás
        
        # Crear malla de puntos 2D
        x1 = np.linspace(lb, ub, resolucion)
        x2 = np.linspace(lb, ub, resolucion)
        X1, X2 = np.meshgrid(x1, x2)
        
        # Evaluar la función
        Z = np.zeros_like(X1)
        
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                # CAMBIO: Crear vector con dimensión apropiada
                if dimension_total == 2:
                    x = np.array([[X1[i, j], X2[i, j]]])
                else:
                    # Para funciones de alta dimensión, fijar las demás en 0
                    x = np.zeros((1, dimension_total))
                    x[0, 0] = X1[i, j]
                    x[0, 1] = X2[i, j]
                
                Z[i, j] = func(x)[0]
        
        # Limpiar valores infinitos/NaN
        Z_valid = Z[np.isfinite(Z)]
        if len(Z_valid) == 0:
            return f'[ERROR] {nombre}: Todos los valores son infinitos/NaN'
        
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
            
            # Título general con nombre completo
            titulo = f'CEC2017 {nombre.upper()}: {nombre_completo}'
            fig.suptitle(f'CEC2017 {nombre.upper()}: {nombre_completo}', fontsize=13, fontweight='bold')
            
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
            
            # CAMBIO: Actualizar info_text para indicar dimensión
            dim_text = f"{dimension_total}D (visualizando 2D slice)" if dimension_total > 2 else "2D"
            optimo_global = OPTIMOS_GLOBALES_CEC2017.get(nombre, 'Desconocido')
            
            info_text = f"""
CEC2017 - {nombre.upper()}

Nombre:
  {nombre_completo}

Parámetros:
  • Rango: [{lb}, {ub}]
  • Resolución: {resolucion}×{resolucion}
  • Dimensión: {dim_text}
  • Óptimo global: {optimo_global:>12.1f}

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
            
            # Título general con nombre completo
            titulo = f'CEC2017 {nombre.upper()}: {nombre_completo}'
            fig.suptitle(f'CEC2017 {nombre.upper()}: {nombre_completo}', fontsize=13, fontweight='bold')
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(file_3d, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        return f'[OK] {nombre}'
        
    except Exception as e:
        plt.close('all')
        return f'[ERROR] {nombre}: {str(e)}'

def generar_todos_graficos(n_jobs=-1, resolucion=200):
    """
    Genera gráficos para todas las funciones CEC2017.
    
    Args:
        n_jobs: Número de trabajos paralelos (-1 = todos los cores)
        resolucion: Calidad de los gráficos (150-300)
    """
    print("=" * 80)
    print("GENERADOR DE GRÁFICOS CEC2017 (cec2017py)")
    print("=" * 80)
    
    funciones = all_functions
    
    print(f"\n[INFO] Total de funciones CEC2017: {len(funciones)}")
    print(f"[INFO] Resolución: {resolucion}x{resolucion} puntos")
    print(f"[INFO] Directorio de salida: {OUTPUT_DIR}")
    
    # Listar funciones con nombres completos
    print(f"\n[INFO] Funciones a graficar:")
    for i, func in enumerate(funciones, 1):
        nombre = func.__name__
        nombre_completo = NOMBRES_CEC2017.get(nombre, nombre)
        print(f"  {i:2d}. {nombre:4s} - {nombre_completo}")
    
    print("\n" + "-" * 80)
    print("Generando gráficos...")
    print("-" * 80 + "\n")
    
    # Generar en paralelo
    resultados = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(graficar_funcion_cec2017)(func, resolucion) 
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
    funciones = all_functions
    
    for i, func in enumerate(funciones, 1):
        nombre = func.__name__
        nombre_completo = NOMBRES_CEC2017.get(nombre, nombre)
        print(f"[{i}/{len(funciones)}] {nombre} - {nombre_completo[:50]}")
        resultado = graficar_funcion_cec2017(func, resolucion)
        print(f"  → {resultado}\n")

def main():
    """Punto de entrada principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generador de gráficos para funciones CEC2017 (cec2017py)',
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