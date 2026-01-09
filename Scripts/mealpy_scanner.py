import os
import sys

# Permite ejecutar este script directamente (python Scripts/mealpy_scanner.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pkgutil
import inspect
import importlib
import re
import mealpy.swarm_based as swarm
import mealpy.evolutionary_based as evo
import mealpy.physics_based as phys
import mealpy.human_based as human
import mealpy.math_based as math_mod
from mealpy.optimizer import Optimizer

def limpiar_nombre(docstring):
    """
    Extrae el nombre real del algoritmo limpiando la "basura" 
    típica de los docstrings de Mealpy (prefijos, paréntesis, etc).
    """
    if not docstring:
        return "--- Sin descripción ---"
    
    # 1. Obtener la primera línea útil (ignorando parámetros de Sphinx que empiezan con :)
    lines = docstring.strip().split('\n')
    text = next((line.strip() for line in lines if line.strip() and not line.strip().startswith(":")), "---")

    # 2. Limpieza con Expresiones Regulares (Regex)
    
    # Eliminar frases como "The original version of:", "The developed version of:", etc.
    # (?i) hace que sea insensible a mayúsculas/minúsculas
    text = re.sub(r'(?i)^The\s+(original|developed|modified|fully tuned|simple)\s+version\s+(of|:)*\s*:?\s*', '', text)
    
    # A veces el texto empieza directo con "version of:" por errores de tipografía en la lib
    text = re.sub(r'(?i)^version of:\s*', '', text)

    # Eliminar el acrónimo repetido entre paréntesis al final (ej: "Grey Wolf Optimizer (GWO)") -> Se borra (GWO)
    text = re.sub(r'\s*\([A-Za-z0-9\-\s]+\)$', '', text)

    return text.strip()

def listar_detallado(paquete, nombre_categoria):
    print(f"\n{'='*85}")
    print(f" {nombre_categoria.upper()}")
    print(f"{'='*85}")
    # Ajustamos un poco el ancho de columnas para nombres largos
    print(f"{'ACRÓNIMO':<12} | {'CLASE PYTHON':<22} | {'NOMBRE REAL DEL ALGORITMO'}")
    print(f"{'-'*12} | {'-'*22} | {'-'*45}")

    modulos = [name for _, name, _ in pkgutil.iter_modules(paquete.__path__)]
    
    # Ordenamos alfabéticamente por acrónimo para que sea más fácil de buscar
    modulos.sort()

    for mod_name in modulos:
        try:
            # 1. Importación dinámica
            ruta = f"{paquete.__name__}.{mod_name}"
            modulo = importlib.import_module(ruta)
            
            # 2. Buscar clases Optimizer
            clases_encontradas = []
            for name, obj in inspect.getmembers(modulo):
                if inspect.isclass(obj) and issubclass(obj, Optimizer) and obj.__module__ == ruta:
                    clases_encontradas.append(obj)
            
            if not clases_encontradas:
                continue
                
            # 3. Heurística para encontrar la clase principal (Original > Base > La que sea)
            clase_principal = next((c for c in clases_encontradas if c.__name__.startswith("Original")), None)
            if not clase_principal:
                 # A veces se llaman "BaseGWO" o simplemente "GWO"
                clase_principal = next((c for c in clases_encontradas if c.__name__.startswith("Base")), clases_encontradas[0])

            # 4. Extraer y limpiar info
            nombre_clase = clase_principal.__name__
            full_name = limpiar_nombre(clase_principal.__doc__)
            
            # Recortar solo si es excesivamente largo (estético)
            if len(full_name) > 50:
                full_name = full_name[:47] + "..."

            print(f"{mod_name.upper():<12} | {nombre_clase:<22} | {full_name}")

        except Exception:
            pass

# --- EJECUCIÓN ---
print("GENERANDO REPORTE LIMPIO DE ALGORITMOS MEALPY...")
listar_detallado(swarm, "Swarm Based (Enjambres)")
listar_detallado(evo, "Evolutionary Based (Evolutivos)")
listar_detallado(phys, "Physics Based (Físicos)")
listar_detallado(human, "Human Based (Sociales)")
listar_detallado(math_mod, "Math Based (Matemáticos)")