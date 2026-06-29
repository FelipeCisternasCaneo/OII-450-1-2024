import os
import sys

# Permite ejecutar este script directamente (python Scripts/analisis.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# Agregar la carpeta de scripts al path para resolución del paquete de análisis
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analisis_pkg.processing import main

if __name__ == "__main__":
    main()
