import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from solver.runner import ejecutar_pipeline

def main():
    """Punto de entrada principal para ejecutar el solucionador secuencial."""
    ejecutar_pipeline()

if __name__ == "__main__":
    main()
