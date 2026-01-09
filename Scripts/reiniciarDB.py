import os
import sys

# Permite ejecutar este script directamente (python Scripts/reiniciarDB.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from BD.sqlite import BD

bd = BD()

def reiniciarDB():
    bd.reiniciarDB()

    print("Base de datos reiniciada exitosamente.")
    
if __name__ == '__main__':
    reiniciarDB()