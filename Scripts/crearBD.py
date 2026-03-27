import os
import sys

# Permite ejecutar este script directamente (python Scripts/crearBD.py)
# sin errores de imports del proyecto (BD, Util, etc.)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from BD.sqlite import BD

bd = BD()

def crear_BD():
    if not os.path.exists(bd.getDataBase()):
        print("La base de datos no existe, se procederá a crearla.")
        bd.construirTablas()
        print("Base de datos creada exitosamente.")

if __name__ == '__main__':
    crear_BD()