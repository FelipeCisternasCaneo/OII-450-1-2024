import os
import sys
import sqlite3

# Permite ejecutar este script directamente (python Scripts/crearBD.py)
# sin errores de imports del proyecto (BD, Util, etc.)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from BD.sqlite import BD

bd = BD()


def _requiere_bootstrap(db_path):
    """Determina si la BD requiere crear tablas y poblar instancias base."""
    if not os.path.exists(db_path):
        return True, "no existe"

    try:
        conn = sqlite3.connect(db_path, timeout=10)
        cur = conn.cursor()

        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='instancias'"
        )
        if cur.fetchone() is None:
            conn.close()
            return True, "falta tabla instancias"

        cur.execute("SELECT COUNT(*) FROM instancias")
        total = int(cur.fetchone()[0])
        conn.close()

        if total == 0:
            return True, "tabla instancias vacia"

        return False, "ok"
    except Exception:
        return True, "error al inspeccionar BD"

def crear_BD():
    db_path = bd.getDataBase()
    bootstrap, motivo = _requiere_bootstrap(db_path)

    if bootstrap:
        print(f"Inicializando base de datos ({motivo}).")
        bd.construirTablas()
        print("Base de datos inicializada exitosamente.")
    else:
        print("Base de datos ya inicializada.")

if __name__ == '__main__':
    crear_BD()