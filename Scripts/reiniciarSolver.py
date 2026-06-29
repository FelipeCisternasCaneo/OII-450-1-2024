import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from scripts.reiniciarDB import reiniciarDB
from scripts.limpiarEntorno import limpiarEntorno

reiniciarDB()
limpiarEntorno()