import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Scripts.reiniciarDB import reiniciarDB
from Scripts.limpiarEntorno import limpiarEntorno

reiniciarDB()
limpiarEntorno()