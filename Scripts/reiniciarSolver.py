import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reiniciarDB import reiniciarDB
from limpiarEntorno import limpiarEntorno

reiniciarDB()
limpiarEntorno()