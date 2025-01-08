import os

from BD.sqlite import BD

bd = BD()

def crear_BD():
    if not os.path.exists('./BD/resultados.db'):
        print("La base de datos no existe, se procederá a crearla.")
        bd.construirTablas()
        print("Base de datos creada exitosamente.")

if __name__ == '__main__':
    crear_BD()