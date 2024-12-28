from BD.sqlite import BD

bd = BD()

def reiniciarDB():
    bd.reiniciarDB()

    print("Base de datos reiniciada exitosamente.")
    
if __name__ == '__main__':
    reiniciarDB()