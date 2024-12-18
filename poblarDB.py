from BD.sqlite import BD

import json

bd = BD()

ben = True
scp = True
uscp = False

mhs = ['SBOA']
cantidad = 0

# Mapeo de dimensiones según la función
dimensiones_funciones = {
    'F1': [30], 'F2': [30], 'F3': [30], 'F4': [30], 'F5': [30],
    'F6': [30], 'F7': [30], 'F8': [30], 'F9': [30], 'F10': [30],
    'F11': [30], 'F12': [30], 'F13': [30], 'F14': [2],
    'F15': [4], 'F16': [2], 'F17': [2], 'F18': [2],
    'F19': [3], 'F20': [6], 'F21': [4], 'F22': [4], 'F23': [4]
}

DS_actions = [
    'V1-STD', 'V1-COM', 'V1-PS', 'V1-ELIT',
    'V2-STD', 'V2-COM', 'V2-PS', 'V2-ELIT',
    'V3-STD', 'V3-COM', 'V3-PS', 'V3-ELIT',
    'V4-STD', 'V4-COM', 'V4-PS', 'V4-ELIT',
    'S1-STD', 'S1-COM', 'S1-PS', 'S1-ELIT',
    'S2-STD', 'S2-COM', 'S2-PS', 'S2-ELIT',
    'S3-STD', 'S3-COM', 'S3-PS', 'S3-ELIT',
    'S4-STD', 'S4-COM', 'S4-PS', 'S4-ELIT',
]

def insertar_experimentos(instancias, dimensiones, mhs, experimentos, iteraciones, poblacion, extra_params = ""):
    global cantidad
    
    for instancia in instancias:
        for dim in dimensiones:
            for mh in mhs:
                data = {
                    'experimento': f'{instancia[1]} {dim} {mh}',
                    'MH': mh,
                    'paramMH': f'iter:{iteraciones},pop:{poblacion}{extra_params}',
                    'ML': '',
                    'paramML': '',
                    'ML_FS': '',
                    'paramML_FS': '',
                    'estado': 'pendiente'
                }
                
                cantidad += experimentos
                bd.insertarExperimentos(data, experimentos, instancia[0]) #

if ben:
    funciones = ['F5', 'F6', 'F8']  # Ejemplo: 'F1', 'F2', 'F3', 'F4'
    
    iteraciones = 100
    experimentos = 2
    poblacion = 50
    
    for funcion in funciones:
        instancias = bd.obtenerInstancias(f'''"{funcion}"''')
        dimensiones = dimensiones_funciones.get(funcion, [30])  # Default a [30] si no se encuentra la función
        insertar_experimentos(instancias, dimensiones, mhs, experimentos, iteraciones, poblacion)

if scp:
    instancias = bd.obtenerInstancias(f'''"41", "42"''') # Ejemplo: "41", "42", "43", "44"
    
    binarizaciones = ['V3-STD']
    iteraciones = 4
    experimentos = 2
    poblacion = 30

    for instancia in instancias:
        for mh in mhs:
            for binarizacion in binarizaciones:
                extra_params = f',DS:{binarizacion},repair:complex,cros:0.4;mut:0.50'
                
                data = {
                    'experimento': f'{mh} {binarizacion}',
                    'MH': mh,
                    'paramMH': f'iter:{iteraciones},pop:{poblacion}{extra_params}',
                    'ML': '',
                    'paramML': '',
                    'ML_FS': '',
                    'paramML_FS': '',
                    'estado': 'pendiente'
                }
                
                cantidad += experimentos
                bd.insertarExperimentos(data, experimentos, instancia[0])

if uscp:
    instancias = bd.obtenerInstancias(f'''"u43", "u44"''') # Ejemplo: "u43", "u44", "cyc06", "cyc11"
    
    binarizaciones = ['V3-STD']
    iteraciones = 4
    experimentos = 2
    poblacion = 30

    for instancia in instancias:
        for mh in mhs:
            for binarizacion in binarizaciones:
                extra_params = f',DS:{binarizacion},repair:complex,cros:0.4;mut:0.50'
                
                data = {
                    'experimento': f'{mh} {binarizacion}',
                    'MH': mh,
                    'paramMH': f'iter:{iteraciones},pop:{poblacion}{extra_params}',
                    'ML': '',
                    'paramML': '',
                    'ML_FS': '',
                    'paramML_FS': '',
                    'estado': 'pendiente'
                }
                
                cantidad += experimentos
                bd.insertarExperimentos(data, experimentos, instancia[0])

print("------------------------------------------------------------------")
print(f'Se ingresaron {cantidad} experimentos a la base de datos')
print("------------------------------------------------------------------")