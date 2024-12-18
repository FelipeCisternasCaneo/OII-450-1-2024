from BD.sqlite import BD

import json

bd = BD()

ben = True
scp = True
uscp = True

mhs = ['SBOA']
cantidad = 0

# Dimensiones válidas globales para BEN
DIMENSIONES_VALIDAS = [30, 100, 500, 1000]

# Dimensiones predefinidas por función para BEN
dimensiones_funciones = {
    'F1': [30, 100, 500, 1000], 'F2': [30, 100, 500, 1000], 'F3': [30, 100, 500, 1000],
    'F4': [30, 100, 500, 1000], 'F5': [30, 100, 500, 1000], 'F6': [30, 100, 500, 1000],
    'F7': [30, 100, 500, 1000], 'F8': [30, 100, 500, 1000], 'F9': [30, 100, 500, 1000],
    'F10': [30, 100, 500, 1000], 'F11': [30, 100, 500, 1000], 'F12': [30, 100, 500, 1000],
    'F13': [30, 100, 500, 1000], 'F14': [2], 'F15': [4], 'F16': [2], 'F17': [2],
    'F18': [2], 'F19': [3], 'F20': [6], 'F21': [4], 'F22': [4], 'F23': [4]
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

# Validación de dimensiones ingresadas (solo para BEN)
def validar_dimensiones(dimensiones):
    for dim in dimensiones:
        if dim not in DIMENSIONES_VALIDAS:
            raise ValueError(f"❌ Error: Dimensión {dim} no permitida. Las dimensiones válidas son: {DIMENSIONES_VALIDAS}.")

# Inserta experimentos en la base de datos
def insertar_experimentos(instancias, dimensiones, mhs, experimentos, iteraciones, poblacion, problemaActual, extra_params=""):
    global cantidad

    for instancia in instancias:
        total_experimentos = len(dimensiones) * len(mhs) * experimentos
        
        print(f"🔍 Generando {total_experimentos} experimentos para {problemaActual} {instancia[1]}")  # Mostrar nombre correcto

        for dim in dimensiones:
            for mh in mhs:
                experimento = f'{instancia[1]} {dim} {mh}' if problemaActual == 'BEN' else f'{instancia[1]} {mh}'

                data = {
                    'experimento': experimento,
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

# Proceso BEN
if ben:
    funciones = ['F5', 'F6', 'F8']
    iteraciones = 100
    experimentos = 2
    poblacion = 50
    dimensiones_usuario = [30]  # Cambiar según sea necesario
    
    validar_dimensiones(dimensiones_usuario)

    for funcion in funciones:
        instancias = bd.obtenerInstancias(f'''"{funcion}"''')
        dimensiones = dimensiones_usuario
        insertar_experimentos(instancias, dimensiones, mhs, experimentos, iteraciones, poblacion, problemaActual='BEN')
    
    print("")
        
# Proceso SCP y USCP (dimensiones automáticas calculadas)
for problema, activar in [('SCP', scp), ('USCP', uscp)]:
    if activar:
        instancias = bd.obtenerInstancias(f'''"41", "nrh5"''') if problema == 'SCP' else bd.obtenerInstancias(f'''"u43", "uclr11"''')
        binarizaciones = ['V3-STD']
        iteraciones = 4
        experimentos = 2
        poblacion = 30

        for instancia in instancias:
            total_experimentos = len(binarizaciones) * len(mhs) * experimentos
            print(f"🔍 Generando {total_experimentos} experimentos para {problema} {instancia[1]}")

            for mh in mhs:
                for binarizacion in binarizaciones:
                    extra_params = f',DS:{binarizacion},repair:complex,cros:0.4;mut:0.50'
                    insertar_experimentos([instancia], [1], [mh], experimentos, iteraciones, poblacion, problemaActual=problema, extra_params=extra_params)
        
        print("")

# Mensaje final
print("------------------------------------------------------------------")
print(f'Se ingresaron {cantidad} experimentos a la base de datos')
print("------------------------------------------------------------------")