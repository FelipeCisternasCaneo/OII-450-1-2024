from collections import defaultdict
from BD.sqlite import BD
import os
import opfunu.cec_based

bd = BD()

ben = False
scp = True
uscp = True

mhs = ['SBOA']
cantidad = 0
log_resumen = []  # Lista para almacenar el resumen de cada experimento

DS_lista = [
    'V1-STD', 'V1-COM', 'V1-PS', 'V1-ELIT',
    'V2-STD', 'V2-COM', 'V2-PS', 'V2-ELIT',
    'V3-STD', 'V3-COM', 'V3-PS', 'V3-ELIT',
    'V4-STD', 'V4-COM', 'V4-PS', 'V4-ELIT',
    'S1-STD', 'S1-COM', 'S1-PS', 'S1-ELIT',
    'S2-STD', 'S2-COM', 'S2-PS', 'S2-ELIT',
    'S3-STD', 'S3-COM', 'S3-PS', 'S3-ELIT',
    'S4-STD', 'S4-COM', 'S4-PS', 'S4-ELIT',
]

DS_actions = ['V3-ELIT'] # Lista de binarizaciones a utilizar - SCP y USCP

dimensiones_cache = {}

def obtener_dimensiones_ben(funcion):
   # if ben:
        if funcion in BD.data:
            try:
                # Para funciones F1-F13
                if funcion in ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13']:
                    # dimensiones = [30, 100, 500, 1000]
                    dimensiones = [30]
                    
                # Para funciones F14, F16, F17, F18
                if funcion in ['F14', 'F16', 'F17', 'F18']:
                    dimensiones = [2]

                # Para funciones F19
                if funcion in ['F19']:
                    dimensiones = [3]
                    
                # Para funciones F20
                if funcion in ['F20']:
                    dimensiones = [6]
                    
                # Para funciones F15, F21, F22, F23
                if funcion in ['F15', 'F21', 'F22', 'F23']:
                    dimensiones = [4]
                
                return dimensiones
            
            except:
                raise ValueError(f"Advertencia: La función '{funcion}' no está definida.")
        
        elif funcion in BD.opfunu_cec_data:
            func_class = getattr(opfunu.cec_based, f"{funcion}")
            dimensiones = [func_class().dim_default]
        else:
            raise ValueError(f"Función {funcion} no encontrada en la base da datos.")
        
        return dimensiones

def obtener_dimensiones(instance, problema):

    if (instance, problema) in dimensiones_cache:
        return dimensiones_cache[(instance, problema)]
    
    directorios = {
        'SCP': './Problem/SCP/Instances/',
        'USCP': './Problem/USCP/Instances/'
    }

    if problema in directorios:
        if problema == 'USCP' and instance.startswith('u'):
            instance = instance[1:]

        prefijo = 'scp' if problema == 'SCP' else 'uscp'
        ruta_instancia = os.path.join(directorios[problema], f"{prefijo}{instance}.txt")

        if not os.path.exists(ruta_instancia):
            raise FileNotFoundError(f"Archivo {ruta_instancia} no encontrado.")

        with open(ruta_instancia, 'r') as file:
            line = file.readline().strip()
            filas, columnas = map(int, line.split())

        dimensiones = f"{filas} x {columnas}"
        dimensiones_cache[(instance, problema)] = dimensiones
        return dimensiones

    return "-"

def insertar_experimentos(instancias, dimensiones, mhs, experimentos, iteraciones, poblacion, problemaActual, extra_params = ""):
    global cantidad, log_resumen

    for instancia in instancias:
        dimensiones_instancia = obtener_dimensiones(instancia[1], problemaActual)

        for dim in dimensiones:
            for mh in mhs:
                experimento = f'{instancia[1]} {dim}' if problemaActual == 'BEN' else f'{instancia[1]}'

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

                log_resumen.append({
                    "Problema": problemaActual,
                    "Instancia": instancia[1],
                    "Dimensión": dimensiones_instancia if problemaActual != 'BEN' else dim,
                    "MH": mh,
                    "Iteraciones": iteraciones,
                    "Población": poblacion,
                    "Extra Params": extra_params,
                    "Total Experimentos": experimentos
                })

# Proceso BEN
if ben:
    funciones = ['F5', 'F6', 'F8']
    iteraciones = 100
    experimentos = 10
    poblacion = 50

    #validar_dimensiones(dimensiones_usuario)

    for funcion in funciones:
        instancias = bd.obtenerInstancias(f'''"{funcion}"''')
        dimensiones = obtener_dimensiones_ben(funcion)
        insertar_experimentos(instancias, dimensiones, mhs, experimentos, iteraciones, poblacion, problemaActual = 'BEN')

# Proceso SCP y USCP
for problema, activar in [('SCP', scp), ('USCP', uscp)]:
    if activar:
        instancias = bd.obtenerInstancias(f'''"41", "nrh5"''') if problema == 'SCP' else bd.obtenerInstancias(f'''"u43", "uclr11"''')
        iteraciones = 4
        experimentos = 3
        poblacion = 30

        for instancia in instancias:
            for mh in mhs:
                for binarizacion in DS_actions:
                    extra_params = f',DS:{binarizacion},repair:complex,cros:0.4;mut:0.50'
                    insertar_experimentos([instancia], [1], [mh], experimentos, iteraciones, poblacion, problemaActual=problema, extra_params=extra_params)

# Resumen final

print("\n" + "-" * 100)
print(f"{'RESUMEN DETALLADO DE EXPERIMENTOS':^100}")
print("-" * 100)

print(f"{'Problema':<10} {'Instancia':<12} {'Dimensión':<15} {'MH':<10} "
      f"{'Iteraciones':<12} {'Población':<10} {'DS':<10} {'# Experimentos':<15}")
print("-" * 100)

for log in log_resumen:
    ds_value = "-"
    if "DS:" in log['Extra Params']:
        ds_value = log['Extra Params'].split("DS:")[1].split(",")[0]

    print(f"{log['Problema']:<10} {log['Instancia']:<12} {log['Dimensión']:<15} {log['MH']:<10} "
          f"{log['Iteraciones']:<12} {log['Población']:<10} {ds_value:<10} {log['Total Experimentos']:<15}")

print("-" * 100)
print(f"TOTAL EXPERIMENTOS INGRESADOS: {cantidad}")
print("-" * 100)
