import os
import json
import opfunu.cec_based

from BD.sqlite import BD
from Util.log import resumen_experimentos
from Util.util import cargar_configuracion

from crearBD import crear_BD

config = cargar_configuracion('util/json/experiments_config.json')

bd = BD()
dimensiones_cache = {}

'''
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
'''

if __name__ == '__main__':
    crear_BD()

def obtener_dimensiones_ben(funcion):
    for key, dims in config['dimensiones']['BEN'].items():
        if funcion in key.split("-"):
            return dims
        
    if funcion in bd.opfunu_cec_data:
        func_class = getattr(opfunu.cec_based, f"{funcion}")
        return [func_class().dim_default]
    
    raise ValueError(f"La función '{funcion}' no está definida en la configuración de dimensiones ni en opfunu.")

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

def insertar_experimentos(instancias, dimensiones, mhs, num_experimentos, iteraciones, poblacion, problemaActual, extra_params=""):
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

                cantidad += num_experimentos
                bd.insertarExperimentos(data, num_experimentos, instancia[0])

                log_resumen.append({
                    "Problema": problemaActual,
                    "Instancia": instancia[1],
                    "Dimensión": dimensiones_instancia if problemaActual != 'BEN' else dim,
                    "MH": mh,
                    "Iteraciones": iteraciones,
                    "Población": poblacion,
                    "Extra Params": extra_params,
                    "Total Experimentos": num_experimentos
                })

def agregar_experimentos():
    if config.get('ben', False):
        iteraciones = config['experimentos']['BEN']['iteraciones']
        poblacion = config['experimentos']['BEN']['poblacion']
        num_experimentos = config['experimentos']['BEN']['num_experimentos']

        for funcion in config['instancias']['BEN']:
            instancias = bd.obtenerInstancias(f'''"{funcion}"''')
            dimensiones = obtener_dimensiones_ben(funcion)
            insertar_experimentos(instancias, dimensiones, config['mhs'], num_experimentos, iteraciones, poblacion, problemaActual='BEN')

    for problema, activar in [('SCP', config.get('scp', False)), ('USCP', config.get('uscp', False))]:
        if activar:
            instancias_clave = config['instancias'][problema]
            instancias = bd.obtenerInstancias(",".join(f'"{i}"' for i in instancias_clave))
            iteraciones = config['experimentos'][problema]['iteraciones']
            poblacion = config['experimentos'][problema]['poblacion']
            num_experimentos = config['experimentos'][problema]['num_experimentos']

            for instancia in instancias:
                for mh in config['mhs']:
                    for binarizacion in config['DS_actions']:
                        extra_params = f',DS:{binarizacion},repair:complex,cros:0.4;mut:0.50'
                        insertar_experimentos([instancia], [1], [mh], num_experimentos, iteraciones, poblacion, problemaActual=problema, extra_params=extra_params)

# Resumen final
if __name__ == '__main__':
    log_resumen = []
    cantidad = 0
    
    agregar_experimentos()
    resumen_experimentos(log_resumen, cantidad)