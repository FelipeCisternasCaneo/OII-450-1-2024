from BD.sqlite import BD

import json

bd = BD()

ben = False
scp = True

mhs = ['SBOA'] # mhs = ['EOO', 'FOX', 'GOA', 'GWO', 'HBA', 'PSA', 'PSO', 'RSA', 'SCA', 'SHO', 'TDA', 'WOA', 'SBOA']

cantidad = 0

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

if ben:
    # funciones = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F16', 'F17', 'F18', 'F19', 'F20', 'F15', 'F21', 'F22', 'F23']
    
    funciones = ['F8']
    
    for funcion in funciones:
        # poblar ejecuciones Benchmark
        instancias = bd.obtenerInstancias(f'''"{funcion}"''')
        
        # Para funciones F1-F13
        if funcion in ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F1_cec2017', 'F2_cec2017', 'F3_cec2017', 'F5_cec2017']:
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
            
        iteraciones = 500
        experimentos = 1
        poblacion = 30
        
        for instancia in instancias:
            for dim in dimensiones:
                for mh in mhs:
                    data = {}
                    data['experimento'] = f'{instancia[1]} {dim} {mh}'
                    data['MH'] = mh
                    data['paramMH'] = f'iter:{str(iteraciones)},pop:{str(poblacion)}'
                    data['ML'] = ''
                    data['paramML'] = ''
                    data['ML_FS'] = ''
                    data['paramML_FS'] = ''
                    data['estado'] = 'pendiente'
                    cantidad += experimentos
                    bd.insertarExperimentos(data, experimentos, instancia[0])
                    
if scp:
    # poblar ejecuciones SCP
    instancias = bd.obtenerInstancias(f'''"scp41"''')
    
    iteraciones = 10
    experimentos = 5
    poblacion = 30
    
    for instancia in instancias:
        for mh in mhs:
            binarizaciones = ['V3-STD']
            
            for binarizacion in binarizaciones:
                data = {}
                data['experimento'] = f'{mh} {binarizacion}'
                data['MH'] = mh
                data['paramMH'] = f'iter:{str(iteraciones)},pop:{str(poblacion)},DS:{binarizacion},repair:complex,cros:0.4;mut:0.50'
                data['ML'] = ''
                data['paramML'] = ''
                data['ML_FS'] = ''
                data['paramML_FS'] = ''
                data['estado'] = 'pendiente'
                cantidad += experimentos
                bd.insertarExperimentos(data, experimentos, instancia[0])

print("------------------------------------------------------------------")
print(f'Se ingresaron {cantidad} experimentos a la base de datos')
print("------------------------------------------------------------------")