"""
poblarDB.py - Sistema integrado con experiments_config.json

LEE la configuración de mapas caóticos desde el JSON y genera
automáticamente las binarizaciones según el modo elegido.

MODOS DISPONIBLES EN JSON:
- "solo_estandar": Sin mapas caóticos
- "solo_caoticos": Solo con mapas caóticos
- "comparacion": Estándar + Caóticos
- "manual": Lista manual de binarizaciones
"""

import os
from BD.sqlite import BD
from Util.log import resumen_experimentos
from Util.util import cargar_configuracion
from Scripts.crearBD import crear_BD

# Cargar configuración desde JSON
config = cargar_configuracion('util/json/experiments_config.json')

bd = BD()
dimensiones_cache = {}

if __name__ == '__main__':
    crear_BD()


# ========== FUNCIONES DE UTILIDAD ==========

def obtener_dimensiones_ben(funcion):
    """Obtiene dimensiones para funciones BEN."""
    for key, dims in config['dimensiones']['BEN'].items():
        if funcion in key.split("-"):
            return dims
        
    if funcion in bd.opfunu_cec_data:
        import opfunu.cec_based
        func_class = getattr(opfunu.cec_based, f"{funcion}")
        return [func_class().dim_default]
    
    raise ValueError(f"La función '{funcion}' no está definida.")


def obtener_dimensiones(instance, problema):
    """Obtiene dimensiones para instancias SCP/USCP."""
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


# ==========  FUNCIONES PARA MAPAS CAÓTICOS ==========

def generar_binarizaciones_desde_json(problema):
    """
    Genera lista de binarizaciones según configuración JSON.
    
    Args:
        problema (str): 'SCP', 'USCP', etc.
    
    Returns:
        list: Lista de strings de binarización (e.g., ['V3-ELIT', 'V3-ELIT_LOG'])
    """
    
    # Verificar si mapas caóticos están habilitados globalmente
    usar_caoticos_global = config.get('usar_mapas_caoticos', False)
    
    if not usar_caoticos_global:
        print(f"[INFO] Mapas caóticos deshabilitados globalmente. Usando solo estándar para {problema}.")
        return config['DS_actions']  # Solo binarizaciones estándar
    
    # Obtener configuración del problema específico
    config_caoticos = config.get('mapas_caoticos', {})
    config_problema = config_caoticos.get(problema, {})
    
    # Verificar si este problema usa caóticos
    usar_caoticos_problema = config_problema.get('usar_caoticos', False)
    
    if not usar_caoticos_problema:
        print(f"[INFO] Mapas caóticos deshabilitados para {problema}. Usando solo estándar.")
        return config['DS_actions']
    
    # Obtener modo
    modo = config_problema.get('modo', 'solo_estandar')
    
    print(f"\n[] {problema}: Modo '{modo}'")
    
    # ========== MODO: SOLO ESTÁNDAR ==========
    if modo == 'solo_estandar':
        binarizaciones = config['DS_actions']
        print(f"  → Binarizaciones estándar: {binarizaciones}")
        return binarizaciones
    
    # ========== MODO: MANUAL ==========
    elif modo == 'manual':
        binarizaciones_manuales = config_caoticos.get('binarizaciones_manuales', {})
        binarizaciones = binarizaciones_manuales.get(problema, config['DS_actions'])
        print(f"   Binarizaciones manuales: {binarizaciones}")
        return binarizaciones
    
    # ========== MODO: SOLO CAÓTICOS ==========
    elif modo == 'solo_caoticos':
        bases = config['DS_actions']
        mapas = config_problema.get('mapas', [])
        
        if not mapas:
            print(f"  [WARN] No hay mapas especificados. Usando estándar.")
            return bases
        
        binarizaciones = []
        for base in bases:
            for mapa in mapas:
                binarizaciones.append(f"{base}_{mapa}")
        
        print(f"   Bases: {bases}")
        print(f"   Mapas: {mapas}")
        print(f"   Generadas {len(binarizaciones)} binarizaciones caóticas")
        return binarizaciones
    
    # ========== MODO: COMPARACIÓN ==========
    elif modo == 'comparacion':
        bases = config['DS_actions']
        mapas = config_problema.get('mapas', [])
        
        # Empezar con estándar
        binarizaciones = list(bases)
        
        # Agregar caóticos
        if mapas:
            for base in bases:
                for mapa in mapas:
                    binarizaciones.append(f"{base}_{mapa}")
        
        print(f"   Bases estándar: {bases}")
        print(f"   Mapas caóticos: {mapas}")
        print(f"   Total: {len(binarizaciones)} binarizaciones ({len(bases)} std + {len(bases)*len(mapas)} caóticos)")
        return binarizaciones
    
    else:
        print(f"  [ERROR] Modo '{modo}' no reconocido. Usando estándar.")
        return config['DS_actions']


# ========== FUNCIONES DE INSERCIÓN ==========

def crear_data_experimento(instancia, dim, mh, binarizacion, iteraciones, poblacion, extra_params, problemaActual):
    """Crea diccionario de datos para un experimento."""
    return {
        'experimento': f'{instancia[1]} {dim}' if problemaActual == 'BEN' else f'{instancia[1]}',
        'MH': mh,
        'binarizacion': binarizacion if binarizacion else 'N/A',
        'paramMH': f'iter:{iteraciones},pop:{poblacion}{extra_params}',
        'ML': '',
        'paramML': '',
        'ML_FS': '',
        'paramML_FS': '',
        'estado': 'pendiente'
    }


def crear_resumen_log(instancia, dim, mh, binarizacion, iteraciones, poblacion, extra_params, problemaActual, num_experimentos=1):
    """Crea entrada de log para resumen."""
    dimensiones_instancia = obtener_dimensiones(instancia[1], problemaActual)
    return {
        "Problema": problemaActual,
        "Instancia": instancia[1],
        "Dimensión": dimensiones_instancia if problemaActual != 'BEN' else dim,
        "MH": mh,
        "Iteraciones": iteraciones,
        "Población": poblacion,
        "Binarización": binarizacion if binarizacion else 'N/A',
        "Extra Params": extra_params,
        "Total Experimentos": num_experimentos
    }


def insertar_experimentos(instancias, dimensiones, mhs, num_experimentos, iteraciones, poblacion, problemaActual, extra_params=""):
    """Inserta experimentos en la base de datos."""
    global cantidad, log_resumen

    for instancia in instancias:
        for dim in dimensiones:
            for mh in mhs:
                #  GENERAR BINARIZACIONES DESDE JSON
                binarizaciones = generar_binarizaciones_desde_json(problemaActual)
                
                for binarizacion in binarizaciones:
                    data = crear_data_experimento(instancia, dim, mh, binarizacion, iteraciones, poblacion, extra_params, problemaActual)
                    bd.insertarExperimentos(data, num_experimentos, instancia[0])
                    cantidad += num_experimentos
                    log_resumen.append(crear_resumen_log(instancia, dim, mh, binarizacion, iteraciones, poblacion, extra_params, problemaActual, num_experimentos))


def agregar_experimentos():
    """
    Agrega experimentos según configuración JSON.
     Ahora lee la configuración de mapas caóticos del JSON.
    """
    
    # ========== BEN ==========
    if config.get('ben', False):
        iteraciones = config['experimentos']['BEN']['iteraciones']
        poblacion = config['experimentos']['BEN']['poblacion']
        num_experimentos = config['experimentos']['BEN']['num_experimentos']

        for funcion in config['instancias']['BEN']:
            instancias = bd.obtenerInstancias(f'''"{funcion}"''')
            dimensiones = obtener_dimensiones_ben(funcion)
            insertar_experimentos(instancias, dimensiones, config['mhs'], num_experimentos, iteraciones, poblacion, problemaActual='BEN')

    # ========== SCP/USCP ==========
    for problema in ['SCP', 'USCP']:
        activar = config.get(problema.lower(), False)
        
        if activar:
            instancias_clave = config['instancias'][problema]
            
            if not instancias_clave:
                print(f"[INFO] No hay instancias configuradas para {problema}")
                continue
            
            instancias = bd.obtenerInstancias(",".join(f'"{i}"' for i in instancias_clave))
            iteraciones = config['experimentos'][problema]['iteraciones']
            poblacion = config['experimentos'][problema]['poblacion']
            num_experimentos = config['experimentos'][problema]['num_experimentos']

            for instancia in instancias:
                # Construir parámetros extra según el problema
                if problema in ['SCP', 'USCP']:
                    extra_params = ',repair:complex,cros:0.4;mut:0.50'
                else:
                    extra_params = ''
                
                # Insertar experimentos (binarizaciones se generan automáticamente)
                insertar_experimentos([instancia], [1], config['mhs'], num_experimentos, iteraciones, poblacion, problemaActual=problema, extra_params=extra_params)


# ========== MAIN ==========

if __name__ == '__main__':
    log_resumen = []
    cantidad = 0
    
    print("\n" + "="*70)
    print(" SISTEMA DE EXPERIMENTOS CON MAPAS CAÓTICOS")
    print("="*70)
    
    # Mostrar configuración global
    usar_caoticos = config.get('usar_mapas_caoticos', False)
    print(f"  Mapas caóticos globalmente: {' ACTIVADOS' if usar_caoticos else ' DESACTIVADOS'}")
    
    # Mostrar problemas activos
    problemas_activos = []
    if config.get('ben', False): problemas_activos.append('BEN')
    if config.get('scp', False): problemas_activos.append('SCP')
    if config.get('uscp', False): problemas_activos.append('USCP')
    
    print(f"  Problemas activos: {', '.join(problemas_activos) if problemas_activos else 'Ninguno'}")
    print(f"  Metaheurísticas: {config['mhs']}")
    
    # Mostrar configuración de cada problema
    if usar_caoticos and 'mapas_caoticos' in config:
        print("\n  Configuración por problema:")
        for problema in problemas_activos:
            if problema in config['mapas_caoticos']:
                cfg = config['mapas_caoticos'][problema]
                usar = cfg.get('usar_caoticos', False)
                modo = cfg.get('modo', 'solo_estandar')
                mapas = cfg.get('mapas', [])
                print(f"    {problema}: {'SI' if usar else 'NO'} | Modo: {modo} | Mapas: {mapas if usar else 'N/A'}")
    
    print("="*70 + "\n")
    
    # Connection pooling
    with bd:
        agregar_experimentos()
    
    print("\n" + "="*70)
    print(f" Total de experimentos insertados: {cantidad}")
    print("="*70 + "\n")
    
    resumen_experimentos(log_resumen, cantidad)