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
import sys

# Permite ejecutar este script directamente (python Scripts/poblarDB.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from bd.sqlite import BD
from util.log import resumen_experimentos
from util.util import cargar_configuracion
from scripts.crearBD import crear_BD

# Cargar configuración desde JSON
config = cargar_configuracion("util/json/experiments_config.json")

bd = BD()
dimensiones_cache = {}

if __name__ == "__main__":
    crear_BD()


# ========== FUNCIONES DE UTILIDAD ==========


def obtener_dimensiones_ben(funcion):
    """Obtiene dimensiones para funciones BEN."""
    for key, dims in config["dimensiones"]["BEN"].items():
        if funcion in key.split("-"):
            return dims

    raise ValueError(f"La función '{funcion}' no está definida.")


def obtener_dimensiones(instance, problema):
    """Obtiene dimensiones para instancias SCP/USCP (o cualquier dominio con instance_dir)."""
    if (instance, problema) in dimensiones_cache:
        return dimensiones_cache[(instance, problema)]

    # Construir directorios desde el Domain Registry
    from solver.domain_managers.registry import get_all as get_all_domains

    directorios = {
        dtype: entry.instance_dir
        for dtype, entry in get_all_domains().items()
        if entry.instance_dir is not None
    }

    if problema in directorios:
        if problema == "USCP" and instance.startswith("u"):
            instance = instance[1:]

        prefijo = "scp" if problema == "SCP" else "uscp"
        ruta_instancia = os.path.join(directorios[problema], f"{prefijo}{instance}.txt")

        if not os.path.exists(ruta_instancia):
            raise FileNotFoundError(f"Archivo {ruta_instancia} no encontrado.")

        with open(ruta_instancia, "r") as file:
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
    usar_caoticos_global = config.get("usar_mapas_caoticos", False)

    if not usar_caoticos_global:
        return config["DS_actions"]  # Solo binarizaciones estándar

    # Obtener configuración del problema específico
    config_caoticos = config.get("mapas_caoticos", {})
    config_problema = config_caoticos.get(problema, {})

    # Verificar si este problema usa caóticos
    usar_caoticos_problema = config_problema.get("usar_caoticos", False)

    if not usar_caoticos_problema:
        return config["DS_actions"]

    # Obtener modo
    modo = config_problema.get("modo", "solo_estandar")

    # ========== MODO: SOLO ESTÁNDAR ==========
    if modo == "solo_estandar":
        return config["DS_actions"]

    # ========== MODO: MANUAL ==========
    elif modo == "manual":
        binarizaciones_manuales = config_caoticos.get("binarizaciones_manuales", {})
        return binarizaciones_manuales.get(problema, config["DS_actions"])

    # ========== MODO: SOLO CAÓTICOS ==========
    elif modo == "solo_caoticos":
        bases = config["DS_actions"]
        mapas = config_problema.get("mapas", [])

        if not mapas:
            return bases

        binarizaciones = []
        for base in bases:
            for mapa in mapas:
                binarizaciones.append(f"{base}_{mapa}")
        return binarizaciones

    # ========== MODO: COMPARACIÓN ==========
    elif modo == "comparacion":
        bases = config["DS_actions"]
        mapas = config_problema.get("mapas", [])

        # Empezar con estándar
        binarizaciones = list(bases)

        # Agregar caóticos
        if mapas:
            for base in bases:
                for mapa in mapas:
                    binarizaciones.append(f"{base}_{mapa}")
        return binarizaciones

    else:
        return config["DS_actions"]


def imprimir_tabla_configuracion_problemas(problemas_activos):
    """Muestra la configuración de binarizaciones de cada problema activo como una tabla de ancho 100."""
    if not problemas_activos:
        print("No hay problemas activos.")
        return

    SEP = "-" * 100
    print(SEP)
    print(f"{'Problema':<10} {'Modo':<15} {'Bases Estándar':<25} {'Mapas Caóticos':<22} {'Total Binarizaciones':<28}")
    print(SEP)

    usar_caoticos_global = config.get("usar_mapas_caoticos", False)
    bases = config.get("DS_actions", [])

    for problema in problemas_activos:
        if not usar_caoticos_global:
            modo = "solo_estandar"
            mapas_str = "N/A"
            totales_str = str(len(bases))
        else:
            config_caoticos = config.get("mapas_caoticos", {})
            config_problema = config_caoticos.get(problema, {})
            usar_caoticos_problema = config_problema.get("usar_caoticos", False)

            if not usar_caoticos_problema:
                modo = "solo_estandar"
                mapas_str = "N/A"
                totales_str = str(len(bases))
            else:
                modo = config_problema.get("modo", "solo_estandar")
                if modo == "solo_estandar":
                    mapas_str = "N/A"
                    totales_str = str(len(bases))
                elif modo == "manual":
                    mapas_str = "N/A"
                    binarizaciones_manuales = config_caoticos.get("binarizaciones_manuales", {})
                    binarizaciones = binarizaciones_manuales.get(problema, bases)
                    totales_str = str(len(binarizaciones))
                elif modo == "solo_caoticos":
                    mapas = config_problema.get("mapas", [])
                    mapas_str = str(mapas)
                    totales_str = f"{len(bases) * len(mapas)} (solo caot)"
                elif modo == "comparacion":
                    mapas = config_problema.get("mapas", [])
                    mapas_str = str(mapas)
                    totales_str = f"{len(bases) + len(bases) * len(mapas)} ({len(bases)} std + {len(bases) * len(mapas)} caot)"
                else:
                    mapas_str = "N/A"
                    totales_str = str(len(bases))

        print(f"{problema:<10} {modo:<15} {str(bases):<25} {mapas_str:<22} {totales_str:<28}")
    print(SEP)


# ========== FUNCIONES DE INSERCIÓN ==========


def crear_data_experimento(
    instancia,
    dim,
    mh,
    binarizacion,
    iteraciones,
    poblacion,
    extra_params,
    problemaActual,
    max_fe=None,
    modo_terminacion="iter",
):
    """Crea diccionario de datos para un experimento."""
    modo = (modo_terminacion or "iter").lower()
    param_parts = [f"pop:{poblacion}"]

    if modo in ("iter", "both"):
        param_parts.insert(0, f"iter:{iteraciones}")
    if modo in ("fe", "both") and max_fe is not None:
        param_parts.append(f"max_fe:{max_fe}")

    param_base = ",".join(param_parts)

    return {
        "experimento": f"{instancia[1]} {dim}"
        if problemaActual == "BEN"
        else f"{instancia[1]}",
        "MH": mh,
        "binarizacion": binarizacion if binarizacion else "N/A",
        "paramMH": f"{param_base}{extra_params}",
        "ML": "",
        "paramML": "",
        "ML_FS": "",
        "paramML_FS": "",
        "estado": "pendiente",
    }


def crear_resumen_log(
    instancia,
    dim,
    mh,
    binarizacion,
    iteraciones,
    poblacion,
    extra_params,
    problemaActual,
    num_experimentos=1,
    max_fe=None,
    modo_terminacion="iter",
):
    """Crea entrada de log para resumen."""
    dimensiones_instancia = obtener_dimensiones(instancia[1], problemaActual)
    return {
        "Problema": problemaActual,
        "Instancia": instancia[1],
        "Dimensión": dimensiones_instancia if problemaActual != "BEN" else dim,
        "MH": mh,
        "Modo Terminación": modo_terminacion,
        "Iteraciones": iteraciones,
        "Max FE": max_fe if max_fe is not None else "-",
        "Población": poblacion,
        "Binarización": binarizacion if binarizacion else "N/A",
        "Extra Params": extra_params,
        "Total Experimentos": num_experimentos,
    }


def insertar_experimentos(
    instancias,
    dimensiones,
    mhs,
    num_experimentos,
    iteraciones,
    poblacion,
    problemaActual,
    extra_params="",
    max_fe=None,
    modo_terminacion="iter",
):
    """Inserta experimentos en la base de datos (versión optimizada con batch)."""
    global cantidad, log_resumen, _batch_valores

    # Cachear binarizaciones fuera del loop (solo depende de problemaActual)
    binarizaciones = generar_binarizaciones_desde_json(problemaActual)

    for instancia in instancias:
        for dim in dimensiones:
            for mh in mhs:
                for binarizacion in binarizaciones:
                    data = crear_data_experimento(
                        instancia,
                        dim,
                        mh,
                        binarizacion,
                        iteraciones,
                        poblacion,
                        extra_params,
                        problemaActual,
                        max_fe=max_fe,
                        modo_terminacion=modo_terminacion,
                    )

                    batch_id = data.get("batch_id") if isinstance(data, dict) else None
                    fila = (
                        str(data["experimento"]),
                        str(data["MH"]),
                        str(data["binarizacion"]),
                        str(data["paramMH"]),
                        str(data["ML"]),
                        str(data["paramML"]),
                        str(data["ML_FS"]),
                        str(data["paramML_FS"]),
                        str(data["estado"]),
                        instancia[0],
                        batch_id,
                        None,
                        None,
                    )
                    # Agregar la misma fila N veces (num_experimentos corridas)
                    _batch_valores.extend([fila] * num_experimentos)
                    cantidad += num_experimentos
                    log_resumen.append(
                        crear_resumen_log(
                            instancia,
                            dim,
                            mh,
                            binarizacion,
                            iteraciones,
                            poblacion,
                            extra_params,
                            problemaActual,
                            num_experimentos,
                            max_fe=max_fe,
                            modo_terminacion=modo_terminacion,
                        )
                    )


def agregar_experimentos():
    """
    Agrega experimentos según configuración JSON.
    Itera los dominios registrados en el Domain Registry en vez de
    hardcodear listas de problemas.
    """
    from solver.domain_managers.registry import get_all as get_all_domains

    # Configuración global de terminación (aplica a todos los problemas)
    exp_cfg = config.get("experimentos", {})
    modo_terminacion_global = exp_cfg.get("modo_terminacion", "iter")
    max_fe_global = exp_cfg.get("max_fe", exp_cfg.get("fe"))

    if modo_terminacion_global not in ("iter", "fe", "both"):
        raise ValueError(
            "experimentos.modo_terminacion debe ser 'iter', 'fe' o 'both'."
        )
    if modo_terminacion_global in ("fe", "both") and max_fe_global is None:
        raise ValueError(
            "modo_terminacion global en 'fe'/'both' requiere experiments.max_fe (o fe)."
        )

    # Iterar todos los dominios registrados
    for domain_type, entry in get_all_domains().items():
        config_key = entry.config_key  # e.g. "ben", "scp", "uscp"

        if not config.get(config_key, False):
            continue

        # BEN tiene lógica especial (dimensiones por función, no por archivo)
        if domain_type == "BEN":
            iteraciones = config["experimentos"]["BEN"]["iteraciones"]
            poblacion = config["experimentos"]["BEN"]["poblacion"]
            num_experimentos = config["experimentos"]["BEN"]["num_experimentos"]

            for funcion in config["instancias"]["BEN"]:
                instancias = bd.obtenerInstancias([funcion])
                dimensiones = obtener_dimensiones_ben(funcion)
                insertar_experimentos(
                    instancias,
                    dimensiones,
                    config["mhs"],
                    num_experimentos,
                    iteraciones,
                    poblacion,
                    problemaActual="BEN",
                    max_fe=max_fe_global,
                    modo_terminacion=modo_terminacion_global,
                )
        else:
            # Dominios con instancias (SCP, USCP, y futuros KP, MKP, etc.)
            instancias_clave = config.get("instancias", {}).get(domain_type)

            if not instancias_clave:
                print(f"[INFO] No hay instancias configuradas para {domain_type}")
                continue

            instancias = bd.obtenerInstancias(instancias_clave)
            iteraciones = config["experimentos"][domain_type]["iteraciones"]
            poblacion = config["experimentos"][domain_type]["poblacion"]
            num_experimentos = config["experimentos"][domain_type]["num_experimentos"]

            for instancia in instancias:
                extra_params = entry.default_extra_params
                insertar_experimentos(
                    [instancia],
                    [1],
                    config["mhs"],
                    num_experimentos,
                    iteraciones,
                    poblacion,
                    problemaActual=domain_type,
                    extra_params=extra_params,
                    max_fe=max_fe_global,
                    modo_terminacion=modo_terminacion_global,
                )


# ========== MAIN ==========

if __name__ == "__main__":
    # Auto-registrar dominios
    from solver.domain_managers import ensure_registered

    ensure_registered()
    from solver.domain_managers.registry import get_all as get_all_domains

    log_resumen = []
    cantidad = 0
    _batch_valores = []  # Buffer de filas para inserción batch

    W = 100
    SEP = "-" * W

    print(f"\n{'SISTEMA DE EXPERIMENTOS - POBLAR BASE DE DATOS':^{W}}")
    print(SEP)

    # ── Configuración Global ──
    usar_caoticos = config.get("usar_mapas_caoticos", False)
    problemas_activos = [
        dtype
        for dtype, entry in get_all_domains().items()
        if config.get(entry.config_key, False)
    ]

    print(f"{'Parámetro':<30} {'Valor':<70}")
    print(SEP)
    print(f"{'Mapas caóticos':<30} {'ACTIVADOS' if usar_caoticos else 'DESACTIVADOS':<70}")
    print(f"{'Problemas activos':<30} {', '.join(problemas_activos) if problemas_activos else 'Ninguno':<70}")
    print(f"{'Metaheurísticas':<30} {str(config['mhs']):<70}")
    print(SEP)

    # ── Configuración por Problema ──
    print(f"\n{'CONFIGURACIÓN POR PROBLEMA':^{W}}")
    imprimir_tabla_configuracion_problemas(problemas_activos)

    # ── Inserción ──
    import time

    t0 = time.perf_counter()

    with bd:
        agregar_experimentos()

        # Inserción batch: un solo executemany + commit para TODOS los experimentos
        if _batch_valores:
            bd.getCursor().executemany(
                """INSERT INTO experimentos (
                       experimento, MH, binarizacion, paramMH, ML, paramML, ML_FS, paramML_FS,
                       estado, fk_id_instancia, batch_id, ts_inicio, ts_fin
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                _batch_valores,
            )
            bd.commit()

    t1 = time.perf_counter()

    print(f"\n{'INSERCIÓN EN BASE DE DATOS':^{W}}")
    print(SEP)
    print(f"{'Experimentos insertados':<30} {cantidad:<70}")
    print(f"{'Tiempo de inserción':<30} {f'{t1 - t0:.3f}s':<70}")
    print(SEP)

    # ── Resumen Detallado ──
    print(f"\n{'RESUMEN DETALLADO DE EXPERIMENTOS':^{W}}")
    resumen_experimentos(log_resumen, cantidad)

