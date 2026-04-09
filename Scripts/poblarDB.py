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

from BD.sqlite import BD
from Util.log import resumen_experimentos
from Util.util import cargar_configuracion
from Scripts.crearBD import crear_BD

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
    from Solver.domain_managers.registry import get_all as get_all_domains

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
        print(
            f"[INFO] Mapas caóticos deshabilitados globalmente. Usando solo estándar para {problema}."
        )
        return config["DS_actions"]  # Solo binarizaciones estándar

    # Obtener configuración del problema específico
    config_caoticos = config.get("mapas_caoticos", {})
    config_problema = config_caoticos.get(problema, {})

    # Verificar si este problema usa caóticos
    usar_caoticos_problema = config_problema.get("usar_caoticos", False)

    if not usar_caoticos_problema:
        print(
            f"[INFO] Mapas caóticos deshabilitados para {problema}. Usando solo estándar."
        )
        return config["DS_actions"]

    # Obtener modo
    modo = config_problema.get("modo", "solo_estandar")

    print(f"\n[] {problema}: Modo '{modo}'")

    # ========== MODO: SOLO ESTÁNDAR ==========
    if modo == "solo_estandar":
        binarizaciones = config["DS_actions"]
        print(f"  → Binarizaciones estándar: {binarizaciones}")
        return binarizaciones

    # ========== MODO: MANUAL ==========
    elif modo == "manual":
        binarizaciones_manuales = config_caoticos.get("binarizaciones_manuales", {})
        binarizaciones = binarizaciones_manuales.get(problema, config["DS_actions"])
        print(f"   Binarizaciones manuales: {binarizaciones}")
        return binarizaciones

    # ========== MODO: SOLO CAÓTICOS ==========
    elif modo == "solo_caoticos":
        bases = config["DS_actions"]
        mapas = config_problema.get("mapas", [])

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

        print(f"   Bases estándar: {bases}")
        print(f"   Mapas caóticos: {mapas}")
        print(
            f"   Total: {len(binarizaciones)} binarizaciones ({len(bases)} std + {len(bases) * len(mapas)} caóticos)"
        )
        return binarizaciones

    else:
        print(f"  [ERROR] Modo '{modo}' no reconocido. Usando estándar.")
        return config["DS_actions"]


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
    from Solver.domain_managers.registry import get_all as get_all_domains

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
    from Solver.domain_managers import ensure_registered

    ensure_registered()
    from Solver.domain_managers.registry import get_all as get_all_domains

    log_resumen = []
    cantidad = 0
    _batch_valores = []  # Buffer de filas para inserción batch

    print("\n" + "=" * 70)
    print(" SISTEMA DE EXPERIMENTOS CON MAPAS CAÓTICOS")
    print("=" * 70)

    # Mostrar configuración global
    usar_caoticos = config.get("usar_mapas_caoticos", False)
    print(
        f"  Mapas caóticos globalmente: {' ACTIVADOS' if usar_caoticos else ' DESACTIVADOS'}"
    )

    # Mostrar problemas activos (desde el registry)
    problemas_activos = [
        dtype
        for dtype, entry in get_all_domains().items()
        if config.get(entry.config_key, False)
    ]

    print(
        f"  Problemas activos: {', '.join(problemas_activos) if problemas_activos else 'Ninguno'}"
    )
    print(f"  Metaheurísticas: {config['mhs']}")

    # Mostrar configuración de cada problema
    if usar_caoticos and "mapas_caoticos" in config:
        print("\n  Configuración por problema:")
        for problema in problemas_activos:
            if problema in config["mapas_caoticos"]:
                cfg = config["mapas_caoticos"][problema]
                usar = cfg.get("usar_caoticos", False)
                modo = cfg.get("modo", "solo_estandar")
                mapas = cfg.get("mapas", [])
                print(
                    f"    {problema}: {'SI' if usar else 'NO'} | Modo: {modo} | Mapas: {mapas if usar else 'N/A'}"
                )

    print("=" * 70 + "\n")

    # Connection pooling: una sola conexión para todo
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

    print("\n" + "=" * 70)
    print(f" Total de experimentos insertados: {cantidad}")
    print(f" Tiempo de inserción: {t1 - t0:.3f}s")
    print("=" * 70 + "\n")

    resumen_experimentos(log_resumen, cantidad)
