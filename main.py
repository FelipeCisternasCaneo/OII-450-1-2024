import time
import shutil
import os

from Solver.universal_solver import universal_solver
from Solver.domain_managers.ben_domain import BenDomainManager
from Solver.domain_managers.scp_domain import ScpDomainManager
from Solver.termination_manager import TerminationCriteria, resolve_effective_max_iter

from BD.sqlite import BD

from Util.log import log_experimento, log_error, log_final, log_fecha_hora
from Util.util import parse_parametros, verificar_y_crear_carpetas


# ========== DETECCIÓN DE MAPAS CAÓTICOS ==========


def detectar_mapa_caotico(ds_string):
    mapas_validos = [
        "LOG",
        "SINE",
        "TENT",
        "CIRCLE",
        "SINGER",
        "SINU",
        "PIECE",
        "CHEB",
        "GAUS",
    ]
    partes = ds_string.split("_")
    if len(partes) == 2:
        ds_base, sufijo = partes
        if sufijo.upper() in mapas_validos:
            return ds_base, sufijo.upper()
    return ds_string, None


def obtener_max_fe(parametros):
    """Retorna max_fe opcional desde paramMH (acepta 'max_fe' o 'fe')."""
    raw = parametros.get("max_fe", parametros.get("fe"))
    if raw is None or raw == "":
        return None

    max_fe = int(raw)
    if max_fe <= 0:
        raise ValueError(
            "El número de evaluaciones de función (max_fe) debe ser mayor a 0."
        )
    return max_fe


def obtener_max_iter(parametros):
    """Retorna iteraciones opcionales desde paramMH."""
    raw = parametros.get("iter")
    if raw is None or raw == "":
        return None

    max_iter = int(raw)
    if max_iter < 4:
        raise ValueError(
            "El número de iteraciones (iter) debe ser al menos 4 cuando se usa terminación por iteraciones."
        )
    return max_iter


def construir_termination(parametros):
    """Construye TerminationCriteria con iteraciones, FE o ambos."""
    max_iter = obtener_max_iter(parametros)
    max_fe = obtener_max_fe(parametros)

    if max_iter is None and max_fe is None:
        raise ValueError(
            "Debe definirse al menos un criterio de término: 'iter' o 'max_fe' (también se acepta 'fe')."
        )

    return TerminationCriteria(max_iter=max_iter, max_fe=max_fe)


# ========== FUNCIONES DE EJECUCIÓN (USANDO UNIVERSAL SOLVER) ==========


def ejecutar_ben(id, experimento, parametrosInstancia, parametros):
    """Ejecuta un problema Benchmark usando el Universal Solver."""
    dim = int(experimento.split(" ")[1])
    lb = float(parametrosInstancia.split(",")[0].split(":")[1])
    ub = float(parametrosInstancia.split(",")[1].split(":")[1])

    mh_name = parametros["mh"]
    pop_size = int(parametros["pop"])
    function_name = parametros["instancia"]

    domain = BenDomainManager(function_name, dim, pop_size, lb, ub)
    termination = construir_termination(parametros)

    universal_solver(id, mh_name, domain, termination)


def ejecutar_problema_scp_uscp(id, instancia, ds, parametros, unicost):
    """Ejecuta un problema SCP/USCP usando el Universal Solver (ruta migrada).

    Para ejecuciones caóticas se instancia ScpDomainManager con chaotic_map_name y
    chaotic_max_iter derivado de TerminationCriteria.
    """
    repair = parametros["repair"]
    parMH = parametros["cros"]
    mh_name = parametros["mh"]
    pop_size = int(parametros["pop"])

    # Detectar si el DS tiene sufijo caótico
    ds_base, chaotic_map = detectar_mapa_caotico(ds)

    # Construir criterios de término ANTES de instanciar el dominio:
    # ScpDomainManager necesita chaotic_max_iter para pregenerar la secuencia caótica
    # con la longitud correcta (max_iter × pop_size × dim).
    termination = construir_termination(parametros)

    # Parámetros extra para GA
    extra_params = None
    if mh_name == "GA" and parMH:
        extra_params = {"param_raw": parMH}

    if chaotic_map:
        # Ruta migrada: ScpDomainManager + universal_solver
        print(
            f"[chaotic] {ds_base} + {chaotic_map} → universal_solver via ScpDomainManager"
        )
        chaotic_max_iter = resolve_effective_max_iter(termination, pop_size, mh_name)
        domain = ScpDomainManager(
            instancia,
            pop_size,
            repair,
            ds_base,
            unicost,
            chaotic_map_name=chaotic_map,
            chaotic_max_iter=chaotic_max_iter,
        )
        universal_solver(id, mh_name, domain, termination, extra_params=extra_params)
    else:
        # Universal Solver para SCP/USCP estándar (sin mapa caótico)
        domain = ScpDomainManager(instancia, pop_size, repair, ds_base, unicost)
        universal_solver(id, mh_name, domain, termination, extra_params=extra_params)


def procesar_experimento(data, bd):
    """Procesa cada experimento según su tipo y maneja errores."""
    id = int(data[0][0])
    id_instancia = int(data[0][10])
    datosInstancia = bd.obtenerInstancia(id_instancia)

    parametros = parse_parametros(data[0][4])

    parametros.update(
        {
            "mh": data[0][2],
            "instancia": datosInstancia[0][2],
        }
    )

    problema = datosInstancia[0][1]

    # Validación de criterio de término
    try:
        construir_termination(parametros)
    except ValueError as ve:
        log_error(id, str(ve))
        bd.actualizarExperimento(id, "error")
        return

    bd.actualizarExperimento(id, "ejecutando")

    try:
        if problema == "BEN":
            ejecutar_ben(id, data[0][1], datosInstancia[0][4], parametros)

        elif problema == "SCP":
            ejecutar_problema_scp_uscp(
                id, f"scp{datosInstancia[0][2]}", data[0][3], parametros, unicost=False
            )

        elif problema == "USCP":
            ejecutar_problema_scp_uscp(
                id,
                f"uscp{datosInstancia[0][2][1:]}",
                data[0][3],
                parametros,
                unicost=True,
            )

    except ValueError as ve:
        log_error(id, f"Error de valor: {str(ve)}")
        bd.actualizarExperimento(id, "error")

    except Exception as e:
        log_error(id, f"Error general: {str(e)}")
        bd.actualizarExperimento(id, "error")

    except (KeyboardInterrupt, SystemExit):
        print(
            f"\n[!] Ejecución interrumpida manualmente (Ctrl+C). Devolviendo experimento {id} a estado 'pendiente'..."
        )
        bd.actualizarExperimento(id, "pendiente")
        raise


def main():
    """Función principal que gestiona la ejecución de los experimentos."""

    verificar_y_crear_carpetas()

    bd = BD()

    start_time = time.time()

    log_fecha_hora("Inicio de la ejecución")

    print("\n" + "=" * 70)
    print(" SISTEMA DE SOLVERS (Universal Solver)")
    print("=" * 70)
    print("    Universal Solver (BEN + SCP/USCP + Caótico via ScpDomainManager)")
    print("=" * 70 + "\n")

    with bd:
        data = bd.obtenerExperimento()

        while data is not None:
            log_experimento(data)
            procesar_experimento(data, bd)
            data = bd.obtenerExperimento()

    end_time = time.time()
    total_time = end_time - start_time

    log_fecha_hora("Fin de la ejecución")
    log_final(total_time)

    shutil.rmtree(os.path.join("Resultados", "transitorio"), ignore_errors=True)


if __name__ == "__main__":
    main()
