import time
import shutil
import os

# Auto-registra todos los dominios al importar el paquete
from Solver.domain_managers import ensure_registered

ensure_registered()
from Solver.domain_managers.registry import get as get_domain

from BD.sqlite import BD

from Util.log import log_experimento, log_error, log_final, log_fecha_hora
from Util.util import parse_parametros, verificar_y_crear_carpetas


# ========== VALIDACIÓN DE PARÁMETROS ==========


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
    from Solver.termination_manager import TerminationCriteria

    max_iter = obtener_max_iter(parametros)
    max_fe = obtener_max_fe(parametros)

    if max_iter is None and max_fe is None:
        raise ValueError(
            "Debe definirse al menos un criterio de término: 'iter' o 'max_fe' (también se acepta 'fe')."
        )

    return TerminationCriteria(max_iter=max_iter, max_fe=max_fe)


# ========== PROCESAMIENTO DE EXPERIMENTOS (VÍA DOMAIN REGISTRY) ==========


def procesar_experimento(data, bd):
    """Procesa cada experimento consultando el Domain Registry.

    El tipo de problema se obtiene de la BD y se despacha al executor
    registrado para ese dominio. Si el dominio no está registrado,
    se marca el experimento como error con un mensaje descriptivo.
    """
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
        # Despacho vía Domain Registry — sin if/elif hardcoded
        entry = get_domain(problema)
        entry.execute_experiment(id, data, datosInstancia, parametros)

    except KeyError as ke:
        log_error(id, f"Dominio no registrado: {ke}")
        bd.actualizarExperimento(id, "error")

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
    print("    Universal Solver (despacho dinámico via Domain Registry)")
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
