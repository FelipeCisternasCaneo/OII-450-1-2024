import time
import shutil

from Solver.solverBEN import solverBEN
from Solver.solverSCP import solverSCP

#  NUEVO: Importar solver caótico (opcional, no rompe si no existe)
try:
    from Solver.solverSCP_Chaotic import solverSCP_Chaotic
    CHAOTIC_AVAILABLE = True
except ImportError:
    CHAOTIC_AVAILABLE = False
    print("[INFO] Mapas caóticos no disponibles. Usando solo solver estándar.")

from BD.sqlite import BD

from Util.log import log_experimento, log_error, log_final, log_fecha_hora
from Util.util import parse_parametros, verificar_y_crear_carpetas


# ========== NUEVA FUNCIÓN (20 líneas) ==========

def detectar_mapa_caotico(ds_string):

    mapas_validos = ['LOG', 'SINE', 'TENT', 'CIRCLE', 'SINGER', 'SINU', 'PIECE']
    
    # Dividir por '_'
    partes = ds_string.split('_')
    
    if len(partes) == 2:
        ds_base, sufijo = partes
        if sufijo.upper() in mapas_validos:
            return ds_base, sufijo.upper()
    
    # Sin sufijo o sufijo inválido → estándar
    return ds_string, None


# ========== FUNCIONES ORIGINALES (SIN CAMBIOS) ==========

def ejecutar_ben(id, experimento, parametrosInstancia, parametros):
    """Ejecuta el problema tipo BEN."""
    dim = int(experimento.split(" ")[1])
    lb = float(parametrosInstancia.split(",")[0].split(":")[1])
    ub = float(parametrosInstancia.split(",")[1].split(":")[1])
    
    solverBEN(
        id, parametros["mh"], int(parametros["iter"]),
        int(parametros["pop"]), parametros["instancia"], lb, ub, dim
    )


def ejecutar_problema_scp_uscp(id, instancia, ds, parametros, solver_func, unicost):
    """
    Ejecuta problemas de tipo SCP o USCP.
    
     MODIFICADO: Detecta automáticamente si usar solver caótico.
    """
    repair = parametros["repair"]
    parMH = parametros["cros"]
    
    #  NUEVO: Detectar si el DS tiene sufijo caótico
    # ds es un string como 'V3-ELIT' o 'V3-ELIT_LOG'
    ds_base, chaotic_map = detectar_mapa_caotico(ds)
    
    #  NUEVO: Decidir qué solver usar
    if chaotic_map and CHAOTIC_AVAILABLE:
        # Usar solver caótico
        print(f"[] Usando solver caótico: {ds_base} + {chaotic_map}")
        solverSCP_Chaotic(
            id=id,
            mh=parametros["mh"],
            maxIter=int(parametros["iter"]),
            pop=int(parametros["pop"]),
            instances=instancia,
            DS=ds_base,  # Sin sufijo
            repairType=repair,
            param=parMH,
            unicost=unicost,
            chaotic_map_name=chaotic_map
        )
    else:
        # Usar solver estándar (original)
        if chaotic_map and not CHAOTIC_AVAILABLE:
            print(
                f"[WARN] Solver caótico no disponible (falló la importación). "
                f"Ignorando mapa '{chaotic_map}' y usando estándar."
            )
        
        solver_func(
            id, parametros["mh"], int(parametros["iter"]),
            int(parametros["pop"]), instancia, ds_base, repair, parMH, unicost
        )


def procesar_experimento(data, bd):
    """Procesa cada experimento según su tipo y maneja errores."""
    id = int(data[0][0])
    id_instancia = int(data[0][10])
    datosInstancia = bd.obtenerInstancia(id_instancia)

    parametros = parse_parametros(data[0][4])
    
    parametros.update({
        "mh": data[0][2],
        "instancia": datosInstancia[0][2],
    })
    
    problema = datosInstancia[0][1]
    
    # Validación de iteraciones
    if int(parametros["iter"]) < 4:
        log_error(id, "El número de iteraciones (iter) debe ser al menos 4. Marcado como error.")
        bd.actualizarExperimento(id, "error")
        return

    bd.actualizarExperimento(id, "ejecutando")

    try:
        if problema == "BEN":
            ejecutar_ben(id, data[0][1], datosInstancia[0][4], parametros)

        elif problema == "SCP":
            ejecutar_problema_scp_uscp(
                id, f"scp{datosInstancia[0][2]}", 
                data[0][3],  # Este es el DS (e.g., 'V3-ELIT' o 'V3-ELIT_LOG')
                parametros, solverSCP, unicost=False
            )

        elif problema == "USCP":
            ejecutar_problema_scp_uscp(
                id, f"uscp{datosInstancia[0][2][1:]}", 
                data[0][3],  # Este es el DS
                parametros, solverSCP, unicost=True
            )
    
    except ValueError as ve:
        log_error(id, f"Error de valor: {str(ve)}")
        bd.actualizarExperimento(id, "error")
    
    except Exception as e:
        log_error(id, f"Error general: {str(e)}")
        bd.actualizarExperimento(id, "error")


def main():
    """Función principal que gestiona la ejecución de los experimentos."""
    
    verificar_y_crear_carpetas()
    
    bd = BD()
    
    start_time = time.time()
    
    log_fecha_hora("Inicio de la ejecución")
    
    #  NUEVO: Mostrar info de solvers disponibles
    print("\n" + "="*70)
    print(" SISTEMA DE SOLVERS")
    print("="*70)
    print("   solverBEN (Benchmark)")
    print("   solverSCP (Estándar)")
    if CHAOTIC_AVAILABLE:
        print("   solverSCP_Chaotic (Mapas Caóticos) ")
    else:
        print("   solverSCP_Chaotic (No disponible)")
    print("="*70 + "\n")

    # Connection pooling: mantener conexión abierta durante todo el procesamiento
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
    
    shutil.rmtree("Resultados\\transitorio", ignore_errors=True)

if __name__ == "__main__":
    main()