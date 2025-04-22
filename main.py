import time

from Solver.solverBEN import solverBEN
from Solver.solverSCP import solverSCP
#from Solver.solverUSCP import solverUSCP

from BD.sqlite import BD

from Util.log import log_experimento, log_error, log_final, log_fecha_hora
from Util.util import parse_parametros, verificar_y_crear_carpetas

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
    """Ejecuta problemas de tipo SCP o USCP."""
    repair = parametros["repair"]
    
    parMH = parametros["cros"]
    
    solver_func(
        id, parametros["mh"], int(parametros["iter"]),
        int(parametros["pop"]), instancia, ds, repair, parMH, unicost
    )

def procesar_experimento(data, bd):
    """Procesa cada experimento según su tipo y maneja errores."""
    id = int(data[0][0])
    id_instancia = int(data[0][10])
    datosInstancia = bd.obtenerInstancia(id_instancia)

    parametros = parse_parametros(data[0][4])
    
    parametros.update({
        "mh": data[0][2],
        "instancia": datosInstancia[0][2],})
    
    problema = datosInstancia[0][1]
    
    # Validación de iteraciones
    if int(parametros["iter"]) < 4:
        log_error(id, "El número de iteraciones (iter) debe ser al menos 4. Marcado como error.")
        bd.actualizarExperimento(id, "error")
        
        return

    #try:
    bd.actualizarExperimento(id, "ejecutando")

    if problema == "BEN":
        ejecutar_ben(id, data[0][1], datosInstancia[0][4], parametros)

    elif problema == "SCP":
        ejecutar_problema_scp_uscp(id, f"scp{datosInstancia[0][2]}", data[0][3], parametros, solverSCP, unicost=False)

    elif problema == "USCP":
        ejecutar_problema_scp_uscp(id, f"uscp{datosInstancia[0][2][1:]}", data[0][3], parametros, solverSCP, unicost=True)

    '''except ValueError as ve:
        log_error(id, f"Error de valor: {str(ve)}")
        bd.actualizarExperimento(id, "error")'''

    '''except Exception as e:
        log_error(id, f"Error general: {str(e)}")
        bd.actualizarExperimento(id, "error")'''

def main():
    """Función principal que gestiona la ejecución de los experimentos."""
    
    verificar_y_crear_carpetas()
    
    bd = BD()
    data = bd.obtenerExperimento()
    
    start_time = time.time()  # Registrar el tiempo inicial
    
    log_fecha_hora("Inicio de la ejecución")

    while data is not None:
        log_experimento(data)
        procesar_experimento(data, bd)
        data = bd.obtenerExperimento()

    end_time = time.time()
    total_time = end_time - start_time
    
    log_fecha_hora("Fin de la ejecución")
    log_final(total_time)
    
if __name__ == "__main__":
    main()