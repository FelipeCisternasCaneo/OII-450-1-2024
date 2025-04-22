from Util.util import cargar_configuracion

import analisisBEN
import analisisSCP
import analisisUSCP

import time
CONFIG_PATH = './util/json/analysis.json'  # Ruta al archivo JSON

def main():
    """
    Función principal que ejecuta los análisis de los diferentes métodos de optimización,
    dependiendo de los flags de configuración en el archivo JSON.
    También mide y reporta tiempos de ejecución.
    """
    config = cargar_configuracion(CONFIG_PATH)
    tiempos = {}

    tiempo_total_inicio = time.time()

    if config.get("ben", False):
        print("[INFO] Ejecutando análisis BEN...")
        print("-" * 50)
        t0 = time.time()
        analisisBEN.analizar_instancias()
        t1 = time.time()
        tiempos["BEN"] = round(t1 - t0, 2)
    
    if config.get("scp", False):
        print("[INFO] Ejecutando análisis SCP...")
        print("-" * 50)
        t0 = time.time()
        analisisSCP.analizar_instancias()
        t1 = time.time()
        tiempos["SCP"] = round(t1 - t0, 2)

    if config.get("uscp", False):
        print("[INFO] Ejecutando análisis USCP...")
        print("-" * 50)
        t0 = time.time()
        analisisUSCP.analizar_instancias()
        t1 = time.time()
        tiempos["USCP"] = round(t1 - t0, 2)

    tiempo_total_fin = time.time()
    tiempo_total = round(tiempo_total_fin - tiempo_total_inicio, 2)

    ancho = 50  # Ancho total de la línea

    print("\n" + "=" * ancho)
    print("RESUMEN DE TIEMPOS DE EJECUCIÓN".center(ancho))
    print("=" * ancho)

    for metodo, duracion in tiempos.items():
        print(f"  ▸ {metodo:<6} : {duracion:>6.2f} segundos")

    print("-" * ancho)
    print(f"    TOTAL    : {tiempo_total:>6.2f} segundos")
    print("=" * ancho)
    
    print("[INFO] Análisis completado.")

if __name__ == '__main__':
    main()
