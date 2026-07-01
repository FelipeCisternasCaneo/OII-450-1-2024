import subprocess
import os
import sys

# Permite ejecutar este script directamente en la raíz o desde la carpeta de scripts
ruta_actual = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(ruta_actual).lower() == 'scripts':
    PROJECT_ROOT = os.path.abspath(os.path.join(ruta_actual, ".."))
else:
    PROJECT_ROOT = ruta_actual

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def abrir_cmds_ejecutar_main(num_cmds):
    programa = 'main.py'
    # Verificar que main.py existe en la ruta de la raíz del proyecto
    main_py_path = os.path.join(PROJECT_ROOT, "main.py")
    
    if not os.path.isfile(main_py_path):
        print(f"No se encontró {programa} en la ruta: {PROJECT_ROOT}")
        return
    
    # Comando para abrir una nueva ventana de cmd y ejecutar main.py usando el mismo intérprete de Python (entorno virtual activo)
    for _ in range(num_cmds):
        cmd_command = f'start cmd /K "cd /d {PROJECT_ROOT} && \"{sys.executable}\" {programa}"'
        subprocess.Popen(cmd_command, shell=True)
        
    print(f"Se han abierto {num_cmds} ventanas de cmd ejecutando {programa} en la ruta: {PROJECT_ROOT}")
    
if __name__ == "__main__":
    # Definir la cantidad de CMD a levantar
    num_cmds = 12  # Cambia este valor según lo que necesites
    abrir_cmds_ejecutar_main(num_cmds)