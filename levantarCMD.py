import subprocess
import os

def abrir_cmds_ejecutar_main(num_cmds):
    # Obtener la ruta actual del script
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    programa = 'main.py'
    # Verificar que main.py existe en la ruta actual
    main_py_path = os.path.join(ruta_actual, "main.py")
    
    if not os.path.isfile(main_py_path):
        print(f"No se encontró {programa} en la ruta: {ruta_actual}")
        
        return
    
    # Comando para abrir una nueva ventana de cmd y ejecutar main.py
    for _ in range(num_cmds):
        # Comando para abrir cmd, cambiar a la ruta actual y ejecutar main.py
        cmd_command = f'start cmd /K "cd /d {ruta_actual} && python {programa}"'
        subprocess.Popen(cmd_command, shell = True)
        
    print(f"Se han abierto {num_cmds} ventanas de cmd ejecutando {programa} en la ruta: {ruta_actual}")
    
if __name__ == "__main__":
    # Definir la cantidad de CMD a levantar
    num_cmds = 12  # Cambia este valor según lo que necesites
    abrir_cmds_ejecutar_main(num_cmds)