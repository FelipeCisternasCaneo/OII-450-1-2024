import os
import shutil

def limpiarEntorno():
    """
    Elimina todo el contenido (archivos y subcarpetas) dentro de la carpeta './resultados/'
    y luego la vuelve a crear vacía.
    """
    directorio_resultados = './Resultados/' # Definir la ruta

    # Verificar si el directorio existe antes de intentar borrarlo
    if os.path.exists(directorio_resultados):
        try:
            # shutil.rmtree elimina la carpeta y TODO su contenido recursivamente
            shutil.rmtree(directorio_resultados)
            print(f"Directorio '{directorio_resultados}' y su contenido eliminados.")
        except OSError as e:
            # Manejar posibles errores (ej: permisos)
            print(f"Error al eliminar el directorio '{directorio_resultados}': {e}")
            return # Salir si no se pudo borrar
    
    try:
        os.makedirs(directorio_resultados, exist_ok=True)
        print(f"Directorio '{directorio_resultados}' creado vacío exitosamente.")
    except OSError as e:
        print(f"Error al crear el directorio '{directorio_resultados}': {e}")

if __name__ == '__main__':
    # Preguntar al usuario para seguridad
    confirmacion = input("¿Estás seguro de que quieres borrar TODO el contenido de './resultados/'? (s/N): ")
    if confirmacion.lower() == 's':
        limpiarEntorno()
    else:
        print("Operación cancelada.")