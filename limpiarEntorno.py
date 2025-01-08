import os

def limpiarEntorno():
    for root, _, files in os.walk('./resultados/'):
        for file in files:
            if file.endswith('.csv') or file.endswith('.pdf'):
                os.remove(os.path.join(root, file))
    
    print("Archivos CSV y PDF eliminados exitosamente.")
    
if __name__ == '__main__':
    limpiarEntorno()