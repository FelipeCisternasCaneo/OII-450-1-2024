import numpy as np
import matplotlib.pyplot as plt
import os
from Problem.Benchmark.Problem import fitness as f

# Límites de las variables
ub = [100, 10, 100, 100, 20, 100, 1, 500, 5, 20, 600, 10, 5, 50, 5, 1, 15, 5, 5, 5, 5, 5, 5]
lb = [-100, -10, -100, -100, -20, -100, -1, -500, -5, -20, -600, -10, -5, -50, -5, -1, -5, -5, -5, -5, -5, -5, -5]

# Funciones a graficar
funciones = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20', 'F21', 'F22', 'F23']

i = 0
for funcion in funciones:
    # Crear malla de puntos
    x1 = np.linspace(lb[i], ub[i], 100)
    x2 = np.linspace(lb[i], ub[i], 100)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Verificar si el archivo ya existe
    file_path = f'./Graficos_Benchmark/{funcion}.pdf'
    if os.path.exists(file_path):
        print(f'El gráfico para la función {funcion} ya existe. Saltando...')
        continue  # Si el archivo ya existe, se salta a la siguiente función
    
    # Parametros adicionales según la función
    if funcion == 'F15':
        Z = np.array([f(funcion, np.array([x1, x2, 0, 0])) for x1, x2 in zip(X1.flatten(), X2.flatten())]).reshape(X1.shape)
    elif funcion == 'F19':
        Z = np.array([f(funcion, np.array([x1, x2, 0])) for x1, x2 in zip(X1.flatten(), X2.flatten())]).reshape(X1.shape)
    elif funcion == 'F20':
        Z = np.array([f(funcion, np.array([x1, x2, 0, 0, 0, 0])) for x1, x2 in zip(X1.flatten(), X2.flatten())]).reshape(X1.shape)
    elif funcion in ['F21', 'F22', 'F23']:
        Z = np.array([f(funcion, np.array([x1, x2, 0, 0])) for x1, x2 in zip(X1.flatten(), X2.flatten())]).reshape(X1.shape)
    else:
        Z = np.array([f(funcion, np.array([x1, x2])) for x1, x2 in zip(X1.flatten(), X2.flatten())]).reshape(X1.shape)

    # Crear el gráfico
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)

    # Agregar la proyección en el plano x1, x2
    proyeccion_offset = {
        'F8': -1000, 'F16': -2, 'F19': -0.12, 'F20': -0.12, 'F21': -0.5, 'F22': -0.6, 'F23': -0.6
    }
    offset = proyeccion_offset.get(funcion, 0)
    ax.contourf(X1, X2, Z, zdir='z', offset=offset, cmap='viridis', alpha=0.7)

    # Configurar límites en el eje Z
    lim_z = {
        'F8': (-1000, np.max(Z)),
        'F16': (-2, np.max(Z)),
        'F19': (-0.12, 0),
        'F20': (-0.12, 0),
        'F21': (-0.5, 0),
        'F22': (-0.6, 0),
        'F23': (-0.6, 0)
    }
    ax.set_zlim(lim_z.get(funcion, (0, np.max(Z))))

    # Etiquetas de los ejes
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel(f'{funcion}(x)')

    # Guardar el gráfico
    plt.savefig(file_path)
    print(f'Gráfico para la función {funcion} ha sido creado')

    plt.close(fig)  # Cerrar la figura

    i += 1
