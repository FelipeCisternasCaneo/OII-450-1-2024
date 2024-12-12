import numpy as np
import matplotlib.pyplot as plt

from Problem.Benchmark.Problem import fitness as f

ub = [100, 10, 100, 100, 20, 100, 1, 500, 5, 20, 600, 10, 5, 50, 5, 1, 15, 5, 5, 5, 5, 5, 5]
lb = [-100, -10, -100, -100, -20, -100, -1, -500, -5, -20, -600, -10, -5, -50, -5, -1, -5, -5, -5, -5, -5, -5, -5]

funciones = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20', 'F21', 'F22', 'F23']
i = 0
for funcion in funciones:
    # Crear una malla de puntos en el espacio de entrada
    x1 = np.linspace(lb[i], ub[i], 100)
    x2 = np.linspace(lb[i], ub[i], 100)
    X1, X2 = np.meshgrid(x1, x2)

    # Calcular F1 para cada par (x1, x2) considerando x como un vector numpy
    if funcion == 'F15':
        x3 = 0
        x4 = 0
        Z = np.array([f(funcion,np.array([x1, x2, x3, x4])) for x1, x2 in zip(X1.flatten(), X2.flatten())]).reshape(X1.shape)
    elif funcion == 'F19':
        x3 = 0
        Z = np.array([f(funcion,np.array([x1, x2, x3])) for x1, x2 in zip(X1.flatten(), X2.flatten())]).reshape(X1.shape)   
    elif funcion == 'F20':
        x3 = 0
        x4 = 0
        x5 = 0
        x6 = 0
        Z = np.array([f(funcion,np.array([x1, x2, x3, x4, x5, x6])) for x1, x2 in zip(X1.flatten(), X2.flatten())]).reshape(X1.shape)   
    elif funcion == 'F21' or funcion == 'F22' or funcion == 'F23':
        x3 = 0
        x4 = 0
        Z = np.array([f(funcion,np.array([x1, x2, x3, x4])) for x1, x2 in zip(X1.flatten(), X2.flatten())]).reshape(X1.shape)   
    else:
        Z = np.array([f(funcion,np.array([x1, x2])) for x1, x2 in zip(X1.flatten(), X2.flatten())]).reshape(X1.shape)

    # Crear el gráfico
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)

    # Agregar la proyección en el plano x1, x2
    
    if funcion == 'F8': ax.contourf(X1, X2, Z, zdir = 'z', offset = -1000, cmap = 'viridis', alpha = 0.7)
    elif funcion == 'F16': ax.contourf(X1, X2, Z, zdir = 'z', offset = -2, cmap = 'viridis', alpha = 0.7)
    elif funcion == 'F19': ax.contourf(X1, X2, Z, zdir = 'z', offset = -0.12, cmap = 'viridis', alpha = 0.7)
    elif funcion == 'F20': ax.contourf(X1, X2, Z, zdir = 'z', offset = -0.12, cmap = 'viridis', alpha = 0.7)
    elif funcion == 'F21': ax.contourf(X1, X2, Z, zdir = 'z', offset = -0.5, cmap = 'viridis', alpha = 0.7)
    elif funcion == 'F22' or funcion == 'F23': ax.contourf(X1, X2, Z, zdir = 'z', offset= -0.6, cmap = 'viridis', alpha = 0.7)
    else: ax.contourf(X1, X2, Z, zdir = 'z', offset = 0, cmap = 'viridis', alpha = 0.7)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel(f'{funcion}(x)')
    
    if funcion == 'F8': ax.set_zlim(-1000, np.max(Z))  # Ajustar el límite inferior del eje Z para que la proyección se vea bien
    elif funcion == 'F16': ax.set_zlim(-2, np.max(Z))  # Ajustar el límite inferior del eje Z para que la proyección se vea bien
    elif funcion == 'F19': ax.set_zlim(-0.12, 0)  # Ajustar el límite inferior del eje Z para que la proyección se vea bien    
    elif funcion == 'F20': ax.set_zlim(-0.12, 0)  # Ajustar el límite inferior del eje Z para que la proyección se vea bien    
    elif funcion == 'F21': ax.set_zlim(-0.5, 0)  # Ajustar el límite inferior del eje Z para que la proyección se vea bien 
    elif funcion == 'F22' or funcion == 'F23': ax.set_zlim(-0.6, 0)  # Ajustar el límite inferior del eje Z para que la proyección se vea bien 
    else: ax.set_zlim(0, np.max(Z))  # Ajustar el límite inferior del eje Z para que la proyección se vea bien

    plt.savefig(f'./Graficos_Benchmark/{funcion}.pdf')
    print(f'Grafico para la funcion {funcion} ha sido creado')
    plt.close('all')
    i += 1