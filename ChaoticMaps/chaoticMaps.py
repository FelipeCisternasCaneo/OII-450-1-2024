import numpy as np
import matplotlib.pyplot as plt
import math

def chebyshevMap(initial,iteration):
    mapValues = np.zeros(iteration)
       
    
    i = 1
    mapValues[0] = initial
    xPrevious = initial
    while i < iteration:
        
        x = math.cos( xPrevious * ( 1 / math.cos( xPrevious ) ) )
        
        mapValues[i] = x
        
        xPrevious = x
        
        i+=1
    return mapValues

def gaussianAndGauss_mouseMap(initial,iteration):
    mapValues = np.zeros(iteration)
       
    
    i = 1
    mapValues[0] = initial
    xPrevious = initial
    while i < iteration:
        
        if xPrevious == 0:
            x = 0
        else:
            x = ( 1 / ( xPrevious % 1.0 ) )
        
        mapValues[i] = x
        
        xPrevious = x
        
        i+=1
    return mapValues

def circleMap(initial,iteration):
    mapValues = np.zeros(iteration)
    a = 0.5
    b = 0.2    
    k = a / ( 2 * math.pi )
    
    
    i = 1
    mapValues[0] = initial
    xPrevious = initial
    while i < iteration:
        
        x = ( xPrevious + b - k * math.sin( 2 * math.pi * xPrevious ) ) % 1.0
        
        mapValues[i] = x
        
        xPrevious = x
        
        i+=1
    return mapValues

def logisticMap(initial,iteration):
    mapValues = np.zeros(iteration)
    a = 4
    i = 1
    mapValues[0] = initial
    xPrevious = initial
    while i < iteration:
        
        x = a * xPrevious * (1 - xPrevious)
        
        mapValues[i] = x
        
        xPrevious = x
        
        i+=1
    return mapValues
    
def piecewiseMap(initial,iteration):
    mapValues = np.zeros(iteration)
    P = 0.4
    i = 1
    mapValues[0] = initial
    xPrevious = initial
    while i < iteration:
        
        if P > xPrevious and xPrevious >= 0:
            x = xPrevious / P
        if 1/2 > xPrevious and xPrevious >= P:
            x = ( xPrevious - P ) / ( 0.5 - P )
        if (1-P) > xPrevious and xPrevious >= 1/2:
            x = ( 1 - P - xPrevious ) / ( 0.5 - P )
        if 1 > xPrevious and xPrevious >= ( 1 - P ):
            x = ( 1 - xPrevious ) / P
        
            
        
        mapValues[i] = x
        
        xPrevious = x
        
        i+=1
    return mapValues

def sineMap(initial,iteration):
    mapValues = np.zeros(iteration)
    a = 4
    i = 1
    mapValues[0] = initial
    xPrevious = initial
    while i < iteration:
        
        x = ( a / 4 ) * ( math.sin( math.pi * xPrevious ) )
        
        mapValues[i] = x
        
        xPrevious = x
        
        i+=1
    return mapValues

def singerMap(initial,iteration):
    mapValues = np.zeros(iteration)
    a = 4
    u = 1.07
    i = 1
    mapValues[0] = initial
    xPrevious = initial
    while i < iteration:
        
        x = u * ( ( 7.86 * xPrevious ) - ( 23.31 * pow(xPrevious,2) ) + ( 28.75 * pow(xPrevious,3) ) - ( 13.302875 * pow(xPrevious,4) ) )
        
        mapValues[i] = x
        
        xPrevious = x
        
        i+=1
    return mapValues

def sinusoidalMap(initial,iteration):
    mapValues = np.zeros(iteration)
    a = 2.3
    i = 1
    mapValues[0] = initial
    xPrevious = initial
    while i < iteration:
        
        x = a * pow(xPrevious,2) * math.sin( math.pi * xPrevious )
        
        mapValues[i] = x
        
        xPrevious = x
        
        i+=1
    return mapValues

def tentMap(initial,iteration):
    mapValues = np.zeros(iteration)
    
    i = 1
    mapValues[0] = initial
    xPrevious = initial
    while i < iteration:
        
        if 0.7 > xPrevious:
            x = xPrevious / 0.7
        if xPrevious >= 0.7:
            x = ( 10 / 3 ) * ( 1 - xPrevious )
        
        mapValues[i] = x
        
        xPrevious = x
        
        i+=1
    return mapValues

def graficarLogisticMap(iteration):
    iterationes = np.zeros(iteration)
    
    for i in range(iteration):
        iterationes[i] = i + 1
        
    plt.plot(iterationes, logisticMap(0.7,iteration), label="logistic Map")
    plt.title("logistic Map")
    plt.xlim(0,iteration)
    plt.ylim(0,1)
    
    plt.xlabel('$iterations (k)$')
    plt.ylabel('$Value (x_{k})$')
    plt.savefig(f"./Resultados/ChaoticMaps/logistic Map.pdf")
    plt.close()
    
def graficarpiecewiseMap(iteration):
    iterationes = np.zeros(iteration)
    
    for i in range(iteration):
        iterationes[i] = i + 1
        
    plt.plot(iterationes, piecewiseMap(0.7,iteration), label="piecewise Map")
    plt.title("piecewise Map")
    plt.xlim(0,iteration)
    plt.ylim(0,1)
    
    plt.xlabel('$iterations (k)$')
    plt.ylabel('$Value (x_{k})$')
    plt.savefig(f"./Resultados/ChaoticMaps/piecewise Map.pdf")
    plt.close()
    
def graficarsineMap(iteration):
    iterationes = np.zeros(iteration)
    
    for i in range(iteration):
        iterationes[i] = i + 1
        
    plt.plot(iterationes, sineMap(0.7,iteration), label="sine Map")
    plt.title("sine Map")
    plt.xlim(0,iteration)
    plt.ylim(0,1)
    
    plt.xlabel('$iterations (k)$')
    plt.ylabel('$Value (x_{k})$')
    plt.savefig(f"./Resultados/ChaoticMaps/sine Map.pdf")
    plt.close()
    
def graficarsingerMap(iteration):
    iterationes = np.zeros(iteration)
    
    for i in range(iteration):
        iterationes[i] = i + 1
        
    plt.plot(iterationes, singerMap(0.7,iteration), label="singer Map")
    plt.title("singer Map")
    plt.xlim(0,iteration)
    plt.ylim(0,1)
    
    plt.xlabel('$iterations (k)$')
    plt.ylabel('$Value (x_{k})$')
    plt.savefig(f"./Resultados/ChaoticMaps/singer Map.pdf")
    plt.close()
    
def graficarsinusoidalMap(iteration):
    iterationes = np.zeros(iteration)
    
    for i in range(iteration):
        iterationes[i] = i + 1
        
    plt.plot(iterationes, sinusoidalMap(0.7,iteration), label="sinusoidal Map")
    plt.title("sinusoidal Map")
    plt.xlim(0,iteration)
    plt.ylim(0,1)
    
    plt.xlabel('$iterations (k)$')
    plt.ylabel('$Value (x_{k})$')
    plt.savefig(f"./Resultados/ChaoticMaps/sinusoidal Map.pdf")
    plt.close()
    
def graficartentMap(iteration):
    iterationes = np.zeros(iteration)
    
    for i in range(iteration):
        iterationes[i] = i + 1
        
    plt.plot(iterationes, tentMap(0.6,iteration), label="tent Map")
    plt.title("tent Map")
    plt.xlim(0,iteration)
    plt.ylim(0,1)
    
    plt.xlabel('$iterations (k)$')
    plt.ylabel('$Value (x_{k})$')
    plt.savefig(f"./Resultados/ChaoticMaps/tent Map.pdf")
    plt.close()

def graficarcircleMap(iteration):
    iterationes = np.zeros(iteration)
    
    for i in range(iteration):
        iterationes[i] = i + 1
        
    plt.plot(iterationes, circleMap(0.7,iteration), label="circle Map")
    plt.title("circle Map")
    plt.xlim(0,iteration)
    plt.ylim(0,1)
    
    plt.xlabel('$iterations (k)$')
    plt.ylabel('$Value (x_{k})$')
    plt.savefig(f"./Resultados/ChaoticMaps/circle Map.pdf")
    
    plt.close()




# graficar(100)

iteraciones = 100

graficarLogisticMap(iteraciones)
graficarcircleMap(iteraciones)
graficarpiecewiseMap(iteraciones)
graficarsineMap(iteraciones)
graficarsingerMap(iteraciones)
graficarsinusoidalMap(iteraciones)
graficartentMap(iteraciones)