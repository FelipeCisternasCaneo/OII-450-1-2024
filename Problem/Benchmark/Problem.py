import numpy as np 
import math
import opfunu.cec_based

from BD.sqlite import BD

def fitness(problem, individual):
    """
    Evalúa la función de fitness para un individuo o una población.
    """
    # Asegurarnos de que estamos trabajando con un individuo
    if isinstance(individual, np.ndarray) and len(individual.shape) == 2:
        raise ValueError("La función 'fitness' solo admite un individuo, no una población completa. Asegúrate de aplicar correctamente.")
    
    fitness_value = 0

    def opfunu_cec_function(x):
        func_class = getattr(opfunu.cec_based, f"{problem}")
        return func_class().evaluate(x)

    # Asegurar que `problem` sea un string válido
    if isinstance(problem, str):
        if problem in BD.data:
            try:
                fitness_value = globals()[problem](individual)
            except KeyError:
                raise ValueError(f"La función '{problem}' no está definida en el contexto global.")
        elif problem in BD.opfunu_cec_data:
            fitness_value = opfunu_cec_function(individual)
        else:
            raise ValueError(f"El problema '{problem}' no es reconocido.")
    else:
        raise TypeError("El parámetro 'problem' debe ser un string que identifique la función objetivo.")

    return fitness_value

# define the function blocks
def prod(it):
    p = 1
    
    for n in it:
        p *= n
        
    return p

def Ufun(x, a, k, m):
    y = k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))
    
    return y

def F1(x):  #Sphere Function (CEC 2005 F1)
    s = np.sum(x ** 2)
    
    return s

def F2(x):  #Schwefel's Problem 2.22
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def F3(x):  #Schwefel's Function No.1.2 (Double-Sum or Rotated Hyper-Ellipsoid Function) (CEC 2005 F2)
    dim = len(x) + 1
    o = 0
    
    for i in range(1, dim):
        o = o + (np.sum(x[0:i])) ** 2

    return o

def F4(x):  #Schwefel's Function No.2.21 (or MaxMod Function) (CEC 2008 F2)
    o = max(abs(x))
    
    return o

def F5(x):  #Rosenbrock's Function (CEC 2005 F6)
    dim = len(x)
    o = np.sum(
        100 * (x[1:dim] - (x[0 : dim - 1] ** 2)) ** 2 + (x[0 : dim - 1] - 1) ** 2
        )
    
    return o

def F6(x):  #Shifted Function (CEC 2005 F1)
    o = np.sum(abs((x + 0.5)) ** 2)
    
    return o

def F7(x):  #Quartic (or Modified 4th De Jong's) Function With Noise 
    dim = len(x)

    w = [i for i in range(len(x))]
    
    for i in range(0, dim):
        w[i] = i + 1
        
    o = np.sum(w * (x ** 4)) + np.random.uniform(0, 1)
    
    return o

def F8(x): #Schwefel's Function No.2.26
    o = sum(-x * (np.sin(np.sqrt(abs(x)))))
    
    return o

def F9(x):  #Rastrigin's Function (CEC 2005 F9)
    dim = len(x)
    o = np.sum(x ** 2 - 10 * np.cos(2 * math.pi * x)) + 10 * dim
    
    return o

def F10(x): #Ackley's Function No.01 (or Ackley's Path Function) (CEC 2014 F5)
    dim = len(x)
    o = (
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / dim))
        - np.exp(np.sum(np.cos(2 * math.pi * x)) / dim)
        + 20
        + np.exp(1)
    )
    
    return o

def F11(x): #Griewank's Function (CEC 2014 F7)
    w = [i for i in range(len(x))]
    w = [i + 1 for i in w]
    o = np.sum(x ** 2) / 4000 - prod(np.cos(x / np.sqrt(w))) + 1
    
    return o

def F12(x): #Generalized Penalized Function No.01
    dim = len(x)
    o = (math.pi / dim) * (
        10 * ((np.sin(math.pi * (1 + (x[0] + 1) / 4))) ** 2)
        + np.sum(
            (((x[: dim - 1] + 1) / 4) ** 2)
            * (1 + 10 * ((np.sin(math.pi * (1 + (x[1 :] + 1) / 4)))) ** 2)
        )
        + ((x[dim - 1] + 1) / 4) ** 2
    ) + np.sum(Ufun(x, 10, 100, 4))
    
    return o

def F13(x): #Generalized Penalized Function No.02
    if x.ndim==1:
        x = x.reshape(1,-1)

    o = 0.1 * (
        (np.sin(3 * np.pi * x[:, 0])) ** 2
        + np.sum(
            (x[:, :-1] - 1) ** 2
            * (1 + (np.sin(3 * np.pi * x[:,1:])) ** 2), axis = 1
        )
        + ((x[:, -1] - 1) ** 2) * (1 + (np.sin(2 * np.pi * x[:, -1])) ** 2)
    ) + np.sum(Ufun(x, 5, 100, 4))
    
    return o

def F14(x): #Shekel's Foxholes Function
    aS = [
        [
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
        ],
        [
            -32, -32, -32, -32, -32,
            -16, -16, -16, -16, -16,
            0, 0, 0, 0, 0,
            16, 16, 16, 16, 16,
            32, 32, 32, 32, 32,
        ],
    ]
    
    aS = np.asarray(aS)
    bS = np.zeros(25)
    v = np.matrix(x)
    
    for i in range(0, 25):
        H = v - aS[:, i]
        bS[i] = np.sum((np.power(H, 6)))
        
    w = [i for i in range(25)]
    
    for i in range(0, 24):
        w[i] = i + 1
    
    o = ((1.0 / 500) + np.sum(1.0 / (w + bS))) ** (-1)
    
    return o

def F15(L): #Kowalik Function
    aK = [
        0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627,
        0.0456, 0.0342, 0.0323, 0.0235, 0.0246,
    ]
    
    bK = [0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16]
    
    aK = np.asarray(aK)
    bK = np.asarray(bK)
    bK = 1 / bK
    
    fit = np.sum(
        (aK - ((L[0] * (bK ** 2 + L[1] * bK)) / (bK ** 2 + L[2] * bK + L[3]))) ** 2
    )
    
    return fit

def F16(L): #Six-Hump Camel-Back Function
    o = (
        4 * (L[0] ** 2)
        - 2.1 * (L[0] ** 4)
        + (L[0] ** 6) / 3
        + L[0] * L[1]
        - 4 * (L[1] ** 2)
        + 4 * (L[1] ** 4)
    )
    
    return o

def F17(L): #Branin's RCOS Function No.01
    o = (
        (L[1] - (L[0] ** 2) * 5.1 / (4 * (np.pi ** 2)) + 5 / np.pi * L[0] - 6)
        ** 2
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(L[0])
        + 10
    )
    
    return o

def F18(L): #Goldstein-Price's Function
    o = (
        1
        + (L[0] + L[1] + 1) ** 2
        * (
            19
            - 14 * L[0]
            + 3 * (L[0] ** 2)
            - 14 * L[1]
            + 6 * L[0] * L[1]
            + 3 * L[1] ** 2
        )
    ) * (
        30
        + (2 * L[0] - 3 * L[1]) ** 2
        * (
            18
            - 32 * L[0]
            + 12 * (L[0] ** 2)
            + 48 * L[1]
            - 36 * L[0] * L[1]
            + 27 * (L[1] ** 2)
        )
    )
    
    return o

# map the inputs to the function blocks

def F19(L): #Hartman's Function No.01 (or Hartmann 3D Function)
    aH = [[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]]
    aH = np.asarray(aH)
    cH = [1, 1.2, 3, 3.2]
    cH = np.asarray(cH)
    
    pH = [
        [0.3689, 0.117, 0.2673],
        [0.4699, 0.4387, 0.747],
        [0.1091, 0.8732, 0.5547],
        [0.03815, 0.5743, 0.8828],
    ]
    
    pH = np.asarray(pH)
    o = 0
    
    for i in range(0, 4):
        o = o - cH[i] * np.exp(-(np.sum(aH[i, :] * ((L - pH[i, :]) ** 2))))
        
    return o

def F20(L): #Hartman's Function No.02 (or Hartmann 6D Function)
    aH = [
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ]
    
    aH = np.asarray(aH)
    cH = [1, 1.2, 3, 3.2]
    cH = np.asarray(cH)
    
    pH = [
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
    ]
    
    pH = np.asarray(pH)
    o = 0
    
    for i in range(0, 4):
        o = o - cH[i] * np.exp(-(np.sum(aH[i, :] * ((L - pH[i, :]) ** 2))))
        
    return o

def F21(L): #Shekel’s Function (Variant No. 5)
    aSH = [
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6],
    ]
    
    cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    
    aSH = np.asarray(aSH)
    cSH = np.asarray(cSH)
    fit = 0
    
    for i in range(5):
        v = np.matrix(L - aSH[i, :])
        fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
        
    o = fit.item(0)
    
    return o

def F22(L): #Shekel’s Function (Variant No. 7)
    aSH = [
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6],
    ]
    
    cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    
    aSH = np.asarray(aSH)
    cSH = np.asarray(cSH)
    fit = 0
    
    for i in range(7):
        v = np.matrix(L - aSH[i, :])
        fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
    o = fit.item(0)
    
    return o

def F23(L): #Shekel’s Function (Variant No. 10)
    aSH = [
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6],
    ]
    
    cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    
    aSH = np.asarray(aSH)
    cSH = np.asarray(cSH)
    fit = 0
    
    for i in range(10):
        v = np.matrix(L - aSH[i, :])
        fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
        
    o = fit.item(0)
    
    return o