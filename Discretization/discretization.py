import math
import random
import numpy as np
import time

from scipy import special as scyesp

def aplicarBinarizacion(ind, transferFunction, binarizationFunction, bestSolutionBin, indBin):
    step1 = transferir(transferFunction, ind)
    individuoBin = binarizar(binarizationFunction, step1, bestSolutionBin, indBin)
    
    return individuoBin

def transferir(transferFunction, dimension):
    if transferFunction == "S1":
        return S1(dimension)
    
    elif transferFunction == "S2":
        return S2(dimension)
    
    elif transferFunction == "S3":
        return S3(dimension)
    
    elif transferFunction == "S4":
        return S4(dimension)
    
    elif transferFunction == "V1":
        return V1(dimension)
    
    elif transferFunction == "V2":
        return V2(dimension)
    
    elif transferFunction == "V3":
        return V3(dimension)
    
    elif transferFunction == "V4":
        return V4(dimension)
    
    elif transferFunction == "X1":
        return X1(dimension)
    
    elif transferFunction == "X2":
        return X2(dimension)
    
    elif transferFunction == "X3":
        return X3(dimension)
    
    elif transferFunction == "X4":
        return X4(dimension)
    
    elif transferFunction == "Z1":
        return Z1(dimension)
    
    elif transferFunction == "Z2":
        return Z2(dimension)
    
    elif transferFunction == "Z3":
        return Z3(dimension)
    
    elif transferFunction == "Z4":
        return Z4(dimension)

def binarizar(binarizationFunction, step1, bestSolutionBin, indBin):
    if binarizationFunction == "STD":
        return Standard(step1)
    
    elif binarizationFunction == "COM":
        return Complement(step1, indBin)
    
    elif binarizationFunction == "PS":
        return ProblabilityStrategy(step1, indBin)
    
    elif binarizationFunction == "ELIT":
        return Elitist(step1, bestSolutionBin)

def S1(dimension):
    return np.divide(1, (1 + np.exp(-2 * dimension)))

def S2(dimension):
    return np.divide(1, (1 + np.exp(-1 * dimension)))

def S3(dimension):
    return np.divide(1, (1 + np.exp(np.divide((-1 * dimension), 2))))

def S4(dimension):
    return np.divide(1, (1 + np.exp(np.divide(( -1 * dimension ), 3))))

def V1(dimension):
    return np.abs(scyesp.erf(np.divide(np.sqrt(np.pi), 2) * dimension))

def V2(dimension):
    return np.abs(np.tanh(dimension))

def V3(dimension):
    return np.abs(np.divide(dimension, np.sqrt(1 + np.power(dimension, 2 ))))

def V4(dimension):
    return np.abs(np.divide(2, np.pi) * np.arctan(np.divide(np.pi, 2 ) * dimension))

def X1(dimension):
    return np.divide(1, (1 + np.exp(2 * dimension)))

def X2(dimension):
    return np.divide(1, (1 + np.exp(dimension)))

def X3(dimension):
    return np.divide(1, (1 + np.exp(np.divide(dimension, 2))))

def X4(dimension):
    return np.divide(1, (1 + np.exp(np.divide(dimension, 3))))

def Z1(dimension):
    return np.power((1 - np.power(2, dimension)), 0.5)

def Z2(dimension):
    return np.power((1 - np.power(5, dimension)), 0.5)

def Z3(dimension):
    return np.power((1 - np.power(8 , dimension)), 0.5)

def Z4(dimension):
    return np.power((1 - np.power(20, dimension)), 0.5)

def Standard(step1):
    # Generar un vector de números aleatorios entre [0, 1] del mismo tamaño que step1_vector
    random_numbers = np.random.rand(len(step1))
    # Comparar cada elemento de step1_vector con los números aleatorios generados
    
    return np.where(step1 >= random_numbers, 1, 0)

def Complement(step1, bin):
    # Generar un vector de números aleatorios entre [0, 1] del mismo tamaño que step1_vector
    random_numbers = np.random.rand(len(step1))
    # Invertir el valor binario: 1 -> 0 y 0 -> 1
    inverted_binary = 1 - np.array(bin)
    # Comparar step1_vector con los números aleatorios
    # Si step1 >= random, devolver el opuesto del valor binario, si no, devolver 0
    
    return np.where(step1 >= random_numbers, inverted_binary, 0) 

def Elitist(step1, bestBin):
    # Generar un vector de números aleatorios entre [0, 1] del mismo tamaño que step1_vector
    random_numbers = np.random.rand(len(step1))
    bestBin = np.array(bestBin)
    # Comparar step1_vector con los números aleatorios
    # Si step1 >= random, devolver la mejor solucion, si no, devolver 0
    
    return np.where(step1 >= random_numbers, bestBin, 0)

def ProblabilityStrategy(step1, bin):
    alpha = 1/3
    
    limit2 = (1/2) * (1 + alpha)
    bin = np.array(bin)
    # Condiciones vectorizadas:
    # 1. Si alpha < step1 <= limit2, devolver solucion actual
    # 2. Si step1 >= limit2, devolver 1
    # 3. Si step1 <= alpha, devolver 0
    return np.where(step1 >= limit2, 1, np.where((step1 > alpha) & (step1 <= limit2), bin, 0))