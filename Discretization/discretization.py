import math
import random
import numpy as np 
from scipy import special as scyesp
import time

def aplicarBinarizacion(ind, transferFunction, binarizationFunction, bestSolutionBin, indBin):
    individuoBin = []
    for j in range(ind.__len__()):
        individuoBin.append(0)
    for j in range(ind.__len__()):
        step1 = transferir(transferFunction, ind[j])
        individuoBin[j] = binarizar(binarizationFunction, step1, bestSolutionBin[j], indBin[j])
    return np.array(individuoBin)

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
    return np.divide( 1 , ( 1 + np.exp( -2 * dimension ) ) )
def S2(dimension):
    return np.divide( 1 , ( 1 + np.exp( -1 * dimension ) ) )
def S3(dimension):
    return np.divide( 1 , ( 1 + np.exp( np.divide( ( -1 * dimension ) , 2 ) ) ) )
def S4(dimension):
    return np.divide( 1 , ( 1 + np.exp( np.divide( ( -1 * dimension ) , 3 ) ) ) )
def V1(dimension):
    return np.abs( scyesp.erf( np.divide( np.sqrt( np.pi ) , 2 ) * dimension ) )
def V2(dimension):
    return np.abs( np.tanh( dimension ) )
def V3(dimension):
    return np.abs( np.divide( dimension , np.sqrt( 1 + np.power( dimension , 2 ) ) ) )
def V4(dimension):
    return np.abs( np.divide( 2 , np.pi ) * np.arctan( np.divide( np.pi , 2 ) * dimension ) )
def X1(dimension):
    return np.divide( 1 , ( 1 + np.exp( 2 * dimension ) ) )
def X2(dimension):
    return np.divide( 1 , ( 1 + np.exp( dimension ) ) )
def X3(dimension):
    return np.divide( 1 , ( 1 + np.exp( np.divide( dimension , 2 ) ) ) )
def X4(dimension):
    return np.divide( 1 , ( 1 + np.exp( np.divide( dimension , 3 ) ) ) )
def Z1(dimension):
    return np.power( ( 1 - np.power( 2 , dimension ) ) , 0.5 )
def Z2(dimension):
    return np.power( ( 1 - np.power( 5 , dimension ) ) , 0.5 )
def Z3(dimension):
    return np.power( ( 1 - np.power( 8 , dimension ) ) , 0.5 )
def Z4(dimension):
    return np.power( ( 1 - np.power( 20 , dimension ) ) , 0.5 )

def Standard(step1):
    rand = random.uniform(0.0, 1.0)
    binario = 0
    if rand <= step1:
        binario = 1
    return binario

def Complement(step1, bin):
    rand = random.uniform(0.0, 1.0)
    binario = 0
    if rand <= step1:
        if bin == 1:
            binario = 0
        if bin == 0:
            
            binario =  1
    return binario

def ProblabilityStrategy(step1, bin):
    alpha = 1/3
    binario = 0
    if alpha < step1 and step1 <= ( ( 1/2 ) * ( 1 + alpha ) ):
        binario = bin
    if step1 > ( ( 1/2 ) * ( 1 + alpha ) ):
        binario = 1
    return binario

def Elitist(step1, bestBin):
    rand = random.uniform(0.0, 1.0)
    binario = 0
    if rand < step1:
        binario = bestBin
    return binario