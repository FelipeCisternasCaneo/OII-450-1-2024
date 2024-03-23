import math
import random
import numpy as np 
from scipy import special as scyesp
import time

def aplicarBinarizacion(ind, transferFunction, binarizationFunction, bestSolutionBin, indBin, iter, pop, maxIter, pos_ind, chaotic_map):
    individuoBin = []
    for j in range(ind.__len__()):
        individuoBin.append(0)

    for j in range(ind.__len__()):
        step1 = transferir(transferFunction, ind[j])
        individuoBin[j] = binarizar(binarizationFunction, step1, bestSolutionBin[j], indBin[j], iter, pop, maxIter, pos_ind, j, len(ind), chaotic_map)
    return np.array(individuoBin)

def transferir(transferFunction, dimension):
    if transferFunction == "S1":
        return S1(dimension)
    if transferFunction == "S2":
        return S2(dimension)
    if transferFunction == "S3":
        return S3(dimension)
    if transferFunction == "S4":
        return S4(dimension)
    if transferFunction == "V1":
        return V1(dimension)
    if transferFunction == "V2":
        return V2(dimension)
    if transferFunction == "V3":
        return V3(dimension)
    if transferFunction == "V4":
        return V4(dimension)
    if transferFunction == "X1":
        return X1(dimension)
    if transferFunction == "X2":
        return X2(dimension)
    if transferFunction == "X3":
        return X3(dimension)
    if transferFunction == "X4":
        return X4(dimension)
    if transferFunction == "Z1":
        return Z1(dimension)
    if transferFunction == "Z2":
        return Z2(dimension)
    if transferFunction == "Z3":
        return Z3(dimension)
    if transferFunction == "Z4":
        return Z4(dimension)


def binarizar(binarizationFunction, step1, bestSolutionBin, indBin, iter, pop, maxIter, pos_ind, j, dim, chaotic_map):
    if binarizationFunction == "STD":
        return Standard(step1)
    if binarizationFunction == "STD_LOG" or binarizationFunction == "STD_PIECE" or binarizationFunction == "STD_SINE" or binarizationFunction == "STD_SINGER" or binarizationFunction == "STD_SINU" or binarizationFunction == "STD_TENT" or binarizationFunction == "STD_CIRCLE":
        return Standard_Map(step1, iter, pop, pos_ind, j, dim, chaotic_map)
    
    if binarizationFunction == "COM":
        return Complement(step1, indBin)
    if binarizationFunction == "COM_LOG" or binarizationFunction == "COM_PIECE" or binarizationFunction == "COM_SINE" or binarizationFunction == "COM_SINGER" or binarizationFunction == "COM_SINU" or binarizationFunction == "COM_TENT" or binarizationFunction == "COM_CIRCLE":
        return Complement_Map(step1, indBin, iter, pop, pos_ind, j, dim, chaotic_map)
    
    if binarizationFunction == "PS":
        return ProblabilityStrategy(step1, indBin)
    if binarizationFunction == "PS_LOG" or binarizationFunction == "PS_PIECE" or binarizationFunction == "PS_SINE" or binarizationFunction == "PS_SINGER" or binarizationFunction == "PS_SINU" or binarizationFunction == "PS_TENT" or binarizationFunction == "PS_CIRCLE":
        return ProblabilityStrategy_Map(step1, indBin, iter, pop, pos_ind, j, dim, chaotic_map)
    
    if binarizationFunction == "ELIT":
        return Elitist(step1, bestSolutionBin)
    if binarizationFunction == "ELIT_LOG" or binarizationFunction == "ELIT_PIECE" or binarizationFunction == "ELIT_SINE" or binarizationFunction == "ELIT_SINGER" or binarizationFunction == "ELIT_SINU" or binarizationFunction == "ELIT_TENT" or binarizationFunction == "ELIT_CIRCLE":
        return Elitist_Map(step1, bestSolutionBin, iter, pop, pos_ind, j, dim, chaotic_map)


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

def Standard_Map(step1, iter, pop, pos_ind, j, dim, chaotic_map):
    pivote = (iter * pop * dim) + (pos_ind * dim)
    pos = pivote + j 
    rand = chaotic_map[pos]
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

def Complement_Map(step1, bin, iter, pop, pos_ind, j, dim, chaotic_map):
    pivote = (iter * pop * dim) + (pos_ind * dim)
    pos = pivote + j 
    rand = chaotic_map[pos]
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

def ProblabilityStrategy_Map(step1, bin, iter, pop, pos_ind, j, dim, chaotic_map):
    pivote = (iter * pop * dim) + (pos_ind * dim)
    pos = pivote + j 
    alpha = chaotic_map[pos]
    
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

def Elitist_Map(step1, bestBin, iter, pop, pos_ind, j, dim, chaotic_map):
    pivote = (iter * pop * dim) + (pos_ind * dim)
    pos = pivote + j 
    rand = chaotic_map[pos]
    binario = 0
    if rand < step1:
        binario = bestBin
    return binario



