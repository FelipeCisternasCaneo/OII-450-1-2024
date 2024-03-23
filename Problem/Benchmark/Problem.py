import numpy as np 
import math

# define the function blocks
def prod(it):
    p = 1
    for n in it:
        p *= n
    return p

def fitness(problem, individual):
    fitness = 0
    if problem == 'F1':
        fitness = F1(individual)
    if problem == 'F2':
        fitness = F2(individual)
    if problem == 'F3':
        fitness = F3(individual)
    if problem == 'F4':
        fitness = F4(individual)
    if problem == 'F5':
        fitness = F5(individual)
    if problem == 'F6':
        fitness = F6(individual)
    if problem == 'F7':
        fitness = F7(individual)
    if problem == 'F8':
        fitness = F8(individual)
    if problem == 'F9':
        fitness = F9(individual)
    if problem == 'F10':
        fitness = F10(individual)
    if problem == 'F11':
        fitness = F11(individual)
    
    return fitness


def F1(x):
    s = np.sum(x ** 2)
    return s    

def F2(x):
    o = sum(abs(x)) + prod(abs(x))
    return o

def F3(x):
    dim = len(x) + 1
    o = 0
    for i in range(1, dim):
        o = o + (np.sum(x[0:i])) ** 2
    return o

def F4(x):
    o = max(abs(x))
    return o


def F5(x):
    dim = len(x)
    o = np.sum(
        100 * (x[1:dim] - (x[0 : dim - 1] ** 2)) ** 2 + (x[0 : dim - 1] - 1) ** 2
    )
    return o


def F6(x):
    o = np.sum(abs((x + 0.5)) ** 2)
    return o


def F7(x):
    dim = len(x)

    w = [i for i in range(len(x))]
    for i in range(0, dim):
        w[i] = i + 1
    o = np.sum(w * (x ** 4)) + np.random.uniform(0, 1)
    return o


def F8(x):
    o = sum(-x * (np.sin(np.sqrt(abs(x)))))
    return o


def F9(x):
    dim = len(x)
    o = np.sum(x ** 2 - 10 * np.cos(2 * math.pi * x)) + 10 * dim
    return o

def F10(x):
    dim = len(x)
    o = (
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / dim))
        - np.exp(np.sum(np.cos(2 * math.pi * x)) / dim)
        + 20
        + np.exp(1)
    )
    return o


def F11(x):
    dim = len(x)
    w = [i for i in range(dim)]
    w = [i + 1 for i in w]
    o = np.sum(x ** 2) / 4000 - prod(np.cos(x / np.sqrt(w))) + 1
    return o