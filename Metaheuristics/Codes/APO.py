# -*- coding: utf-8 -*-
import random
import numpy as np
import math

def iterarAPO(maxIter, population, dim, fitness):
  #Definición párametros iniciales
  X, dim = population, dim
  poblacion = population.shape[0]
  vecinos_par = 2
  proporcion_max = 5
  max_FE = 10
  FE = 0
  iter_max = maxIter
  prob_ayh = 0.5
  prob_dyr = 0.5
  protozoa_inferior = -100
  protozoa_superior = 100
  eps = 1e-6
  fitness_values = fitness

  #fitness_calculator
  def calculator(X):
      return np.sum(X**2)

  def fitness_calculator(X):
      f_x = calculator(X)
      f_x_eps = calculator(X + eps)

      if f_x_eps == 0:   # evitar división por cero
          f_x_eps = eps

      ratio = (f_x / f_x_eps)
      return math.exp(-abs(ratio))

  #Factor de Forrajeo
  i = np.random.uniform(0, 1)
  f = np.random.uniform(0, 1) * (1 + math.cos((i/ iter_max) * math.pi))

  #Vector Rand
  vector = np.zeros(dim)
  for i in range(dim):
    vector[i] = np.random.uniform(0, 1)

  #Sorteo
  def sorteo_protozoos(X, fitness_values):
    sorted_indices = np.argsort(fitness_values)  # ordena de menor a mayor
    X_sorted = X[sorted_indices]
    fitness_sorted = fitness_values[sorted_indices]
    return X_sorted, fitness_sorted

  #Index Dormancia//Reproducción
  def index_eleccion(poblacion, proporcion):
    ps = poblacion
    pf = proporcion

    num_selected = int(np.ceil(ps * pf))
    num_selected = min(num_selected, ps)
    return np.random.choice(ps, num_selected, replace=True)

  #Mapeo Reproducción
  def set_mapeo_reproduccion():

    Mr = np.zeros(dim)
    num_actual = int(np.ceil(dim * np.random.uniform(0, 1)))
    indices_update = np.random.choice(dim, num_actual, replace=False)
    Mr[indices_update] = 1
    return Mr

    print("Vector Mr:", Mr)

  #Mapeo Forrajeo
  def set_mapeo_forrajeo():

    Mf = np.zeros(dim)
    num_actual = int(np.ceil(dim * (1/poblacion)))
    indices_update = np.random.choice(dim, num_actual, replace=False)
    Mf[indices_update] = 1
    return Mf

    print("Vector Mf:", Mf)

  #Protozoa cercano
  def cercano(protozoa_actual):
    for i in range(dim):
      x_n = (1 + vector[i] *(1 - i/iter_max)) * protozoa_actual
    return x_n

  def peso_autotrofico(protozoa_actual, X, i):
    #Seleccionar un vecino aleatorio
    k = np.random.choice([idx for idx in range(len(X)) if idx != i])
    #Cálculo de fitness_calculator
    fitness_i = fitness_calculator(protozoa_actual)
    fitness_k = fitness_calculator(X[k])

    if fitness_k == 0:
        w_a = 1
    else:
        w_a = math.exp(-(abs(fitness_i / fitness_k)))
    return w_a

  #Ecuación 1 (forrajeo autotrófico)
  def ecuacion_1(protozoa_actual, peso, X, i):
    j = np.random.choice([idx for idx in range(poblacion) if idx != i])
    protozoa_j = X[j]
    Mf = set_mapeo_forrajeo()

    sum_vecinos = np.zeros(dim)
    for i in range(vecinos_par):
      #Seleccion de vecinos
      indices_vecinos = np.random.choice([idx for idx in range(poblacion) if idx != i], 2, replace=False)
      k_minus_idx = indices_vecinos[0]
      k_plus_idx = indices_vecinos[1]

      sum_vecinos += peso * (X[k_minus_idx] - X[k_plus_idx])

    nuevo_actual = protozoa_actual + f * ((protozoa_j) - protozoa_actual + (1/vecinos_par) * sum_vecinos) * Mf

    return nuevo_actual

  #Calculo de peso heterotrófico
  def peso_heterotrofico(protozoa_actual, X, i):
    #Seleccionar un vecino aleatorio
    k = np.random.choice([idx for idx in range(len(X)) if idx != i])
    #Cálculo de fitness_calculator
    fitness_i = fitness_calculator( protozoa_actual)
    fitness_i_plus_k = fitness_calculator(X[(i + k) % len(X)])
    fitness_i_minus_k = fitness_calculator(X[(i - k) % len(X)])

    if fitness_i_plus_k == 0:
        w_h = 1
    else:
        w_h = math.exp(-(abs(fitness_i_minus_k / fitness_i_plus_k)))
    return w_h

  #Ecuación 7 (forrajeo heterotrófico)
  def ecuacion_7(protozoa_actual, peso, X, i):
    x_n = cercano(protozoa_actual)
    Mf = set_mapeo_forrajeo()

    sum_vecinos = np.zeros(dim)
    for i in range(vecinos_par):
      # Selección de vecinos
      indices_vecinos = np.random.choice([idx for idx in range(poblacion) if idx != i], 2, replace=False)
      k_minus_idx = indices_vecinos[0]
      k_plus_idx = indices_vecinos[1]

      sum_vecinos += peso * (X[k_minus_idx] - X[k_plus_idx])

    nuevo_actual = protozoa_actual + f * (x_n - protozoa_actual + (1/vecinos_par) * sum_vecinos) * Mf

    return nuevo_actual

  #Ecuación 11 (Dormancia)
  def ecuacion_11(protozoa_actual):
    nuevo_actual = protozoa_inferior + vector * (protozoa_superior - protozoa_inferior)
    return nuevo_actual

  #Ecuación 13 (Reproducción)
  def ecuacion_13(protozoa_actual):
    random = np.random.uniform(0, 1)
    mapeo_reproduccion = set_mapeo_reproduccion()
    nuevo_actual = protozoa_actual + random * (protozoa_inferior + vector * (protozoa_inferior - protozoa_superior)) * mapeo_reproduccion
    return nuevo_actual

  while FE < max_FE:
    X, fitness_values = sorteo_protozoos(X, fitness_values)
    proporcion = proporcion_max * np.random.uniform(0, 1)
    index_dyr = index_eleccion(poblacion, proporcion)

    for i in range(poblacion):
      protozoa_actual = X[i]

      if i in index_dyr: #Dormancia o Reproducción
        if prob_dyr > np.random.uniform(0, 1):
          nuevo_actual = ecuacion_11(protozoa_actual)#Ecuación 11

        else:
          set_mapeo_reproduccion()
          nuevo_actual = ecuacion_13(protozoa_actual)#Ecuación 13

      else: #Autotrofia o Heterotrifia
        set_mapeo_forrajeo()
        if prob_ayh > random.uniform(0, 1):
          w_a = peso_autotrofico(protozoa_actual, X, i)
          nuevo_actual = ecuacion_1(protozoa_actual, w_a, X, i)#Ecuación 1

        else:
          w_h = peso_heterotrofico(protozoa_actual, X, i)
          nuevo_actual = ecuacion_7(protozoa_actual, w_h, X, i)#Ecuación 7

      if fitness_calculator(nuevo_actual) < fitness_calculator(protozoa_actual):
        X[i] = nuevo_actual
      
    FE += 1

  return X