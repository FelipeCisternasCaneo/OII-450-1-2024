import numpy as np
import matplotlib.pyplot as plt

# Puma Optimizer (PO)
# https://doi.org/10.1007/s10586-023-04221-5

class IterarPO:
    def __init__(self, cost_function, dim, n_sol, max_iter, lb, ub):
        self._cost_function = cost_function
        self._dim = dim
        self._n_sol = n_sol
        self._max_iter = max_iter
        self._lb = lb
        self._ub = ub
        self._unselected = [1, 1]  # 1: Exploración, 2: Explotación
        self._f3_explore = 0
        self._f3_exploit = 0
        self._seq_time_explore = [1, 1, 1]
        self._seq_time_exploit = [1, 1, 1]
        self._seq_cost_explore = [1, 1, 1]
        self._seq_cost_exploit = [1, 1, 1]
        self._score_explore = 0
        self._score_exploit = 0
        self._pf = [0.5, 0.1, 0.9]  # 1 & 2 para intensificación (F1 y F2), 3 para diversificación (F3)
        self._pf_f3 = []
        self._mega_explor = 0.99 # Alpha de exploracion
        self._mega_exploit = 0.99 # Alpha de explotacion
        self._sol = []
        self._flag_change = 1
        self._initialBest =[]
        self._costs_explor = []
        self._costs_exploit = []
        self._convergence = []
        self._best = []
        self._Q = 0.67
        self._beta = 2
        self._pCR = 0.20
        self._PCR = 1 - self._pCR
        self._cost_explore = []
        self._cost_exploit = []
        self._use_exploit = []
        self._use_explore = []
    
    # Crear la poblacion que tenga los requisitos que necesita PO para utilizarlo
    def pob(self, pob, iter):
        poblation = np.array(pob)

        for row in poblation:
            posible_pob, cost = self._cost_function(row)
            self._sol.append({'X': posible_pob, 'Cost': cost})
        
        self._best = min(self._sol, key=lambda x: x['Cost'])

        # Si es la primera iteracion, guardar el mejor costo inicial
        if iter == 1:
            self._initialBest = self._best['Cost']

    # Covertir la poblacion del PO a uno que pueda entender Solver
    def _convert(self):
        poblacion = []
        solucion = []
        for i in range(self._n_sol):
            for j in range(self._dim):
                solucion.append(self._sol[i]['X'][j])
            poblacion.append(solucion)
            solucion = []

        return np.array(poblacion)

    # fase de explotación
    def _exploitation(self, Iter):
        NewSol = [None] * self._n_sol

        for i in range(self._n_sol):
            beta1 = 2 * np.random.rand()
            beta2 = np.random.randn(self._dim)
            w = np.random.randn(self._dim)  # Eq 37
            v = np.random.randn(self._dim)  # Eq 38
            F1 = np.random.randn(self._dim) * np.exp(2 - Iter * (2 / self._max_iter))  # Eq 35
            F2 = w * v ** 2 * np.cos((2 * np.random.rand()) * w)  # Eq 36

            # Corrección aquí: accediendo a la clave 'X'
            mbest = np.mean([sol['X'] for sol in self._sol]) / self._n_sol
            R_1 = 2 * np.random.rand() - 1  # Eq 34
            S1 = 2 * np.random.rand() - 1 + np.random.randn(self._dim)
            S2 = F1 * R_1 * self._sol[i]['X'] + F2 * (1 - R_1) * self._best['X']
            VEC = S2 / S1

            if np.random.rand() <= 0.5:
                Xatack = VEC
                
                if np.random.rand() > self._Q:
                    NewSol[i] = {'X': self._best['X'] + beta1 * np.exp(beta2) * (self._sol[np.random.randint(self._n_sol)]['X'] - self._sol[i]['X'])}
                    
                else:
                    NewSol[i] = {'X': beta1 * Xatack - self._best['X']}
            
            else:
                r1 = np.random.randint(self._n_sol)  # Eq 33
                NewSol[i] = {'X': (mbest * self._sol[r1]['X'] - ((-1) ** np.random.randint(2)) * self._sol[i]['X']) / (1 + (self._beta * np.random.rand()))}

            # Apply boundary constraints
            NewSol[i]['X'] = np.maximum(NewSol[i]['X'], self._lb)
            NewSol[i]['X'] = np.minimum(NewSol[i]['X'], self._ub)

            # Evaluate new solution
            Nashe, NewSol[i]['Cost'] = self._cost_function(NewSol[i]['X'])

            contendiente_numero_1 = NewSol[i]['Cost']
            contendiente_numero_2 = self._sol[i]['Cost']

            # Update solution if the new one is better
            if contendiente_numero_1 < contendiente_numero_2:
                self._sol[i] = NewSol[i]
            
            return self._sol

    # Exploracion
    def _exploration(self):
        # Ordenar soluciones por su coste
        self._sol = sorted(self._sol, key=lambda x: x['Cost'])
        p = self._PCR / self._n_sol  # Eq 29

        NewSol = [{'X': None, 'Cost': None} for _ in range(self._n_sol)]  # Inicializar nuevas soluciones

        for i in range(self._n_sol):
            x = self._sol[i]['X']
            A = np.random.permutation(self._n_sol)
            A = A[A != i]  # Eliminar el índice actual

            # Asegurarse de seleccionar índices únicos para a, b, c, d, e, f
            a, b, c, d, e, f = A[:6]

            G = 2 * np.random.rand() - 1  # Eq 26

            # Condición para generar el vector `y`
            if np.random.rand() < 0.5:
                y = np.random.rand(self._dim) * (self._ub - self._lb) + self._lb  # Eq 25
            
            else:
                y = (self._sol[a]['X'] + G * (self._sol[a]['X'] - self._sol[b]['X']) +
                    G * ((self._sol[a]['X'] - self._sol[b]['X']) - (self._sol[c]['X'] - self._sol[d]['X'])) +
                    ((self._sol[c]['X'] - self._sol[d]['X']) - (self._sol[e]['X'] - self._sol[f]['X'])))  # Eq 25

            # Aplicar límites a `y`
            y = np.clip(y, self._lb, self._ub)  # Esto reemplaza las dos líneas de máximo y mínimo

            z = np.zeros_like(x)
            j0 = np.random.randint(len(x))

            # Generar el nuevo vector `z`
            for j in range(len(x)):
                if j == j0 or np.random.rand() <= self._pCR:
                    z[j] = y[j]
                
                else:
                    z[j] = x[j]

            # Crear nueva solución
            NewSol[i]['X'] = np.array(z)
            Nashe, NewSol[i]['Cost'] = self._cost_function(NewSol[i]['X'])
            
            # Comparar la nueva solución con la anterior
            contendiente_numero_1 = NewSol[i]['Cost']
            contendiente_numero_2 = self._sol[i]['Cost']

            if contendiente_numero_1 < contendiente_numero_2:
                self._sol[i] = NewSol[i]
            
            else:
                self._pCR += p  # Eq 30
            
            return self._sol

    # Optimización con PO
    def optimizer(self, iter):
        iter = iter - 1
        
        if iter < 3:
            sol_explor = self._exploration()
            self._cost_explore.append(min([sol['Cost'] for sol in sol_explor]))

            sol_exploit = self._exploitation(iter)
            self._cost_exploit.append(min([sol['Cost'] for sol in sol_exploit]))

            self._sol += sol_explor + sol_exploit
            self._sol.sort(key=lambda x: x['Cost'])
            self._sol = self._sol[:self._n_sol]
            self._best = self._sol[0]

            self._convergence.append(self._best['Cost'])
            return self._convert()
        
        else:
            if iter == 3:
                # Fase de inexperiencia
                self._seq_cost_explore[0] = abs(self._initialBest - self._cost_explore[0])
                self._seq_cost_exploit[0] = abs(self._initialBest - self._cost_exploit[0])
                self._seq_cost_explore[1] = abs(self._cost_explore[1] - self._cost_explore[0])
                self._seq_cost_exploit[1] = abs(self._cost_exploit[1] - self._cost_exploit[0])
                self._seq_cost_explore[2] = abs(self._cost_explore[2] - self._cost_explore[1])
                self._seq_cost_exploit[2] = abs(self._cost_exploit[2] - self._cost_exploit[1])

                for i in range(3):
                    self._use_exploit.append(1)
                    self._use_explore.append(1)
                    
                    if self._seq_cost_explore[i] != 0:
                        self._pf_f3.append(self._seq_cost_explore[i])
                        
                    if self._seq_cost_exploit[i] != 0:
                        self._pf_f3.append(self._seq_cost_exploit[i])

                if self._pf_f3 == []:
                    self._pf_f3 = [np.random.rand()]

                # Cálculos para F1 y F2
                f1_explor = self._pf[0] * (self._seq_cost_explore[0] / self._seq_time_explore[0])
                f1_exploit = self._pf[0] * (self._seq_cost_exploit[0] / self._seq_time_exploit[0])
                f2_explor = self._pf[1] * (sum(self._seq_cost_explore) / sum(self._seq_time_explore))
                f2_exploit = self._pf[1] * (sum(self._seq_cost_exploit) / sum(self._seq_time_exploit))

                self._score_explore = (self._pf[0] * f1_explor) + (self._pf[1] * f2_explor)
                self._score_exploit = (self._pf[0] * f1_exploit) + (self._pf[1] * f2_exploit)
            
            # Fase de experiencia
            if self._score_explore > self._score_exploit:
                self._use_explore.append(1)
                self._use_exploit.append(0)
                select_flag = 1
                self._sol = self._exploration()
                count_select = self._unselected
                self._unselected[1] += 1
                self._unselected[0] = 1
                self._f3_explore = self._pf[2]
                self._f3_exploit += self._pf[2]

            else:
                self._use_exploit.append(1)
                self._use_explore.append(0)
                select_flag = 2
                self._sol = self._exploitation(iter)
                count_select = self._unselected
                self._unselected[0] += 1
                self._unselected[1] = 1
                self._f3_explore += self._pf[2]
                self._f3_exploit = self._pf[2]

            self._sol.sort(key=lambda x: x['Cost'])
            self._best = self._sol[0]
            self._convergence.append(self._best['Cost'])

            # Recalculando Scores y parámetros
            if self._flag_change != select_flag:
                self._flag_change = select_flag
                self._seq_time_explore = [count_select[0]] + self._seq_time_explore[:-1]
                self._seq_time_exploit = [count_select[1]] + self._seq_time_exploit[:-1]

            f1_explor = self._pf[0] * (self._seq_cost_explore[0] / self._seq_time_explore[0])
            f1_exploit = self._pf[0] * (self._seq_cost_exploit[0] / self._seq_time_exploit[0])
            f2_explor = self._pf[1] * (sum(self._seq_cost_explore) / sum(self._seq_time_explore))
            f2_exploit = self._pf[1] * (sum(self._seq_cost_exploit) / sum(self._seq_time_exploit))

            if self._score_explore < self._score_exploit:
                self._mega_explor = max((self._mega_explor - 0.01), 0.01)
                self._mega_exploit = 0.99
                
            else:
                self._mega_explor = 0.99
                self._mega_exploit = max((self._mega_exploit - 0.01), 0.01)

            self._score_explore = (self._mega_explor * f1_explor) + (self._mega_explor * f2_explor) + ((1 - self._mega_explor) * min(self._pf_f3) * self._f3_explore)
            self._score_exploit = (self._mega_exploit * f1_exploit) + (self._mega_exploit * f2_exploit) + ((1 - self._mega_exploit) * min(self._pf_f3) * self._f3_exploit)

            return self._convert()