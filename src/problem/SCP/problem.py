"""
Set Covering Problem — Unified Domain
======================================
Clase unificada ``SetCoveringProblem`` que soporta tanto SCP (weighted)
como USCP (unicost) mediante el flag ``unicost``.

Uso directo::

    from problem.SCP.problem import SetCoveringProblem
    p = SetCoveringProblem('scp41')              # SCP weighted
    p = SetCoveringProblem('uscp41', unicost=True)  # USCP

Aliases de compatibilidad::

    from problem.SCP.problem import SCP           # equivale a unicost=False
    from problem.USCP.problem import USCP          # equivale a unicost=True
"""

import os
import random

import numpy as np
from scipy.sparse import csr_matrix


# ── Diccionarios de óptimos conocidos centralizados ───────────────────────────
from problem.optima import OPTIMA_SCP, OPTIMA_USCP

# Mantener aliases privados para compatibilidad interna
_OPTIMA_SCP = OPTIMA_SCP
_OPTIMA_USCP = OPTIMA_USCP


# ── Funciones auxiliares de multiplicación por bloques ────────────────────────


def matrix_dot_1(A, B, block_size):
    """Multiplicación matriz-vector por bloques: C = A @ B."""
    C = np.zeros(A.shape[0])
    for i in range(0, A.shape[0], block_size):
        A_block = A[i : i + block_size, :]
        C[i : i + block_size] = np.dot(A_block, B)
    return C


def matrix_dot_2(A, B, block_size):
    """Producto punto por bloques: result = A · B."""
    result = 0.0
    for i in range(0, A.shape[0], block_size):
        A_block = A[i : i + block_size]
        B_block = B[i : i + block_size]
        result += np.dot(A_block, B_block)
    return result


# ── Clase unificada ───────────────────────────────────────────────────────────


class SetCoveringProblem:
    """
    Dominio unificado para Set Covering Problem (SCP) y Unicost SCP (USCP).

    La única diferencia funcional entre SCP y USCP es:
    - SCP lee los costos del archivo de instancia.
    - USCP ignora los costos del archivo y asigna 1 a cada columna.

    Args:
        instance: Nombre de la instancia (e.g. ``'scp41'``, ``'uscp41'``).
        unicost:  ``True`` para USCP (costos unitarios), ``False`` para SCP.
    """

    def __init__(self, instance, unicost=False):
        self.__rows = 0
        self.__columns = 0
        self.__coverage = []
        self.__cost = []
        self.__optimum = 0
        self.__unicost = unicost
        self.__block_size = self._resolve_block_size(instance, unicost)

        self.readInstance(instance)

    # ── Block size ────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_block_size(instance, unicost):
        """
        Determina el block_size según la familia de instancia.

        La lógica original usaba offsets distintos porque 'scp' tiene 3 chars
        y 'uscp' tiene 4 chars.  Aquí normalizamos usando el nombre de familia
        (la parte después del prefijo).
        """
        # Determinar el prefijo y la parte de familia
        if unicost:
            # Prefijo 'uscp' (4 chars)
            prefix_len = 4
        else:
            # Prefijo 'scp' (3 chars)
            prefix_len = 3

        family = instance[
            prefix_len:
        ]  # e.g. '41', 'a1', 'nre1', 'nrg1', 'cyc06', 'clr10', 'test_11x20'

        # Familias numéricas: 4x, 5x, 6x
        if family and family[0] in ("4", "5", "6"):
            return 40

        # Familias por letra
        if family and family[0] in ("a", "b"):
            return 30

        if family and family[0] in ("c", "d"):
            return 20

        # Familias NR (national rail): nre, nrf, nrg, nrh
        if family.startswith("nr"):
            sub = family[2:]  # e.g. 'e1', 'f1', 'g1', 'h1'
            if sub and sub[0] in ("e", "f"):
                return 10
            if sub and sub[0] in ("g", "h"):
                return 120

        # Familias USCP especiales: cyc, clr
        if "cyc" in instance or "clr" in instance:
            return 20

        return 1

    # ── Getters / Setters ─────────────────────────────────────────────────

    def getBlockSizes(self):
        return self.__block_size

    def getRows(self):
        return self.__rows

    def setRows(self, rows):
        self.__rows = rows

    def getColumns(self):
        return self.__columns

    def setColumns(self, columns):
        self.__columns = columns

    def getCoverange(self):
        return self.__coverage

    def setCoverange(self, coverange):
        self.__coverage = coverange

    def getCost(self):
        return self.__cost

    def setCost(self, cost):
        self.__cost = cost

    def getOptimum(self):
        return self.__optimum

    def setOptimum(self, optimum):
        self.__optimum = optimum

    # ── Lectura de instancias ─────────────────────────────────────────────

    def _get_instances_dir(self):
        """Retorna la ruta absoluta al directorio de instancias correspondiente."""
        if self.__unicost:
            return os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "USCP", "Instances"
            )
        return os.path.join(os.path.dirname(__file__), "Instances")

    def readInstance(self, instance):
        instances_dir = self._get_instances_dir()
        instance_path = os.path.join(instances_dir, instance + ".txt")

        self.setOptimum(self._lookup_optimum(instance))

        file = open(instance_path, "r")

        # Lectura de las dimensiones del problema
        line = file.readline().split()
        self.setRows(int(line[0]))
        self.setColumns(int(line[1]))

        # Lectura de los costos
        costos = []
        line = file.readline()
        countDim = 1

        while line != "" and countDim <= self.getColumns():
            values = line.split()
            for i in range(len(values)):
                if self.__unicost:
                    costos.append(1)
                else:
                    costos.append(int(values[i]))
                countDim += 1
            line = file.readline()

        # Preparar matriz de restricciones (matriz A)
        constrains = np.zeros(
            (self.getRows(), self.getColumns()), dtype=np.int32
        ).tolist()

        # Lectura de restricciones
        row = 0

        while line != "":
            numUnos = int(line)
            countUnos = 0
            line = file.readline()

            line = line.replace("\n", "").replace("\\n", "")

            while line != "" and countUnos < numUnos:
                columns = line.split()

                for i in range(len(columns)):
                    column = int(columns[i]) - 1
                    constrains[row][column] = 1
                    countUnos += 1

                line = file.readline()

            row += 1

        file.close()

        self.setCoverange(np.array(constrains))
        self.setCost(np.array(costos))

    # ── Óptimos ───────────────────────────────────────────────────────────

    def _lookup_optimum(self, instance_name):
        """Busca el óptimo conocido en el diccionario correspondiente."""
        optima = _OPTIMA_USCP if self.__unicost else _OPTIMA_SCP
        return optima.get(instance_name, [None])[1]

    @staticmethod
    def get_known_optimum(instance_name, unicost=False):
        """
        Método estático para obtener el óptimo sin instanciar el problema.
        Útil para BD/sqlite.py y scripts.
        """
        optima = _OPTIMA_USCP if unicost else _OPTIMA_SCP
        return optima.get(instance_name, [None])[1]

    def obtenerInstancia(self, archivoInstancia):
        instancia = os.path.basename(archivoInstancia).replace(".txt", "")
        return instancia

    def obtenerOptimo(self, archivoInstancia):
        """Compatibilidad con la interfaz legacy SCP."""
        instancia = self.obtenerInstancia(archivoInstancia)
        return _OPTIMA_SCP.get(instancia, [None])[1]

    def obtenerOptimoUSCP(self, archivoInstancia):
        """Compatibilidad con la interfaz legacy USCP."""
        instancia = self.obtenerInstancia(archivoInstancia)
        return _OPTIMA_USCP.get(instancia, [None])[1]

    # ── Factibilidad ──────────────────────────────────────────────────────

    def factibilityTest(self, solution):
        check = True
        if isinstance(self.getCoverange(), csr_matrix):
            validation = self.getCoverange() @ solution
        else:
            validation = matrix_dot_1(self.getCoverange(), solution, self.__block_size)

        if 0 in validation:
            check = False

        return check, validation

    # ── Reparación ────────────────────────────────────────────────────────

    def repair(self, solution, repairType):
        if repairType == "simple":
            solution = self.repairSimple(solution)

        if repairType == "complex":
            solution = self.repairComplex(solution)

        return solution

    def repairSimple(self, solution):
        reparaciones = 0
        indices = list(range(self.getRows()))
        coverange = self.getCoverange()
        costs = self.getCost()

        random.shuffle(indices)

        for i in indices:
            if np.sum(coverange[i] * solution) < 1:
                idx = np.argwhere(coverange[i] > 0)
                idxLowcost = idx[np.argmin(costs[idx])]
                solution[idxLowcost[0]] = 1
                reparaciones += 1

        return solution

    def repairComplex(self, solution):
        set_sparse = csr_matrix(self.getCoverange())
        costs = self.getCost()
        coverange = self.getCoverange()
        feasible, aux = self.factibilityTest(solution)
        reparaciones = 0

        while not feasible:
            r_no_cubiertas = (aux == 0).astype(np.int32)
            cnc = r_no_cubiertas @ set_sparse
            indices = np.nonzero(cnc)[0]
            trade_off = costs[indices] / cnc[indices]
            idx = np.argmin(trade_off)
            selected_col = indices[idx]
            solution[selected_col] = 1
            aux += coverange[:, selected_col]
            feasible = not (0 in aux)
            reparaciones += 1

        return solution

    # ── Fitness ───────────────────────────────────────────────────────────

    def fitness(self, solution):
        return matrix_dot_2(solution, self.getCost(), self.__block_size)


# ── Backward-compatibility aliases ────────────────────────────────────────────


# Para que `from problem.SCP.problem import SCP` siga funcionando
def SCP(instance):
    """Alias de compatibilidad: equivale a ``SetCoveringProblem(instance, unicost=False)``."""
    return SetCoveringProblem(instance, unicost=False)


def obtenerOptimo(archivoInstancia):
    """Función libre de compatibilidad para BD/sqlite.py."""
    instancia = os.path.basename(archivoInstancia).replace(".txt", "")
    return _OPTIMA_SCP.get(instancia, [None])[1]


# Alias para que el diccionario sea accesible si alguien lo referenciaba
orden = _OPTIMA_SCP
