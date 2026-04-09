"""
Set Covering Problem — Unified Domain
======================================
Clase unificada ``SetCoveringProblem`` que soporta tanto SCP (weighted)
como USCP (unicost) mediante el flag ``unicost``.

Uso directo::

    from Problem.SCP.problem import SetCoveringProblem
    p = SetCoveringProblem('scp41')              # SCP weighted
    p = SetCoveringProblem('uscp41', unicost=True)  # USCP

Aliases de compatibilidad::

    from Problem.SCP.problem import SCP           # equivale a unicost=False
    from Problem.USCP.problem import USCP          # equivale a unicost=True
"""

import os
import random

import numpy as np
from scipy.sparse import csr_matrix


# ── Diccionarios de óptimos conocidos ─────────────────────────────────────────

_OPTIMA_SCP = {
    "scp41": [0, 429],
    "scp42": [1, 512],
    "scp43": [2, 516],
    "scp44": [3, 494],
    "scp45": [4, 512],
    "scp46": [5, 560],
    "scp47": [6, 430],
    "scp48": [7, 492],
    "scp49": [8, 641],
    "scp410": [9, 514],
    "scp51": [10, 253],
    "scp52": [11, 302],
    "scp53": [12, 226],
    "scp54": [13, 242],
    "scp55": [14, 211],
    "scp56": [15, 213],
    "scp57": [16, 293],
    "scp58": [17, 288],
    "scp59": [18, 279],
    "scp510": [19, 265],
    "scp61": [20, 138],
    "scp62": [21, 146],
    "scp63": [22, 145],
    "scp64": [23, 131],
    "scp65": [24, 161],
    "scpa1": [25, 253],
    "scpa2": [26, 252],
    "scpa3": [27, 232],
    "scpa4": [28, 234],
    "scpa5": [29, 236],
    "scpb1": [30, 69],
    "scpb2": [31, 76],
    "scpb3": [32, 80],
    "scpb4": [33, 79],
    "scpb5": [34, 72],
    "scpc1": [35, 227],
    "scpc2": [36, 219],
    "scpc3": [37, 243],
    "scpc4": [38, 219],
    "scpc5": [39, 215],
    "scpd1": [40, 60],
    "scpd2": [41, 66],
    "scpd3": [42, 72],
    "scpd4": [43, 62],
    "scpd5": [44, 61],
    "scpnre1": [45, 29],
    "scpnre2": [46, 30],
    "scpnre3": [47, 27],
    "scpnre4": [48, 28],
    "scpnre5": [49, 28],
    "scpnrf1": [50, 14],
    "scpnrf2": [51, 15],
    "scpnrf3": [52, 14],
    "scpnrf4": [53, 14],
    "scpnrf5": [54, 13],
    "scpnrg1": [55, 176],
    "scpnrg2": [56, 154],
    "scpnrg3": [57, 166],
    "scpnrg4": [58, 168],
    "scpnrg5": [59, 168],
    "scpnrh1": [60, 63],
    "scpnrh2": [61, 63],
    "scpnrh3": [62, 59],
    "scpnrh4": [63, 58],
    "scpnrh5": [64, 55],
    "scptest_11x20": [65, 13],
}

_OPTIMA_USCP = {
    "uscp41": [0, 38],
    "uscp42": [1, 37],
    "uscp43": [2, 38],
    "uscp44": [3, 38],
    "uscp45": [4, 38],
    "uscp46": [5, 37],
    "uscp47": [6, 38],
    "uscp48": [7, 37],
    "uscp49": [8, 38],
    "uscp410": [9, 38],
    "uscp51": [10, 34],
    "uscp52": [11, 34],
    "uscp53": [12, 34],
    "uscp54": [13, 34],
    "uscp55": [14, 34],
    "uscp56": [15, 34],
    "uscp57": [16, 34],
    "uscp58": [17, 34],
    "uscp59": [18, 35],
    "uscp510": [19, 34],
    "uscp61": [20, 21],
    "uscp62": [21, 20],
    "uscp63": [22, 21],
    "uscp64": [23, 20],
    "uscp65": [24, 21],
    "uscpa1": [25, 38],
    "uscpa2": [26, 38],
    "uscpa3": [27, 38],
    "uscpa4": [28, 37],
    "uscpa5": [29, 38],
    "uscpb1": [30, 22],
    "uscpb2": [31, 22],
    "uscpb3": [32, 22],
    "uscpb4": [33, 22],
    "uscpb5": [34, 22],
    "uscpc1": [35, 43],
    "uscpc2": [36, 43],
    "uscpc3": [37, 43],
    "uscpc4": [38, 43],
    "uscpc5": [39, 43],
    "uscpd1": [40, 24],
    "uscpd2": [41, 24],
    "uscpd3": [42, 24],
    "uscpd4": [43, 24],
    "uscpd5": [44, 24],
    "uscpnre1": [45, 16],
    "uscpnre2": [46, 16],
    "uscpnre3": [47, 16],
    "uscpnre4": [48, 16],
    "uscpnre5": [49, 16],
    "uscpnrf1": [50, 10],
    "uscpnrf2": [51, 10],
    "uscpnrf3": [52, 10],
    "uscpnrf4": [53, 10],
    "uscpnrf5": [54, 10],
    "uscpnrg1": [55, 60],
    "uscpnrg2": [56, 60],
    "uscpnrg3": [57, 60],
    "uscpnrg4": [58, 60],
    "uscpnrg5": [59, 60],
    "uscpnrh1": [60, 33],
    "uscpnrh2": [61, 33],
    "uscpnrh3": [62, 33],
    "uscpnrh4": [63, 33],
    "uscpnrh5": [64, 33],
    "uscpcyc06": [65, 60],
    "uscpcyc07": [66, 144],
    "uscpcyc08": [67, 342],
    "uscpcyc09": [68, 772],
    "uscpcyc10": [69, 1794],
    "uscpcyc11": [70, 3968],
    "uscpclr10": [71, 25],
    "uscpclr11": [72, 23],
    "uscpclr12": [73, 23],
    "uscpclr13": [74, 23],
}


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


# Para que `from Problem.SCP.problem import SCP` siga funcionando
def SCP(instance):
    """Alias de compatibilidad: equivale a ``SetCoveringProblem(instance, unicost=False)``."""
    return SetCoveringProblem(instance, unicost=False)


def obtenerOptimo(archivoInstancia):
    """Función libre de compatibilidad para BD/sqlite.py."""
    instancia = os.path.basename(archivoInstancia).replace(".txt", "")
    return _OPTIMA_SCP.get(instancia, [None])[1]


# Alias para que el diccionario sea accesible si alguien lo referenciaba
orden = _OPTIMA_SCP
