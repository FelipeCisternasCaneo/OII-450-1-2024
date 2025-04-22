import random
import numpy as np
from scipy.sparse import csr_matrix

def matrix_dot_1(A, B, block_size):
    # Inicializar el resultado con ceros (es un vector de tamaño n)
    C = np.zeros(A.shape[0])
    # Multiplicación por bloques
    
    for i in range(0, A.shape[0], block_size):
        # Seleccionar un bloque de filas de A
        A_block = A[i: i + block_size, :]
        # Multiplicar el bloque de filas de A por el vector B
        C[i: i + block_size] = np.dot(A_block, B)
        
    return C

def matrix_dot_2(A, B, block_size):
    # Inicializamos el resultado como un escalar (producto punto)
    result = 0.0
    # Multiplicación por bloques
    
    for i in range(0, A.shape[0], block_size):
        # Seleccionar un bloque de A y B
        A_block = A[i: i + block_size]
        B_block = B[i: i + block_size]
        # Calcular el producto punto del bloque y sumarlo al resultado total
        result += np.dot(A_block, B_block)
    
    return result

orden = {
    'scp41': [0, 429], 'scp42': [1, 512], 'scp43': [2, 516], 'scp44': [3, 494], 'scp45': [4, 512],
    'scp46': [5, 560], 'scp47': [6, 430], 'scp48': [7, 492], 'scp49': [8, 641], 'scp410': [9, 514],
    'scp51': [10, 253], 'scp52': [11, 302], 'scp53': [12, 226], 'scp54': [13, 242], 'scp55': [14, 211],
    'scp56': [15, 213], 'scp57': [16, 293], 'scp58': [17, 288], 'scp59': [18, 279], 'scp510': [19, 265],
    'scp61': [20, 138], 'scp62': [21, 146], 'scp63': [22, 145], 'scp64': [23, 131], 'scp65': [24, 161],
    'scpa1': [25, 253], 'scpa2': [26, 252], 'scpa3': [27, 232], 'scpa4': [28, 234], 'scpa5': [29, 236],
    'scpb1': [30, 69], 'scpb2': [31, 76], 'scpb3': [32, 80], 'scpb4': [33, 79], 'scpb5': [34, 72],
    'scpc1': [35, 227], 'scpc2': [36, 219], 'scpc3': [37, 243], 'scpc4': [38, 219], 'scpc5': [39, 215],
    'scpd1': [40, 60], 'scpd2': [41, 66], 'scpd3': [42, 72], 'scpd4': [43, 62], 'scpd5': [44, 61],
    'scpnre1': [45, 29], 'scpnre2': [46, 30], 'scpnre3': [47, 27], 'scpnre4': [48, 28], 'scpnre5': [49, 28],
    'scpnrf1': [50, 14], 'scpnrf2': [51, 15], 'scpnrf3': [52, 14], 'scpnrf4': [53, 14], 'scpnrf5': [54, 13],
    'scpnrg1': [55, 176], 'scpnrg2': [56, 154], 'scpnrg3': [57, 166], 'scpnrg4': [58, 168], 'scpnrg5': [59, 168],
    'scpnrh1': [60, 63], 'scpnrh2': [61, 63], 'scpnrh3': [62, 59], 'scpnrh4': [63, 58], 'scpnrh5': [64, 55],
    'scptest_11x20': [65, 13]
}

class SCP:
    def __init__(self, instance):
        self.__rows = 0
        self.__columns = 0
        self.__coverage = []
        self.__cost = []
        self.__optimum = 0
        self.__block_size = 0
        
        if len(instance) == 5:
            if instance[3] == '4' or instance[3] == '5' or instance[3] == '6':
                self.__block_size = 40
                
            elif instance[3] == 'a' or instance[3] == 'b':
                self.__block_size = 30
                
            elif instance[3] == 'c' or instance[3] == 'd':
                self.__block_size = 20
        
        else:
            if instance[5] == 'e' or instance[5] == 'f':
                self.__block_size = 10
            
            elif instance[3] == '4' or instance[3] == '5':
                self.__block_size = 40
            
            elif instance[5] == 'g' or instance[5] == 'h':
                self.__block_size = 120
            
            else:
                self.__block_size = 1
            
        self.readInstance(instance)

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

    def readInstance(self, instance):
        dirSCP = './Problem/SCP/Instances/'
        
        instance = dirSCP + instance + ".txt" 
        
        self.setOptimum(self.obtenerOptimo(instance))
        
        file = open(instance, 'r')

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
                costos.append(int(values[i]))
                countDim +=1
            
            line = file.readline()
        
        # print("Costos para cada columna: "+str(costos))
        # print("Cantidad de costos para cada columna: "+str(costos.__len__()))

        # Preparar matriz de restricciones (matriz A)
        constrains = np.zeros((self.getRows(), self.getColumns()), dtype = np.int32).tolist()

        # Lectura de restricciones
        row = 0

        while line != "":
            numUnos = int(line)
            # print("Cantidad de columnas que cubre la fila "+str(row)+": "+str(numUnos))
            countUnos = 0
            line = file.readline()

            line = line.replace('\n', "").replace('\\n', "")

            while line != "" and countUnos < numUnos:
                columns = line.split()
                
                for i in range(len(columns)):
                    column = int(columns[i]) - 1
                    constrains[row][column] = 1
                    countUnos += 1
                    
                line = file.readline()
                
            # print("Coberturas para la fila "+str(row)+": "+str(constrains[row]))
            # print("Suma de validacion: "+str(sum(constrains[row])))

            row += 1
        
        file.close()

        self.setCoverange(np.array(constrains))
        self.setCost(np.array(costos))
        # print("Chequeo de cobertura: "+str(constrains[0][90]))  
     
    def obtenerInstancia(self, archivoInstancia):
    # Extraemos el nombre de la instancia, eliminando la extensión .txt
        instancia = archivoInstancia.split('/')[-1].replace('.txt', '')
        return instancia

    def obtenerOptimo(self, archivoInstancia):
        instancia = self.obtenerInstancia(archivoInstancia)
        clave_instancia = f"{instancia}"
        
        return orden.get(clave_instancia, [None])[1]  # Devuelve el óptimo si existe, sino None

    def factibilityTest(self, solution):
        check = True        
        # Si la matriz de cobertura es dispersa, la multiplicación es más eficiente
        if isinstance(self.getCoverange(), csr_matrix):
            validation = self.getCoverange() @ solution  # Multiplicación dispersa
        else:
            validation = matrix_dot_1(self.getCoverange(), solution, self.__block_size)

        if 0 in validation:
            check = False
        
        return check, validation
    
    def repair(self, solution, repairType):
        if repairType == 'simple':
            solution = self.repairSimple(solution)
            
        if repairType == 'complex':
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
                # print(f'columnas que satisfacen la restriccion: {idx}')
                idxLowcost = idx[np.argmin(costs[idx])]
                # print(f'indice del menor costo: {idxLowcost}')
                solution[idxLowcost[0]] = 1
                reparaciones += 1
        # print(f'total de reparaciones realizadas: {reparaciones}')
        
        return solution
    
    def repairComplex(self, solution):
        set_sparse = csr_matrix(self.getCoverange())  # Cobertura en formato disperso
        costs = self.getCost()
        # Realizar la prueba de factibilidad inicial
        feasible, aux = self.factibilityTest(solution)
        reparaciones = 0
        
        while not feasible:  # repetimos hasta que la solución sea factible
            # Crear un vector disperso para restricciones no cubiertas
            r_no_cubiertas = (aux == 0).astype(np.int32)
            # Calcular la cantidad de restricciones no cubiertas que cubre cada columna usando multiplicación dispersa
            cnc = r_no_cubiertas @ set_sparse  # Operador @ realiza np.dot en formato disperso
            # Obtener los índices de columnas que cubren restricciones no cubiertas
            indices = np.nonzero(cnc)[0]
            # Calcular el trade-off entre costos y cobertura
            trade_off = costs[indices] / cnc[indices]
            # Seleccionar la columna con el menor trade-off
            idx = np.argmin(trade_off)
            # Actualizar la solución asignando 1 a la columna seleccionada
            solution[indices[idx]] = 1
            # Verificar factibilidad nuevamente
            feasible, aux = self.factibilityTest(solution)
            reparaciones += 1
        
        return solution

    def fitness(self, solution):
        return matrix_dot_2(solution, self.getCost(), self.__block_size)
    
def obtenerOptimo(archivoInstancia):
    instancia = archivoInstancia.split('/')[-1].replace('.txt', '')
    
    clave_instancia = f"{instancia}"
    
    return orden.get(clave_instancia, [None])[1]