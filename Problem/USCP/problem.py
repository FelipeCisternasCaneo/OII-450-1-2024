import random
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

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

class USCP:
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
            elif instance[5] == 'g' or instance[5] == 'h':
                self.__block_size = 120
            elif 'cyc' in instance or 'clr' in instance:
                self.__block_size = 20
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
        
        dirSCP = './Problem/USCP/Instances/'
        
        instance = dirSCP+instance+".txt" 
        
        self.setOptimum(self.obtenerOptimoUSCP(instance))
        
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
                costos.append(1)
                countDim +=1
            line = file.readline()
        
        # print("Costos para cada columna: "+str(costos))
        # print("Cantidad de costos para cada columna: "+str(costos.__len__()))

        # Preparar matriz de restricciones (matriz A)
        constrains = np.zeros((self.getRows(),self.getColumns()), dtype=np.int32).tolist()

        # Lecutra de Restricciones
        row = 0


        while line != "":
            numUnos = int(line)
            # print("Cantidad de columnas que cubre la fila "+str(row)+": "+str(numUnos))
            countUnos = 0
            line = file.readline()

            line = line.replace('\n',"").replace('\\n',"")

            while line != "" and countUnos < numUnos:
                columns = line.split()
                for i in range(len(columns)):
                    column = int(columns[i]) - 1
                    constrains[row][column] = 1
                    countUnos +=1
                line = file.readline()
            # print("Coberturas para la fila "+str(row)+": "+str(constrains[row]))
            # print("Suma de validacion: "+str(sum(constrains[row])))

            row += 1
        
        file.close()

        self.setCoverange(np.array(constrains))
        self.setCost(np.array(costos))
        # print("Chequeo de cobertura: "+str(constrains[0][90]))  
     
    def obtenerOptimoUSCP(self, archivoInstancia):
        orden = {
             'scp41':[0,38]
            ,'scp42':[1,37]
            ,'scp43':[2,38]
            ,'scp44':[3,38]
            ,'scp45':[4,38]
            ,'scp46':[5,37]
            ,'scp47':[6,38]
            ,'scp48':[7,37]
            ,'scp49':[8,38]
            ,'scp410':[9,38]
            ,'scp51':[10,34]
            ,'scp52':[11,34]
            ,'scp53':[12,34]
            ,'scp54':[13,34]
            ,'scp55':[14,34]
            ,'scp56':[15,34]
            ,'scp57':[16,34]
            ,'scp58':[17,34]
            ,'scp59':[18,35]
            ,'scp510':[19,34]
            ,'scp61':[20,21]
            ,'scp62':[21,20]
            ,'scp63':[22,21]
            ,'scp64':[23,20]
            ,'scp65':[24,21]
            ,'scpa1':[25,38]
            ,'scpa2':[26,38]
            ,'scpa3':[27,38]
            ,'scpa4':[28,37]
            ,'scpa5':[29,38]
            ,'scpb1':[30,22]
            ,'scpb2':[31,22]
            ,'scpb3':[32,22]
            ,'scpb4':[33,22]
            ,'scpb5':[34,22]
            ,'scpc1':[35,43]
            ,'scpc2':[36,43]
            ,'scpc3':[37,43]
            ,'scpc4':[38,43]
            ,'scpc5':[39,43]
            ,'scpd1':[40,24]
            ,'scpd2':[41,24]
            ,'scpd3':[42,24]
            ,'scpd4':[43,24]
            ,'scpd5':[44,24]
            ,'scpnre1':[45,16]
            ,'scpnre2':[46,16]
            ,'scpnre3':[47,16]
            ,'scpnre4':[48,16]
            ,'scpnre5':[49,16]
            ,'scpnrf1':[50,10]
            ,'scpnrf2':[51,10]
            ,'scpnrf3':[52,10]
            ,'scpnrf4':[53,10]
            ,'scpnrf5':[54,10]
            ,'scpnrg1':[55,60]
            ,'scpnrg2':[56,60]
            ,'scpnrg3':[57,60]
            ,'scpnrg4':[58,60]
            ,'scpnrg5':[59,60]
            ,'scpnrh1':[60,33]
            ,'scpnrh2':[61,33]
            ,'scpnrh3':[62,33]
            ,'scpnrh4':[63,33]
            ,'scpnrh5':[64,33]
            ,'scpcyc06':[65,60]
            ,'scpcyc07':[66,144]
            ,'scpcyc08':[67,342]
            ,'scpcyc09':[68,772]
            ,'scpcyc10':[69,1794]
            ,'scpcyc11':[70,3968]
            ,'scpclr10':[71,25]
            ,'scpclr11':[72,23]
            ,'scpclr12':[73,23]
            ,'scpclr13':[74,23]
        }

        for nomInstancia in orden:
            if nomInstancia in archivoInstancia:
                #print(f"instancia {nomInstancia}")
                return orden[nomInstancia][1]

        return None

    def factibilityTest(self, solution):
        check = True        
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
                reparaciones +=1
        # print(f'total de reparaciones realizadas: {reparaciones}')
        
        return solution
    
    def repairComplex(self, solution):
        # Convertir la matriz de cobertura a formato disperso CSR si no está ya en ese formato
        set_sparse = csr_matrix(self.getCoverange())  # Cobertura en formato disperso
        costs = self.getCost()
        # Realizar la prueba de factibilidad inicial
        feasible, aux = self.factibilityTest(solution)
        reparaciones = 0
        
        while not feasible: # repetimos hasta que la solución sea factible
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
            
            
def obtenerOptimoUSCP(archivoInstancia):
    orden = {
             'scp41':[0,38]
            ,'scp42':[1,37]
            ,'scp43':[2,38]
            ,'scp44':[3,38]
            ,'scp45':[4,38]
            ,'scp46':[5,37]
            ,'scp47':[6,38]
            ,'scp48':[7,37]
            ,'scp49':[8,38]
            ,'scp410':[9,38]
            ,'scp51':[10,34]
            ,'scp52':[11,34]
            ,'scp53':[12,34]
            ,'scp54':[13,34]
            ,'scp55':[14,34]
            ,'scp56':[15,34]
            ,'scp57':[16,34]
            ,'scp58':[17,34]
            ,'scp59':[18,35]
            ,'scp510':[19,34]
            ,'scp61':[20,21]
            ,'scp62':[21,20]
            ,'scp63':[22,21]
            ,'scp64':[23,20]
            ,'scp65':[24,21]
            ,'scpa1':[25,38]
            ,'scpa2':[26,38]
            ,'scpa3':[27,38]
            ,'scpa4':[28,37]
            ,'scpa5':[29,38]
            ,'scpb1':[30,69]
            ,'scpb2':[31,22]
            ,'scpb3':[32,22]
            ,'scpb4':[33,22]
            ,'scpb5':[34,22]
            ,'scpc1':[35,43]
            ,'scpc2':[36,43]
            ,'scpc3':[37,43]
            ,'scpc4':[38,43]
            ,'scpc5':[39,43]
            ,'scpd1':[40,24]
            ,'scpd2':[41,24]
            ,'scpd3':[42,24]
            ,'scpd4':[43,24]
            ,'scpd5':[44,24]
            ,'scpnre1':[45,16]
            ,'scpnre2':[46,16]
            ,'scpnre3':[47,16]
            ,'scpnre4':[48,16]
            ,'scpnre5':[49,16]
            ,'scpnrf1':[50,10]
            ,'scpnrf2':[51,10]
            ,'scpnrf3':[52,10]
            ,'scpnrf4':[53,10]
            ,'scpnrf5':[54,10]
            ,'scpnrg1':[55,60]
            ,'scpnrg2':[56,60]
            ,'scpnrg3':[57,60]
            ,'scpnrg4':[58,60]
            ,'scpnrg5':[59,60]
            ,'scpnrh1':[60,33]
            ,'scpnrh2':[61,33]
            ,'scpnrh3':[62,33]
            ,'scpnrh4':[63,33]
            ,'scpnrh5':[64,33]
            ,'scpcyc06':[65,60]
            ,'scpcyc07':[66,144]
            ,'scpcyc08':[67,342]
            ,'scpcyc09':[68,772]
            ,'scpcyc10':[69,1794]
            ,'scpcyc11':[70,3968]
            ,'scpclr10':[71,25]
            ,'scpclr11':[72,23]
            ,'scpclr12':[73,23]
            ,'scpclr13':[74,23]
        }

    for nomInstancia in orden:
        if nomInstancia in archivoInstancia:
            #print(f"instancia {nomInstancia}")
            return orden[nomInstancia][1]

    return None