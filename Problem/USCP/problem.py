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
        
        print("largo de la instancia: "+str(len(instance)))
        
        if len(instance) == 6:
            if instance[4] == '4' or instance[4] == '5' or instance[4] == '6':
                self.__block_size = 40
            
            elif instance[4] == 'a' or instance[4] == 'b':
                self.__block_size = 30
            
            elif instance[4] == 'c' or instance[4] == 'd':
                self.__block_size = 20
            
        else:
            if instance[6] == 'e' or instance[6] == 'f':
                self.__block_size = 10
            
            elif instance[6] == 'g' or instance[6] == 'h':
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
        
        instance = dirSCP + instance + ".txt" 
        
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
        constrains = np.zeros((self.getRows(),self.getColumns()), dtype = np.int32).tolist()

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
            'uscp41':[0, 38]
            ,'uscp42':[1, 37]
            ,'uscp43':[2, 38]
            ,'uscp44':[3, 38]
            ,'uscp45':[4, 38]
            ,'uscp46':[5, 37]
            ,'uscp47':[6, 38]
            ,'uscp48':[7, 37]
            ,'uscp49':[8, 38]
            ,'uscp410':[9, 38]
            ,'uscp51':[10, 34]
            ,'uscp52':[11, 34]
            ,'uscp53':[12, 34]
            ,'uscp54':[13, 34]
            ,'uscp55':[14, 34]
            ,'uscp56':[15, 34]
            ,'uscp57':[16, 34]
            ,'uscp58':[17, 34]
            ,'uscp59':[18, 35]
            ,'uscp510':[19, 34]
            ,'uscp61':[20, 21]
            ,'uscp62':[21, 20]
            ,'uscp63':[22, 21]
            ,'uscp64':[23, 20]
            ,'uscp65':[24, 21]
            ,'uscpa1':[25, 38]
            ,'uscpa2':[26, 38]
            ,'uscpa3':[27, 38]
            ,'uscpa4':[28, 37]
            ,'uscpa5':[29, 38]
            ,'uscpb1':[30, 22]
            ,'uscpb2':[31, 22]
            ,'uscpb3':[32, 22]
            ,'uscpb4':[33, 22]
            ,'uscpb5':[34, 22]
            ,'uscpc1':[35, 43]
            ,'uscpc2':[36, 43]
            ,'uscpc3':[37, 43]
            ,'uscpc4':[38, 43]
            ,'uscpc5':[39, 43]
            ,'uscpd1':[40, 24]
            ,'uscpd2':[41, 24]
            ,'uscpd3':[42, 24]
            ,'uscpd4':[43, 24]
            ,'uscpd5':[44, 24]
            ,'uscpnre1':[45, 16]
            ,'uscpnre2':[46, 16]
            ,'uscpnre3':[47, 16]
            ,'uscpnre4':[48, 16]
            ,'uscpnre5':[49, 16]
            ,'uscpnrf1':[50, 10]
            ,'uscpnrf2':[51, 10]
            ,'uscpnrf3':[52, 10]
            ,'uscpnrf4':[53, 10]
            ,'uscpnrf5':[54, 10]
            ,'uscpnrg1':[55, 60]
            ,'uscpnrg2':[56, 60]
            ,'uscpnrg3':[57, 60]
            ,'uscpnrg4':[58, 60]
            ,'uscpnrg5':[59, 60]
            ,'uscpnrh1':[60, 33]
            ,'uscpnrh2':[61, 33]
            ,'uscpnrh3':[62, 33]
            ,'uscpnrh4':[63, 33]
            ,'uscpnrh5':[64, 33]
            ,'uscpcyc06':[65, 60]
            ,'uscpcyc07':[66, 144]
            ,'uscpcyc08':[67, 342]
            ,'uscpcyc09':[68, 772]
            ,'uscpcyc10':[69, 1794]
            ,'uscpcyc11':[70, 3968]
            ,'uscpclr10':[71, 25]
            ,'uscpclr11':[72, 23]
            ,'uscpclr12':[73, 23]
            ,'uscpclr13':[74, 23]
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
                reparaciones += 1
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
             'uscp41':[0, 38]
            ,'uscp42':[1, 37]
            ,'uscp43':[2, 38]
            ,'uscp44':[3, 38]
            ,'uscp45':[4, 38]
            ,'uscp46':[5, 37]
            ,'uscp47':[6, 38]
            ,'uscp48':[7, 37]
            ,'uscp49':[8, 38]
            ,'uscp410':[9, 38]
            ,'uscp51':[10, 34]
            ,'uscp52':[11, 34]
            ,'uscp53':[12, 34]
            ,'uscp54':[13, 34]
            ,'uscp55':[14, 34]
            ,'uscp56':[15, 34]
            ,'uscp57':[16, 34]
            ,'uscp58':[17, 34]
            ,'uscp59':[18, 35]
            ,'uscp510':[19, 34]
            ,'uscp61':[20, 21]
            ,'uscp62':[21, 20]
            ,'uscp63':[22, 21]
            ,'uscp64':[23, 20]
            ,'uscp65':[24, 21]
            ,'uscpa1':[25, 38]
            ,'uscpa2':[26, 38]
            ,'uscpa3':[27, 38]
            ,'uscpa4':[28, 37]
            ,'uscpa5':[29, 38]
            ,'uscpb1':[30, 69]
            ,'uscpb2':[31, 22]
            ,'uscpb3':[32, 22]
            ,'uscpb4':[33, 22]
            ,'uscpb5':[34, 22]
            ,'uscpc1':[35, 43]
            ,'uscpc2':[36, 43]
            ,'uscpc3':[37, 43]
            ,'uscpc4':[38, 43]
            ,'uscpc5':[39, 43]
            ,'uscpd1':[40, 24]
            ,'uscpd2':[41, 24]
            ,'uscpd3':[42, 24]
            ,'uscpd4':[43, 24]
            ,'uscpd5':[44, 24]
            ,'uscpnre1':[45, 16]
            ,'uscpnre2':[46, 16]
            ,'uscpnre3':[47, 16]
            ,'uscpnre4':[48, 16]
            ,'uscpnre5':[49, 16]
            ,'uscpnrf1':[50, 10]
            ,'uscpnrf2':[51, 10]
            ,'uscpnrf3':[52, 10]
            ,'uscpnrf4':[53, 10]
            ,'uscpnrf5':[54, 10]
            ,'uscpnrg1':[55, 60]
            ,'uscpnrg2':[56, 60]
            ,'uscpnrg3':[57, 60]
            ,'uscpnrg4':[58, 60]
            ,'uscpnrg5':[59, 60]
            ,'uscpnrh1':[60, 33]
            ,'uscpnrh2':[61, 33]
            ,'uscpnrh3':[62, 33]
            ,'uscpnrh4':[63, 33]
            ,'uscpnrh5':[64, 33]
            ,'uscpcyc06':[65, 60]
            ,'uscpcyc07':[66, 144]
            ,'uscpcyc08':[67, 342]
            ,'uscpcyc09':[68, 772]
            ,'uscpcyc10':[69, 1794]
            ,'uscpcyc11':[70, 3968]
            ,'uscpclr10':[71, 25]
            ,'uscpclr11':[72, 23]
            ,'uscpclr12':[73, 23]
            ,'uscpclr13':[74, 23]
        }

    for nomInstancia in orden:
        if nomInstancia in archivoInstancia:
            #print(f"instancia {nomInstancia}")
            return orden[nomInstancia][1]

    return None