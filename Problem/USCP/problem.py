import random
import numpy as np

class USCP:
    def __init__(self, instance):
        self.__rows = 0
        self.__columns = 0
        self.__coverage = []
        self.__cost = []
        self.__optimum = 0
        self.readInstance(instance)

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
                countDim += 1
                
            line = file.readline()
        
        # print("Costos para cada columna: "+str(costos))
        # print("Cantidad de costos para cada columna: "+str(costos.__len__()))

        # Preparar matriz de restricciones (matriz A)
        constrains = np.zeros((self.getRows(), self.getColumns()), dtype = np.int32).tolist()

        # Lecutra de Restricciones
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
     
    def obtenerOptimoUSCP(self, archivoInstancia):
        orden = {
             'scp41':[0, 38]
            ,'scp42':[1, 37]
            ,'scp43':[2, 38]
            ,'scp44':[3, 38]
            ,'scp45':[4, 38]
            ,'scp46':[5, 37]
            ,'scp47':[6, 38]
            ,'scp48':[7, 37]
            ,'scp49':[8, 38]
            ,'scp410':[9, 38]
            ,'scp51':[10, 34]
            ,'scp52':[11, 34]
            ,'scp53':[12, 34]
            ,'scp54':[13, 34]
            ,'scp55':[14, 34]
            ,'scp56':[15, 34]
            ,'scp57':[16, 34]
            ,'scp58':[17, 34]
            ,'scp59':[18, 35]
            ,'scp510':[19, 34]
            ,'scp61':[20, 21]
            ,'scp62':[21, 20]
            ,'scp63':[22, 21]
            ,'scp64':[23, 20]
            ,'scp65':[24, 21]
            ,'scpa1':[25, 38]
            ,'scpa2':[26, 38]
            ,'scpa3':[27, 38]
            ,'scpa4':[28, 37]
            ,'scpa5':[29, 38]
            ,'scpb1':[30, 22]
            ,'scpb2':[31, 22]
            ,'scpb3':[32, 22]
            ,'scpb4':[33, 22]
            ,'scpb5':[34, 22]
            ,'scpc1':[35, 43]
            ,'scpc2':[36, 43]
            ,'scpc3':[37, 43]
            ,'scpc4':[38, 43]
            ,'scpc5':[39, 43]
            ,'scpd1':[40, 24]
            ,'scpd2':[41, 24]
            ,'scpd3':[42, 24]
            ,'scpd4':[43, 24]
            ,'scpd5':[44, 24]
            ,'scpnre1':[45, 16]
            ,'scpnre2':[46, 16]
            ,'scpnre3':[47, 16]
            ,'scpnre4':[48, 16]
            ,'scpnre5':[49, 16]
            ,'scpnrf1':[50, 10]
            ,'scpnrf2':[51, 10]
            ,'scpnrf3':[52, 10]
            ,'scpnrf4':[53, 10]
            ,'scpnrf5':[54, 10]
            ,'scpnrg1':[55, 60]
            ,'scpnrg2':[56, 60]
            ,'scpnrg3':[57, 60]
            ,'scpnrg4':[58, 60]
            ,'scpnrg5':[59, 60]
            ,'scpnrh1':[60, 33]
            ,'scpnrh2':[61, 33]
            ,'scpnrh3':[62, 33]
            ,'scpnrh4':[63, 33]
            ,'scpnrh5':[64, 33]
            ,'scpcyc06':[65, 60]
            ,'scpcyc07':[66, 144]
            ,'scpcyc08':[67, 342]
            ,'scpcyc09':[68, 772]
            ,'scpcyc10':[69, 1794]
            ,'scpcyc11':[70, 3968]
            ,'scpclr10':[71, 25]
            ,'scpclr11':[72, 23]
            ,'scpclr12':[73, 23]
            ,'scpclr13':[74, 23]
        }

        for nomInstancia in orden:
            if nomInstancia in archivoInstancia:
                #print(f"instancia {nomInstancia}")
                
                return orden[nomInstancia][1]

        return None

    def factibilityTest(self, solution):
        check = True
        validation = np.dot(self.getCoverange(), solution)

        if 0 in validation:
            check = False
            # print(f'solucion infactible: {solution}')
            # print(f'motivo: {validation}')

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
        set = self.getCoverange()
        
        feasible, aux = self.factibilityTest(solution)
        costs = self.getCost()
        reparaciones = 0
        
        while not feasible:
            r_no_cubiertas = np.zeros((self.getRows()))
            
            r_no_cubiertas[np.argwhere(aux == 0)] = 1           # Vector indica las restricciones no cubiertas
            # print(r_no_cubiertas)
            cnc = np.dot(r_no_cubiertas, set)                   # Cantidad de restricciones no cubiertas que cubre cada columna (de tamaño n)
            # print(cnc)
            trade_off = np.divide(costs,cnc)                    # Trade off entre zonas no cubiertas y costo de seleccionar cada columna
            # print(trade_off)
            idx = np.argmin(trade_off)                          # Selecciono la columna con el trade off mas bajo
            # print(idx)
            solution[idx] = 1                                   # Asigno 1 a esa columna
            feasible, aux = self.factibilityTest(solution)      # Verifico si la solucion actualizada es factible
            reparaciones += 1

        return solution

    def fitness(self, solution):
        return np.dot(solution, self.getCost())
            
def obtenerOptimoUSCP(archivoInstancia):
    orden = {
             'scp41':[0, 38]
            ,'scp42':[1, 37]
            ,'scp43':[2, 38]
            ,'scp44':[3, 38]
            ,'scp45':[4, 38]
            ,'scp46':[5, 37]
            ,'scp47':[6, 38]
            ,'scp48':[7, 37]
            ,'scp49':[8, 38]
            ,'scp410':[9, 38]
            ,'scp51':[10, 34]
            ,'scp52':[11, 34]
            ,'scp53':[12, 34]
            ,'scp54':[13, 34]
            ,'scp55':[14, 34]
            ,'scp56':[15, 34]
            ,'scp57':[16, 34]
            ,'scp58':[17, 34]
            ,'scp59':[18, 35]
            ,'scp510':[19, 34]
            ,'scp61':[20, 21]
            ,'scp62':[21, 20]
            ,'scp63':[22, 21]
            ,'scp64':[23, 20]
            ,'scp65':[24, 21]
            ,'scpa1':[25, 38]
            ,'scpa2':[26, 38]
            ,'scpa3':[27, 38]
            ,'scpa4':[28, 37]
            ,'scpa5':[29, 38]
            ,'scpb1':[30, 69]
            ,'scpb2':[31, 22]
            ,'scpb3':[32, 22]
            ,'scpb4':[33, 22]
            ,'scpb5':[34, 22]
            ,'scpc1':[35, 43]
            ,'scpc2':[36, 43]
            ,'scpc3':[37, 43]
            ,'scpc4':[38, 43]
            ,'scpc5':[39, 43]
            ,'scpd1':[40, 24]
            ,'scpd2':[41, 24]
            ,'scpd3':[42, 24]
            ,'scpd4':[43, 24]
            ,'scpd5':[44, 24]
            ,'scpnre1':[45, 16]
            ,'scpnre2':[46, 16]
            ,'scpnre3':[47, 16]
            ,'scpnre4':[48, 16]
            ,'scpnre5':[49, 16]
            ,'scpnrf1':[50, 10]
            ,'scpnrf2':[51, 10]
            ,'scpnrf3':[52, 10]
            ,'scpnrf4':[53, 10]
            ,'scpnrf5':[54, 10]
            ,'scpnrg1':[55, 60]
            ,'scpnrg2':[56, 60]
            ,'scpnrg3':[57, 60]
            ,'scpnrg4':[58, 60]
            ,'scpnrg5':[59, 60]
            ,'scpnrh1':[60, 33]
            ,'scpnrh2':[61, 33]
            ,'scpnrh3':[62, 33]
            ,'scpnrh4':[63, 33]
            ,'scpnrh5':[64, 33]
            ,'scpcyc06':[65, 60]
            ,'scpcyc07':[66, 144]
            ,'scpcyc08':[67, 342]
            ,'scpcyc09':[68, 772]
            ,'scpcyc10':[69, 1794]
            ,'scpcyc11':[70, 3968]
            ,'scpclr10':[71, 25]
            ,'scpclr11':[72, 23]
            ,'scpclr12':[73, 23]
            ,'scpclr13':[74, 23]
        }

    for nomInstancia in orden:
        if nomInstancia in archivoInstancia:
            #print(f"instancia {nomInstancia}")
            return orden[nomInstancia][1]

    return None