import random
import numpy as np

class SCP:
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
        
        dirSCP = './Problem/SCP/Instances/'
        
        instance = dirSCP+instance+".txt" 
        
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
     
    def obtenerOptimo(self, archivoInstancia):
        orden = {
            'scp41':[0,429]
            ,'scp42':[1,512]
            ,'scp43':[2,516]
            ,'scp44':[3,494]
            ,'scp45':[4,512]
            ,'scp46':[5,560]
            ,'scp47':[6,430]
            ,'scp48':[7,492]
            ,'scp49':[8,641]
            ,'scp410':[9,514]
            ,'scp51':[10,253]
            ,'scp52':[11,302]
            ,'scp53':[12,226]
            ,'scp54':[13,242]
            ,'scp55':[14,211]
            ,'scp56':[15,213]
            ,'scp57':[16,293]
            ,'scp58':[17,288]
            ,'scp59':[18,279]
            ,'scp510':[19,265]
            ,'scp61':[20,138]
            ,'scp62':[21,146]
            ,'scp63':[22,145]
            ,'scp64':[23,131]
            ,'scp65':[24,161]
            ,'scpa1':[25,253]
            ,'scpa2':[26,252]
            ,'scpa3':[27,232]
            ,'scpa4':[28,234]
            ,'scpa5':[29,236]
            ,'scpb1':[30,69]
            ,'scpb2':[31,76]
            ,'scpb3':[32,80]
            ,'scpb4':[33,79]
            ,'scpb5':[34,72]
            ,'scpc1':[35,227]
            ,'scpc2':[36,219]
            ,'scpc3':[37,243]
            ,'scpc4':[38,219]
            ,'scpc5':[39,215]
            ,'scpd1':[40,60]
            ,'scpd2':[41,66]
            ,'scpd3':[42,72]
            ,'scpd4':[43,62]
            ,'scpd5':[44,61]
            ,'scpnre1':[45,29]
            ,'scpnre2':[46,30]
            ,'scpnre3':[47,27]
            ,'scpnre4':[48,28]
            ,'scpnre5':[49,28]
            ,'scpnrf1':[50,14]
            ,'scpnrf2':[51,15]
            ,'scpnrf3':[52,14]
            ,'scpnrf4':[53,14]
            ,'scpnrf5':[54,13]
            ,'scpnrg1':[55,176]
            ,'scpnrg2':[56,154]
            ,'scpnrg3':[57,166]
            ,'scpnrg4':[58,168]
            ,'scpnrg5':[59,168]
            ,'scpnrh1':[60,63]
            ,'scpnrh2':[61,63]
            ,'scpnrh3':[62,59]
            ,'scpnrh4':[63,58]
            ,'scpnrh5':[64,55]
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
            
            
def obtenerOptimo(archivoInstancia):
    orden = {
        'scp41':[0,429]
        ,'scp42':[1,512]
        ,'scp43':[2,516]
        ,'scp44':[3,494]
        ,'scp45':[4,512]
        ,'scp46':[5,560]
        ,'scp47':[6,430]
        ,'scp48':[7,492]
        ,'scp49':[8,641]
        ,'scp410':[9,514]
        ,'scp51':[10,253]
        ,'scp52':[11,302]
        ,'scp53':[12,226]
        ,'scp54':[13,242]
        ,'scp55':[14,211]
        ,'scp56':[15,213]
        ,'scp57':[16,293]
        ,'scp58':[17,288]
        ,'scp59':[18,279]
        ,'scp510':[19,265]
        ,'scp61':[20,138]
        ,'scp62':[21,146]
        ,'scp63':[22,145]
        ,'scp64':[23,131]
        ,'scp65':[24,161]
        ,'scpa1':[25,253]
        ,'scpa2':[26,252]
        ,'scpa3':[27,232]
        ,'scpa4':[28,234]
        ,'scpa5':[29,236]
        ,'scpb1':[30,69]
        ,'scpb2':[31,76]
        ,'scpb3':[32,80]
        ,'scpb4':[33,79]
        ,'scpb5':[34,72]
        ,'scpc1':[35,227]
        ,'scpc2':[36,219]
        ,'scpc3':[37,243]
        ,'scpc4':[38,219]
        ,'scpc5':[39,215]
        ,'scpd1':[40,60]
        ,'scpd2':[41,66]
        ,'scpd3':[42,72]
        ,'scpd4':[43,62]
        ,'scpd5':[44,61]
        ,'scpnre1':[45,29]
        ,'scpnre2':[46,30]
        ,'scpnre3':[47,27]
        ,'scpnre4':[48,28]
        ,'scpnre5':[49,28]
        ,'scpnrf1':[50,14]
        ,'scpnrf2':[51,15]
        ,'scpnrf3':[52,14]
        ,'scpnrf4':[53,14]
        ,'scpnrf5':[54,13]
        ,'scpnrg1':[55,176]
        ,'scpnrg2':[56,154]
        ,'scpnrg3':[57,166]
        ,'scpnrg4':[58,168]
        ,'scpnrg5':[59,168]
        ,'scpnrh1':[60,63]
        ,'scpnrh2':[61,63]
        ,'scpnrh3':[62,59]
        ,'scpnrh4':[63,58]
        ,'scpnrh5':[64,55]
    }

    for nomInstancia in orden:
        if nomInstancia in archivoInstancia:
            #print(f"instancia {nomInstancia}")
            return orden[nomInstancia][1]

    return None



