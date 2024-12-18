import cProfile
import json

from Solver.solverBEN import solverB
from Solver.solverSCP import solverSCP
from Solver.solverUSCP import solverUSCP

from BD.sqlite import BD

def main():
    bd = BD()
    
    data = bd.obtenerExperimento()
    
    id              = 0
    experimento     = ''
    instancia       = ''
    problema        = ''
    mh              = ''
    parametrosMH    = ''
    maxIter         = 0
    pop             = 0
    dim             = 0 
    ds              = []
    
    while data != None:
        print("-------------------------------------------------------------------------------------------------------")
        print(data)
        
        id = int(data[0][0])
        id_instancia = int(data[0][9])
        datosInstancia = bd.obtenerInstancia(id_instancia)
        
        problema = datosInstancia[0][1]
        instancia = datosInstancia[0][2]
        parametrosInstancia = datosInstancia[0][4]
        experimento = data[0][1]
        mh = data[0][2]
        parametrosMH = data[0][3]
        ml = data[0][4]
        
        maxIter = int(parametrosMH.split(",")[0].split(":")[1])
        pop = int(parametrosMH.split(",")[1].split(":")[1])
        ds = []
        
        if problema == 'BEN':
            bd.actualizarExperimento(id, 'ejecutando')
            dim = int(experimento.split(" ")[1])
            lb = float(parametrosInstancia.split(",")[0].split(":")[1])
            ub = float(parametrosInstancia.split(",")[1].split(":")[1])
            
            solverB(id, mh, maxIter, pop, instancia, lb, ub, dim)
        
        if problema == 'SCP':
            bd.actualizarExperimento(id, 'ejecutando')
            
            instancia = f'scp{datosInstancia[0][2]}'
            
            print("-------------------------------------------------------------------------------------------------------")
            print(f"Ejecutando el experimento: {experimento} - id: {str(id)}")
            print("-------------------------------------------------------------------------------------------------------")
            
            repair = parametrosMH.split(",")[3].split(":")[1]
            ds.append(parametrosMH.split(",")[2].split(":")[1].split("-")[0])
            ds.append(parametrosMH.split(",")[2].split(":")[1].split("-")[1])
            
            parMH = parametrosMH.split(",")[4]
            
            solverSCP(id, mh, maxIter, pop, instancia, ds, repair, parMH)
            
        if problema == 'USCP':
            bd.actualizarExperimento(id, 'ejecutando')
            
            instancia = f'uscp{datosInstancia[0][2][1:]}'
            
            print("-------------------------------------------------------------------------------------------------------")
            print(f"Ejecutando el experimento: {experimento} - id: {str(id)}")
            print("-------------------------------------------------------------------------------------------------------")
            
            repair = parametrosMH.split(",")[3].split(":")[1]
            ds.append(parametrosMH.split(",")[2].split(":")[1].split("-")[0])
            ds.append(parametrosMH.split(",")[2].split(":")[1].split("-")[1])
            
            parMH = parametrosMH.split(",")[4]
            
            solverUSCP(id, mh, maxIter, pop, instancia, ds, repair, parMH)
        
        data = bd.obtenerExperimento()
    
    print("------------------------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------------------------")
    print("Se han ejecutado todos los experimentos pendientes.")
    print("------------------------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------------------------")

if __name__ == "__main__":
    main()
    # cProfile.run('main()')