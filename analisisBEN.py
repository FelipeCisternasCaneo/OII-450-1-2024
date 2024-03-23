import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from util import util
from BD.sqlite import BD

dirResultado = './Resultados/'
archivoResumenFitness = open(f'{dirResultado}resumen_fitness_BEN.csv', 'w')
archivoResumenTimes = open(f'{dirResultado}resumen_times_BEN.csv', 'w')
archivoResumenPercentage = open(f'{dirResultado}resumen_percentage_BEN.csv', 'w')

archivoResumenFitness.write("instance,best,avg. fitness, std fitness,best,avg. fitness, std fitness,best,avg. fitness, std fitness, best,avg. fitness, std fitness\n")
archivoResumenTimes.write("instance, min time (s), avg. time (s), std time (s), min time (s), avg. time (s), std time (s), min time (s), avg. time (s), std time (s), min time (s), avg. time (s), std time (s)\n")
archivoResumenPercentage.write("instance, avg. XPL%, avg. XPT%, avg. XPL%, avg. XPT%, avg. XPL%, avg. XPT%, avg. XPL%, avg. XPT%\n")


graficos = False
class InstancesMhs:
    def __init__(self):
        self.div = []
        self.fitness = []
        self.time = []
        self.xpl = []
        self.xpt = []
        self.bestFitness = []
        self.bestTime = []

# Listas de metaheuristicas a implementar
mhsList = ['PSO']
# Lista de colores de grafico por metaheuristica
color = ['r']

# Diccionario de metaheuristicas
mhs = {name: InstancesMhs() for name in mhsList}

bd = BD()

instancias = bd.obtenerInstancias(f'''
                                  "F1"
                                  ''')

for instancia in instancias:
    print(instancia)
    
    blob = bd.obtenerArchivos(instancia[1])
    corrida = 1
    
    archivoFitness = open(f'{dirResultado}fitness_'+instancia[1]+'.csv', 'w')
    archivoFitness.write('MH,FITNESS\n')

    for name in mhsList:
        mhs[name].div = []
        mhs[name].fitness = [] 
        mhs[name].time = []
        mhs[name].xpl = [] 
        mhs[name].xpt = []
        mhs[name].bestFitness = []
        mhs[name].bestTime = []
    
    for d in blob:
        nombreArchivo = d[0]
        archivo = d[1]

        direccionDestiono = './Resultados/Transitorio/'+nombreArchivo+'.csv'
        # print("-------------------------------------------------------------------------------")
        util.writeTofile(archivo,direccionDestiono)
        
        data = pd.read_csv(direccionDestiono)
        
        mh = nombreArchivo.split('_')[0]
        problem = nombreArchivo.split('_')[1]

        iteraciones = data['iter']
        fitness     = data['fitness']
        time        = data['time']
        xpl         = data['XPL']
        xpt         = data['XPT']
        
        for name in mhsList:
            if mh == name:
                mhs[name].fitness.append(np.min(fitness))
                mhs[name].time.append(np.round(np.sum(time),3))
                mhs[name].xpl.append(np.round(np.mean(xpl), decimals=2))
                mhs[name].xpt.append(np.round(np.mean(xpt), decimals=2))
                archivoFitness.write(f'{name},{str(np.min(fitness))}\n')
            
        if graficos:

            # fig , ax = plt.subplots()
            # ax.plot(iteraciones,fitness)
            # ax.set_title(f'Convergence {mh} \n {problem} run {corrida}')
            # ax.set_ylabel("Fitness")
            # ax.set_xlabel("Iteration")
            # plt.savefig(f'{dirResultado}/Graficos/Coverange_{mh}_{problem}_{corrida}.pdf')
            # plt.close('all')
            # print(f'Grafico de covergencia realizado {mh} {problem} ')
            
            figPER, axPER = plt.subplots()
            axPER.plot(iteraciones, xpl, color="r", label=r"$\overline{XPL}$"+": "+str(np.round(np.mean(xpl), decimals=2))+"%")
            axPER.plot(iteraciones, xpt, color="b", label=r"$\overline{XPT}$"+": "+str(np.round(np.mean(xpt), decimals=2))+"%")
            axPER.set_title(f'XPL% - XPT% {mh} \n {problem} run {corrida}')
            axPER.set_ylabel("Percentage")
            axPER.set_xlabel("Iteration")
            axPER.legend(loc = 'upper right')
            plt.savefig(f'{dirResultado}/Graficos/Percentage_{mh}_{problem}_{corrida}.pdf')
            plt.close('all')
            print(f'Grafico de exploracion y explotacion realizado para {mh}, problema: {problem}, corrida: {corrida} ')
        
        corrida +=1
        
        if corrida == 32:
            corrida = 1
        
        os.remove('./Resultados/Transitorio/'+nombreArchivo+'.csv')
    
    resumenFitness = resumenTimes = resumenPercentage = ''''''
    for name in mhsList:
        resumenFitness = resumenFitness + f''',{np.min(mhs[name].fitness)},{np.round(np.average(mhs[name].fitness),3)},{np.round(np.std(mhs[name].fitness),3)}''' 
        resumenTimes = resumenTimes + f''',{np.min(mhs[name].time)},{np.round(np.average(mhs[name].time),3)},{np.round(np.std(mhs[name].time),3)}'''
        resumenPercentage = resumenPercentage + f''',{np.round(np.average(mhs[name].xpl),3)},{np.round(np.average(mhs[name].xpt),3)}'''
    
    archivoResumenFitness.write(f'''{problem}{resumenFitness} \n''')
    archivoResumenTimes.write(f'''{problem}{resumenTimes} \n''')
    archivoResumenPercentage.write(f'''{problem}{resumenPercentage} \n''')

    blob = bd.obtenerMejoresArchivos(instancia[1],"")
    
    for d in blob:

        nombreArchivo = d[4]
        archivo = d[5]

        direccionDestiono = './Resultados/Transitorio/'+nombreArchivo+'.csv'
        util.writeTofile(archivo,direccionDestiono)
        
        data = pd.read_csv(direccionDestiono)
        
        mh = nombreArchivo.split('_')[0]
        problem = nombreArchivo.split('_')[1]

        iteraciones = data['iter']
        fitness     = data['fitness']
        time        = data['time']
        xpl         = data['XPL']
        xpt         = data['XPT']
        
        for name in mhsList:
            if mh == name:
                mhs[name].bestFitness = fitness
                mhs[name].bestTime = time
        
        os.remove('./Resultados/Transitorio/'+nombreArchivo+'.csv')

    print("------------------------------------------------------------------------------------------------------------")
    figPER, axPER = plt.subplots()
    for i,name in enumerate(mhsList):
        axPER.plot(iteraciones, mhs[name].bestFitness, color=color[i], label=name)
    axPER.set_title(f'Coverage \n {problem}')
    axPER.set_ylabel("Fitness")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc = 'upper right')
    plt.savefig(f'{dirResultado}/Best/fitness_{problem}.pdf')
    plt.close('all')
    print(f'Grafico de fitness realizado {problem} ')
    
    figPER, axPER = plt.subplots()
    for i,name in enumerate(mhsList):
        axPER.plot(iteraciones, mhs[name].bestTime, color=color[i], label=name)
    axPER.set_title(f'Time (s) \n {problem}')
    axPER.set_ylabel("Time (s)")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc = 'lower right')
    plt.savefig(f'{dirResultado}/Best/time_{problem}.pdf')
    plt.close('all')
    print(f'Grafico de time realizado {problem} ')
    
    
    archivoFitness.close()
    
    print("------------------------------------------------------------------------------------------------------------")
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------
    datos = pd.read_csv(dirResultado+"/fitness_"+instancia[1]+'.csv')
    figFitness, axFitness = plt.subplots()
    axFitness = sns.boxplot(x='MH', y='FITNESS', data=datos)
    axFitness.set_title(f'Fitness \n{instancia[1]}', loc="center", fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': 'black'})

    axFitness.set_title(f'Fitness \n{instancia[1]}', loc="center", fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': 'black'})
    axFitness.set_ylabel("Fitness")
    axFitness.set_xlabel("Metaheuristics")
    figFitness.savefig(dirResultado+"/boxplot/boxplot_fitness_"+instancia[1]+'.pdf')
    plt.close('all')
    print(f'Grafico de boxplot con fitness para la instancia {instancia[1]} realizado con exito')
    
    figFitness, axFitness = plt.subplots()
    axFitness = sns.violinplot(x='MH', y='FITNESS', data=datos, gridsize=50)
    axFitness.set_title(f'Fitness \n{instancia[1]}', loc="center", fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': 'black'})
    axFitness.set_ylabel("Fitness")
    axFitness.set_xlabel("Metaheuristics")
    figFitness.savefig(dirResultado+"/violinplot/violinplot_fitness_"+instancia[1]+'.pdf')
    plt.close('all')
    print(f'Grafico de violines con fitness para la instancia {instancia[1]} realizado con exito')
    
    os.remove(dirResultado+"/fitness_"+instancia[1]+'.csv')
    
    print("------------------------------------------------------------------------------------------------------------")

archivoResumenFitness.close()
archivoResumenTimes.close()
archivoResumenPercentage.close()
        