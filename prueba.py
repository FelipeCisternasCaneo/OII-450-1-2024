import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#script v1 (manejo de outliers)
dir = 'Problem/FS/Instances/'
#cargar dataset (91183,166)
df = pd.read_csv(dir+"dat_3_3_1.csv")

#eliminar columnas con mas del 10 por ciento de datos faltantes (91183,70)
cols2drop = df.columns[((df.isna().sum(axis=0) / df.shape[0]) > 0.1)]
df.drop(cols2drop, axis=1, inplace=True)

#eliminar columnas tas_diff y hypo_type, one hot encoding a variables categoricas
df.drop(['Hypo_Type','TAS_Diff'], axis=1, inplace=True)
df = pd.get_dummies(df, columns=['H0_Dializador_2', 'H0_Bano_2'])


#transformar outliers a valores Nan
whisker_width_na = 3.5
check_for_outliers = ["NEUTROFILOS#", "MONOCITOS#", "LINFOCITOS#", "V.C.M.", "H.C.M.", "EOSINOFILOS#", "BASOFILOS#", "LEUCOCITOS", "LUC#", "FILTRACIONGLOMERULARCKD-EPI",
            "GGT", "AST/GOT", "PCR", "BILIRRUBINATOTAL", "FOSFATASAALCALINA", "FERRITINA", "PTHintacta", "TRIGLICERIDOS", "BETA2MICROGLOBULINASUERO",
            "VITAMINAB12", "H0_Ganancia_2", "H0_UF_2", "H0_Pulso_2", "H0_ConductividadBano_2", "H0_FlujoSangre_2", "H0_Ganancia_2", "H0_PresionArterial_2", "H0_TAD_2"]
for i in check_for_outliers:
    q1, q3 = df[i].quantile([0.25, 0.75])
    iqr = q3 - q1
    df.loc[df[i] < q1 - whisker_width_na * iqr, i] = np.nan
    df.loc[df[i] > q3 + whisker_width_na * iqr, i] = np.nan
df.loc[df['H0_TAS_2'].abs()>200, 'H0_TAS_2'] = np.nan
df.loc[df['H0_TAS_2'].abs()<80, 'H0_TAS_2'] = np.nan
df.loc[df['H0_PTM_2']>300, 'H0_PTM_2'] = np.nan
df.loc[df['H0_PTM_2'] <=0, 'H0_PTM_2'] = np.nan
df.loc[df['H0_PresionVenosa_2']>=350, 'H0_PresionVenosa_2'] = np.nan
df["PCR"] = np.log10(df.PCR)

#normalizar datos usando minmax
scaler = MinMaxScaler()

df_norm = scaler.fit_transform(df)

# imputacion de datos (puede tomar tiempo)
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)

# fit the imputer to the DataFrame
imputer.fit(df_norm)

# transform the DataFrame using the imputer
imputed_df = imputer.transform(df_norm)

# print the imputed DataFrame
imputed_df.to_csv(dir+'dat_3_3_1_v1.csv')





# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import os
# import seaborn as sns
# from util import util
# from BD.sqlite import BD
# bd = BD()
# tablas = False
# graficos = True
# dirResultado = './Resultados/'

# instancia = 'scp41'

# mhs = bd.obtenerTecnicas()
# rendimientos    = dict()
# experimentos = bd.consultaPrueba()
# for experimento in experimentos:
#     # instancia = 'scp41' for de instancias
#     mejor = bd.obtenerMejoresEjecuciones(instancia, 'GWO', experimento[0])
#     print(f'Analizando experimento {experimento[0]} asociado a la instancia {instancia} metaheuristica GWO')
#     for m in mejor:
#         id                  = m[0]
#         nombre_archivo      = m[2]
#         archivo             = m[3]
#         direccionDestiono = './Resultados/Transitorio/'+nombre_archivo+'.csv'
#         # print("-------------------------------------------------------------------------------")
#         util.writeTofile(archivo,direccionDestiono)                        
#         data = pd.read_csv(direccionDestiono, on_bad_lines='skip')
#         iteraciones = data['iter']
#         fitness     = data['fitness']
#         rendimientos[f'{experimento[0]} - {instancia}'] = fitness

# figCOM, axCOM = plt.subplots()
# for clave in rendimientos:
#     print(clave)
#     etiqueta = f'{clave.split("-")[1]}'
#     if 'GWO' in clave and instancia in clave and 'COM' in clave:
#         print("si")
#         axCOM.plot(iteraciones, rendimientos[clave],  label=etiqueta)

# axCOM.set_title(f'Coverage {instancia} - GWO - COM')
# axCOM.set_ylabel("Fitness")
# axCOM.set_xlabel("Iteration")
# axCOM.legend(loc = 'upper right')
# plt.savefig(f'{dirResultado}/Best/fitness_{instancia}_GWO_COM.pdf')
# plt.close('all')
# print(f'Grafico de fitness realizado {instancia} - GWO - COM')


# from Problem.KP.problem import KP
# import numpy as np
# import pandas as pd
# from Problem.FS.Problem import FeatureSelection as fs

# instance = fs('CTG')

# print(instance.getClases())

# print(instance.getDatos())




# instance = KP('f1_l-d_kp_10_269')

# print(instance.getCapacity())
# print(instance.getItems())
# print(instance.getProfits())
# print(instance.getTradeOff())
# print(instance.getWeights())
# print(instance.getOptimum())

# print("----------------------------------------------------------------")

# # solution = np.array([1,1,1,1,1,1,1,1,1,1])
# solution = np.random.randint(low=0, high=2, size = (instance.getItems()))

# print(solution)

# print(instance.factibilityTest(solution))

# print("----------------------------------------------------------------")

# solution = instance.repair(solution)

# print(solution)

# print(instance.fitness(solution))
# print(instance.factibilityTest(solution))


# print("----------------------------------------------------------------")



# ds = 'COM'

# separacion = ds.split("_")

# print(len(separacion))





















# from Problem.EMPATIA.database.prepare_dataset import prepare_47vol_solap

# data_dir = './Problem/EMPATIA/'

# loader = prepare_47vol_solap(data_dir)

# ids = loader.dataset['vol_id'].unique()

# print(ids)

# eliminar = "["

# i = 0
# for id in ids:
#     if i == 12 or i == 13 or i == 23 or i == 25:
#         print(id)
#     else:
#         eliminar = eliminar + "," + str(id)
#     i+=1
    
# print(eliminar)

# loader.dataset = loader.exclude_volunteers([1,3,4,10,11,13,16,20,23,24,25,26,31,35,36,37,38,41,42,44,46,49,52,53,54,56,59,61,62,68,70,71,75,79,81,83,86,93,95,103,104,108,113])

# ids = loader.dataset['vol_id'].unique()

# print(ids)











# import pandas as pd
# from scipy.stats import mannwhitneyu
# from BD.sqlite import BD

# bd = BD()

# tecnicas = bd.obtenerTecnicas()


# for tecnica in tecnicas:
#     for t in tecnicas:
#         if t[0] != tecnica[0]:
#             archivo = open(f'./Resultados/Test_Estadisticos/{tecnica[0]}_contra_{t[0]}.csv', 'w')
#             archivo.close()

# datos = pd.read_csv('./Resultados/fscore_EMPATIA-V01.csv')


# # print(len(tecnicas))
# # i = 1
# # for tecnica in tecnicas:
# #     test_estadistico.write(f' {tecnica[0]} ')
# #     if i < len(tecnicas):
# #         test_estadistico.write(f' , ')
# #     else:
# #         test_estadistico.write(f' \n ')
# #     i += 1

# for tecnica in tecnicas:
#     data_x = datos[datos['MH'].isin([tecnica[0]])]
#     x = data_x['FSCORE']
#     for t in tecnicas:
#         if t[0] != tecnica[0]:
#             data_y = datos[datos['MH'].isin([t[0]])]
#             y = data_y['FSCORE']
#             p_value = mannwhitneyu(x,y, alternative='greater')
#             print(f'Comparando {tecnica[0]} contra {t[0]}: {p_value[1]}')
#             archivo = open(f'./Resultados/Test_Estadisticos/{tecnica[0]}_contra_{t[0]}.csv', 'a')
#             archivo.write(f'{p_value[1]}\n')

# print("------------------------------------------------------------------------------------")
# datos = pd.read_csv('./Resultados/fscore_EMPATIA-V02.csv')

# for tecnica in tecnicas:
#     data_x = datos[datos['MH'].isin([tecnica[0]])]
#     x = data_x['FSCORE']
#     for t in tecnicas:
#         if t[0] != tecnica[0]:
#             data_y = datos[datos['MH'].isin([t[0]])]
#             y = data_y['FSCORE']
#             p_value = mannwhitneyu(x,y, alternative='greater')
#             print(f'Comparando {tecnica[0]} contra {t[0]}: {p_value[1]}')
#             archivo = open(f'./Resultados/Test_Estadisticos/{tecnica[0]}_contra_{t[0]}.csv', 'a')
            # archivo.write(f'{p_value[1]}\n')









        

# x = [1,2,3,4,5,6,7,8,9,10]
# y = [11,22,33,44,55,66,77,88,99,100]

# 

# print(p_value)


# from Problem.FS.Problem import FeatureSelection as fs
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score, KFold
# from sklearn.ensemble import RandomForestClassifier
# # instance = fs('ionosphere')
# instance = fs('LSVT')


# print(instance.getClases())

# print(instance.getDatos().values)


# SEED = 12
# classifier = RandomForestClassifier(random_state = SEED)
        
# # scoring=['accuracy','f1','precision','recall']
# print("------------------------------------------------------------------------------------------------------")
# for i in range(20):
#     accuracy_cross    = cross_val_score(estimator=classifier, X=instance.getDatos().values, y=instance.getClases(), cv=5, n_jobs=5, scoring='accuracy')
#     f1Score_cross     = cross_val_score(estimator=classifier, X=instance.getDatos().values, y=instance.getClases(), cv=5, n_jobs=5, scoring='f1')
#     presicion_cross   = cross_val_score(estimator=classifier, X=instance.getDatos().values, y=instance.getClases(), cv=5, n_jobs=5, scoring='precision')
#     recall_cross      = cross_val_score(estimator=classifier, X=instance.getDatos().values, y=instance.getClases(), cv=5, n_jobs=5, scoring='recall')
#     print(accuracy_cross, np.round(accuracy_cross.mean(), decimals=3))
#     print(f1Score_cross, np.round(f1Score_cross.mean(), decimals=3))
#     print(presicion_cross, np.round(presicion_cross.mean(), decimals=3))
#     print(recall_cross, np.round(recall_cross.mean(), decimals=3))
#     print("------------------------------------------------------------------------------------------------------")


















# from Problem.FS.Problem import FeatureSelection as fs
# from Classical_Techniques.mutual_information_classification import MIR
# from Classical_Techniques.minimal_redundancy_mazimum_relevance import MRMR
# from Classical_Techniques.sequential_feature_selection import SFS
# from Classical_Techniques.recursive_feature_elimination import rfe
# from Classical_Techniques.boruta import boruta
# import numpy as np
# import time

# instancia = 'breast-cancer-wisconsin'

# instance = fs(instancia)

# print(instance.getDatos())
# print(instance.getClases())

# for i in np.arange(0.001, 0.25, 0.001):
#     tiempoInicializacion1 = time.time()
#     lista = MIR(instance.getDatos(), instance.getClases(), threshold = i)
    
#     seleccion = np.where(np.array(lista) == 1)[0]
#     clasificador = 'KNN'
#     parametrosC = 'k:5'
#     fitness, accuracy, f1Score, presicion, recall, mcc, errorRate, totalFeatureSelected = instance.fitness(seleccion, clasificador, parametrosC)
#     tiempoInicializacion2 = time.time()
#     print(
#         f'{lista}'+
#         f', b: {str(fitness)}'+
#         f', t: {str(round(tiempoInicializacion2-tiempoInicializacion1,3))}'+
#         f', a: {str(accuracy)}'+
#         f', fs: {str(f1Score)}'+
#         f', p: {str(presicion)}'+
#         f', rc: {str(recall)}'+
#         f', mcc: {str(mcc)}'+
#         f', eR: {str(errorRate)}'+
#         f', TFS: {str(totalFeatureSelected)}'
#     )

# for i in range(5, instance.getTotalFeature()):
#     tiempoInicializacion1 = time.time()
#     lista = MRMR(instance.getDatos(), instance.getClases(), n_variables=i)
    
#     seleccion = np.where(np.array(lista) == 1)[0]
#     clasificador = 'KNN'
#     parametrosC = 'k:5'
#     fitness, accuracy, f1Score, presicion, recall, mcc, errorRate, totalFeatureSelected = instance.fitness(seleccion, clasificador, parametrosC)
#     tiempoInicializacion2 = time.time()
#     print(
#         f'{lista}'+
#         f', b: {str(fitness)}'+
#         f', t: {str(round(tiempoInicializacion2-tiempoInicializacion1,3))}'+
#         f', a: {str(accuracy)}'+
#         f', fs: {str(f1Score)}'+
#         f', p: {str(presicion)}'+
#         f', rc: {str(recall)}'+
#         f', mcc: {str(mcc)}'+
#         f', eR: {str(errorRate)}'+
#         f', TFS: {str(totalFeatureSelected)}'
#     )

# for i in range(5, instance.getTotalFeature()):
#     tiempoInicializacion1 = time.time()
#     lista = SFS(instance.getDatos(), instance.getClases(), n_variables=i)
    
#     seleccion = np.where(np.array(lista) == 1)[0]
#     clasificador = 'KNN'
#     parametrosC = 'k:5'
#     fitness, accuracy, f1Score, presicion, recall, mcc, errorRate, totalFeatureSelected = instance.fitness(seleccion, clasificador, parametrosC)
#     tiempoInicializacion2 = time.time()
#     print(
#         f'{lista}'+
#         f', b: {str(fitness)}'+
#         f', t: {str(round(tiempoInicializacion2-tiempoInicializacion1,3))}'+
#         f', a: {str(accuracy)}'+
#         f', fs: {str(f1Score)}'+
#         f', p: {str(presicion)}'+
#         f', rc: {str(recall)}'+
#         f', mcc: {str(mcc)}'+
#         f', eR: {str(errorRate)}'+
#         f', TFS: {str(totalFeatureSelected)}'
#     )

# for i in range(5, instance.getTotalFeature()):
#     tiempoInicializacion1 = time.time()
#     lista = rfe(instance.getDatos(), instance.getClases(), n_variables=i)
    
#     seleccion = np.where(np.array(lista) == 1)[0]
#     clasificador = 'KNN'
#     parametrosC = 'k:5'
#     fitness, accuracy, f1Score, presicion, recall, mcc, errorRate, totalFeatureSelected = instance.fitness(seleccion, clasificador, parametrosC)
#     tiempoInicializacion2 = time.time()
#     print(
#         f'{lista}'+
#         f', b: {str(fitness)}'+
#         f', t: {str(round(tiempoInicializacion2-tiempoInicializacion1,3))}'+
#         f', a: {str(accuracy)}'+
#         f', fs: {str(f1Score)}'+
#         f', p: {str(presicion)}'+
#         f', rc: {str(recall)}'+
#         f', mcc: {str(mcc)}'+
#         f', eR: {str(errorRate)}'+
#         f', TFS: {str(totalFeatureSelected)}'
#     )

# for i in np.arange(10, 100, 10):
#     tiempoInicializacion1 = time.time()
#     lista = boruta(instance.getDatos(), instance.getClases(), i)
    
#     seleccion = np.where(np.array(lista) == 1)[0]
#     clasificador = 'KNN'
#     parametrosC = 'k:5'
#     fitness, accuracy, f1Score, presicion, recall, mcc, errorRate, totalFeatureSelected = instance.fitness(seleccion, clasificador, parametrosC)
#     tiempoInicializacion2 = time.time()
#     print(
#         f'{lista}'+
#         f', b: {str(fitness)}'+
#         f', t: {str(round(tiempoInicializacion2-tiempoInicializacion1,3))}'+
#         f', a: {str(accuracy)}'+
#         f', fs: {str(f1Score)}'+
#         f', p: {str(presicion)}'+
#         f', rc: {str(recall)}'+
#         f', mcc: {str(mcc)}'+
#         f', eR: {str(errorRate)}'+
#         f', TFS: {str(totalFeatureSelected)}'
#     )