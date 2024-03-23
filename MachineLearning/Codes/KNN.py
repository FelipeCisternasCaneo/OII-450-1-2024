import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef


def KNN(trainingData, testingData, trainingClass, testingClass, k):
        # entrenamiento del clasificador
        # con metric = 'minkowski' y p = 1 se esta utilizando la distancia de manhattan
        # con metric = 'minkowski' y p = 2 se esta utilizando la distancia euclidiana

        clasificador = KNeighborsClassifier(
            n_neighbors = k,
            metric      = 'minkowski',
            p           = 2
        )
        clasificador.fit( trainingData , trainingClass )

        # predicción del clasificador

        predictionClass = clasificador.predict(testingData)

        # cm = confusion_matrix(testingClass, predictionClass)
        accuracy    = np.round(accuracy_score(testingClass, predictionClass), decimals=3)
        f1Score     = np.round(f1_score(testingClass, predictionClass), decimals=3)
        presicion   = np.round(precision_score(testingClass, predictionClass), decimals=3)
        recall      = np.round(recall_score(testingClass, predictionClass), decimals=3)
        mcc         = np.round(matthews_corrcoef(testingClass, predictionClass), decimals=3)


        # return cm, accuracy, f1Score, presicion, recall, mcc
        return accuracy, f1Score, presicion, recall, mcc