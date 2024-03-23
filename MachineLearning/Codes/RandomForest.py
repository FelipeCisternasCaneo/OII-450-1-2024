import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, accuracy_score, recall_score, f1_score, precision_score, recall_score, matthews_corrcoef
from sklearn.metrics import classification_report



def RandomForest(trainingData, testingData, trainingClass, testingClass):
    SEED = 12
    rf = RandomForestClassifier(
        criterion = 'gini', 
        n_estimators = 40, 
        max_depth = 40, 
        max_features = "log2", 
        n_jobs=-1, 
        random_state = SEED)
    

    
    rf.fit(trainingData, trainingClass)
    
    # Predict the Test set results
    predictionClass = rf.predict(testingData)
    
    # cm = confusion_matrix(testingClass, predictionClass)
    accuracy    = np.round(accuracy_score(testingClass, predictionClass), decimals=3)
    f1Score     = np.round(f1_score(testingClass, predictionClass), decimals=3)
    presicion   = np.round(precision_score(testingClass, predictionClass), decimals=3)
    recall      = np.round(recall_score(testingClass, predictionClass), decimals=3)
    mcc         = np.round(matthews_corrcoef(testingClass, predictionClass), decimals=3)


    # return cm, accuracy, f1Score, presicion, recall, mcc
    return accuracy, f1Score, presicion, recall, mcc