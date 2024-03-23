from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def boruta(x, y, iter):
    borutaEstimator = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    boruta = BorutaPy(estimator =borutaEstimator, n_estimators = 'auto',  max_iter = iter) 
    Xdata = np.array(x)
    yvector = np.array(y)
    boruta.fit(Xdata, yvector)
    
    fsList = list (1 * np.logical_or(boruta.support_,boruta.support_weak_))
    
    return fsList