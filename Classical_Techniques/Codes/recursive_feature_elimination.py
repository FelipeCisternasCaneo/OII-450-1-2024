from sklearn.feature_selection import RFE
from sklearn.ensemble import AdaBoostClassifier

def rfe(x, y, n_variables):
    
    wrapper = RFE(estimator=AdaBoostClassifier(),n_features_to_select=n_variables)
    wrapper.fit(x, y)
    
    fs_list = list(1*wrapper.support_)
    
    return fs_list