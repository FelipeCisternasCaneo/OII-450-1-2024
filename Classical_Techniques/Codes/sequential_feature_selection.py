from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier

def SFS(x, y, n_variables):
    
    wrapper = SequentialFeatureSelector(estimator=KNeighborsClassifier(n_neighbors=5), 
                                        n_features_to_select=n_variables,
                                        scoring='f1_weighted')
    wrapper.fit(x, y)
    feature_list = list(wrapper.get_support())
    fs_list = [1 if variable else 0 for variable in feature_list]
    
    return fs_list