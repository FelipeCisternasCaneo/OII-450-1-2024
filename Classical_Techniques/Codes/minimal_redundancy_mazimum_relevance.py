from mrmr import mrmr_classif

def MRMR(x, y, n_variables):
    feature_list = mrmr_classif(x, y, K=n_variables)
    feature_names = x.columns.ravel()
    fs_list = [1 if variable in feature_list else 0 for variable in feature_names]
    
    return fs_list