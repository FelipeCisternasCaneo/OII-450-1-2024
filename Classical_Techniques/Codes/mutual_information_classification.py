from sklearn.feature_selection import mutual_info_classif

def MIR(x, y, threshold: float):
    
    mi = mutual_info_classif(x, y)
    fs_list = [1 if valor>=threshold else 0 for valor in mi]
    return fs_list