from Util.util import cargar_configuracion

def get_top_k(mh):
    config = cargar_configuracion('util/json/experiments_config.json')
    
    top_k_values = config['top_k_values']
    
    #print(top_k_values[mh])
    
    return top_k_values[mh]