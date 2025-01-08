from Diversity.imports import diversidadHussain, porcentajesXLPXPT

def initialize_diversity(population):
    maxDiversity = diversidadHussain(population)
    XPL, XPT, _ = porcentajesXLPXPT(maxDiversity, maxDiversity)
    
    return maxDiversity, XPL, XPT

def calculate_diversity(population, maxDiversity):
    div_t = diversidadHussain(population)
    
    if maxDiversity < div_t:
        maxDiversity = div_t
    
    XPL, XPT, _ = porcentajesXLPXPT(div_t, maxDiversity)
    
    return div_t, maxDiversity, XPL, XPT