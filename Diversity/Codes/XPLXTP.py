def porcentajesXLPXPT(div, maxDiv):
    XPL = round((div / maxDiv) * 100, 2)
    XPT = round((abs(div - maxDiv) / maxDiv) * 100, 2)
    state = -1
    # Determinar estado
    
    if XPL >= XPT:
        state = 1 # Exploración
        
    else:
        state = 0 # Explotación
    
    return XPL, XPT, state