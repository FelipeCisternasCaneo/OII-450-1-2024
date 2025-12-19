def diversidadHussain(matriz):
    medianas = []
    n = len(matriz)
    n_cols = len(matriz[0])
    
    for j in range(n_cols):
        suma = 0
        
        for i in range(n):
            suma += matriz[i][j]
            
        medianas.append(suma / n)
    
    l = len(matriz[0])
    diversidad = 0
    
    for d in range(l):
        div_d = 0
    
        for i in range(n):
            div_d = div_d + abs(medianas[d] - matriz[i][d])
        
        diversidad = diversidad + div_d
        
    return round(((1 / (l * n)) * diversidad), 3)