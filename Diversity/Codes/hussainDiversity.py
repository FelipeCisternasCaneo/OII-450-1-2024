def diversidadHussain(matriz):
    medianas = []
    
    for j in range(matriz[0].__len__()):
        suma = 0
        
        for i in range(matriz.__len__()):
            suma += matriz[i][j]
            
        medianas.append(suma / matriz.__len__())
    
    n = len(matriz)
    l = len(matriz[0])
    diversidad = 0
    
    for d in range(l):
        div_d = 0
    
        for i in range(n):
            div_d = div_d + abs(medianas[d] - matriz[i][d])
        
        diversidad = diversidad + div_d
        
    return round(((1 / (l * n)) * diversidad), 3)