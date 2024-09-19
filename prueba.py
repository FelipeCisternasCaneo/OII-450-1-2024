import numpy as np
import pandas as pd
import os
import time
from Problem.SCP.problem import SCP

instancias = os.listdir('./Problem/SCP/Instances')
for instance in instancias:
    instancia = SCP(instance.split('.')[0])
    block_sizes = []
    i = 1
    df = pd.DataFrame(columns=['id','best block size', 'best time', 'worst block size', 'worst time', 'total time'])
    while i <= (instancia.getColumns()/10):
        block_sizes.append(i*10)
        i+=1
    for i in range(31):
        best_time = float('inf')
        best_block_size = None
        worst_time = 0
        worst_block_size = None
        
        suma = 0
        for block_size in block_sizes:
            start_time = time.time()
            solution = np.zeros(instancia.getColumns())
            flag, aux = instancia.factibilityTest(solution, block_size)
            if not flag: #solucion infactible
                solution = instancia.repair(solution, 'complex', block_size)
            fitness = instancia.fitness(solution, block_size)
            elapsed_time = time.time() - start_time
            if elapsed_time < best_time:
                best_time = elapsed_time
                best_block_size = block_size
            if elapsed_time > worst_time:
                worst_time = elapsed_time
                worst_block_size = block_size
            suma+=elapsed_time
        print(f"{instance.split('.')[0]}, BS: {best_block_size}, BT: {best_time:.6f}, WS: {worst_block_size}, WT: {worst_time:.6f}, TT: {suma:.6f}, total exp: {len(block_sizes)}, DF: {(((worst_time*100)/best_time)-100):.2f} %")
        df.loc[i] = [i, best_block_size, best_time, worst_block_size, worst_time, suma]
    df.to_csv(f'./Problem/SCP/Analisis/{instance.split(".")[0]}.csv', index=False)