import numpy as np

# Narwhal Optimization (NO)
# https://doi.org/10.34028/iajit/21/3/6

def iterarNO(maxIter, iter, dim, population, best):
    population = np.array(population)
    best = np.array(best)
    
    alpha_inicial = 0.5  # Controla la intensidad de la señal
    beta_inicial = 0.1   # Regula el ajuste de la posición
    alpha = alpha_inicial * (1 - iter / maxIter)  # alpha disminuye con las iteraciones
    beta = beta_inicial * (iter / maxIter)  # beta aumenta con las iteraciones
    sigma_inicial = 1.0
    delta_t = 0.01
    sigma_t = sigma_inicial * (1 - iter / maxIter)  # Decae con el tiempo

    # Función de distancia de Hamming
    def distancia_hamming(X_i, X_presa):
        return np.sum(X_i != X_presa)
    
    # Función sigmoide para la probabilidad de flip de bits
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    for i in range(population.shape[0]):
        # Emisión de la señal (ajustada para distancia de Hamming)
        dist_hamming = distancia_hamming(population[i], best)
        SE = 0.1 / (1 + alpha * dist_hamming)
        
        # Propagación de la señal (función gaussiana basada en distancia de Hamming)
        PR = np.exp(-dist_hamming**2 / (2 * sigma_t**2))
        
        # Ajuste basado en la propagación
        ajuste_continuo = beta * delta_t * PR
        prob_flip = sigmoid(ajuste_continuo)

        # Actualización de la posición binaria
        for j in range(dim):
            if np.random.rand() < prob_flip:
                population[i, j] = 1 - population[i, j]  # Flip del bit (de 0 a 1 o de 1 a 0)

    return population