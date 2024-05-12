import numpy as np
import matplotlib.pyplot as plt
from utilis_NN import *

def two_layer_model(X, Y, layers_dims, learning_rate=???, num_iterations=???):
    """
    Implementacja dwuwarstwej siecu neuronoweh:
    
    Argumenty:
    X -- dane wejściowe, kształt 
    Y -- wektor prawdziwych etykiet ( 1 - 'kot', 0 - 'nie-kot'),
    layers_dims -- wymiary warstw
    num_iterations -- liczba iteracji 
    learning_rate -- współczynnik uczenia się 
    
    Zwraca:
    parameters -- słownik zawierający W1, W2, b1, b2
    """
    
    np.random.seed(1)
    grads = {}
    costs = []  # do śledzenia kosztu
    m = X.shape[1]  # liczba przykładów0075
    (n_x, n_h, n_y) = layers_dims
    
    # Inicjalizacja słownika parametrów
    parameters = ???  #  kod inicjalizacji parametrów
    
    # Pętla (gradient prosty)
    for i in range(???):

        # Propagacja wprzód:
        A1, cache1 = ???  #  kod propagacji wprzód dla pierwszej warstwy
        A2, cache2 = ???  #  kod propagacji wprzód dla drugiej warstwy
        
        # Obliczanie kosztu
        cost = ???  #  kod obliczania kosztu
        
        # Inicjalizacja propagacji wstecznej
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        # Propagacja wsteczna
        dA1, dW2, db2 = ???  #  kod propagacji wstecznej dla drugiej warstwy
        dA0, dW1, db1 = ???  #  kod propagacji wstecznej dla pierwszej warstwy
        
        # Aktualizacja parametrów
        parameters = ???  #  kod aktualizacji parametrów
        
        # Wyświetlanie kosztów co 100 iteracji
        if print_cost and i % 100 == 0:
            print(f"Koszt po iteracji {i}: {np.squeeze(cost)}")
        if print_cost and i % 100 == 0:
            costs.append(cost)

    
    return parameters
