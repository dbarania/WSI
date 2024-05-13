import numpy as np
def sigmoid(Z):
    """
    Argumenty:
    Z -- tablica numpy o dowolnym kształcie

    Zwraca:
    A -- wynik funkcji sigmoid(z), taki sam kształt jak Z
    cache -- zwraca również Z, przydatne podczas propagacji wstecznej
    """
    A = np.exp(Z)/(1+np.exp(Z))
    cache = Z
    return A, cache


def relu(Z):
    """
    Argumenty:
    Z -- Wynik warstwy liniowej, dowolny kształt

    Zwraca:
    A -- Parametr po aktywacji, ten sam kształt co Z
    cache -- słownik pythona zawierający "A"; przechowywany do efektywnego obliczania propagacji wstecznej
    """
    A = (Z+np.abs(Z))/2
    cache = A #???
    return A, cache


def relu_backward(dA, cache):
    """
    Argumenty:
    dA -- gradient po aktywacji, dowolny kształt
    cache -- 'Z', przechowywane dla efektywnego obliczania propagacji wstecznej

    Zwraca:
    dZ -- Gradient kosztu względem Z
    """
    return dZ

def sigmoid_backward(dA, cache):
    """
    Argumenty:
    dA -- gradient po aktywacji, dowolny kształt
    cache -- 'Z', przechowywane dla efektywnego obliczania propagacji wstecznej

    Zwraca:
    dZ -- Gradient kosztu względem Z
    """
    return dZ


def load_data():
    """
    Zwraca:
    train_set_x_orig -- tablica numpy z cechami zestawu treningowego
    train_set_y_orig -- tablica numpy z etykietami zestawu treningowego
    test_set_x_orig -- tablica numpy z cechami zestawu testowego
    test_set_y_orig -- tablica numpy z etykietami zestawu testowego
    classes -- tablica numpy z listą klas
    """
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def initialize_parameters(n_x, n_h, n_y):
    """
    Argumenty:
    n_x -- rozmiar warstwy wejściowej
    n_h -- rozmiar warstwy ukrytej
    n_y -- rozmiar warstwy wyjściowej

    Zwraca:
    parameters -- słownik Pythona zawierający parametry:
                  W1 -- macierz wag o kształcie (n_h, n_x)
                  b1 -- wektor bias o kształcie (n_h, 1)
                  W2 -- macierz wag o kształcie (n_y, n_h)
                  b2 -- wektor bias o kształcie (n_y, 1)
    """
    return parameters


def linear_forward(A, W, b):
    """
    Argumenty:
    A -- aktywacje z poprzedniej warstwy (lub dane wejściowe): (rozmiar poprzedniej warstwy, liczba przykładów)
    W -- macierz wag: tablica numpy o kształcie (rozmiar bieżącej warstwy, rozmiar poprzedniej warstwy)
    b -- wektor bias, tablica numpy o kształcie (rozmiar bieżącej warstwy, 1)

    Zwraca:
    Z -- wejście funkcji aktywacji, nazywane również parametrem przed aktywacją
    cache -- słownik python zawierający "A", "W" i "b"; przechowywany do efektywnego obliczania propagacji wstecznej
    """
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Argumenty:
    A_prev -- aktywacje z poprzedniej warstwy (lub dane wejściowe): (rozmiar poprzedniej warstwy, liczba przykładów)
    W -- macierz wag: tablica numpy o kształcie (rozmiar bieżącej warstwy, rozmiar poprzedniej warstwy)
    b -- wektor bias, tablica numpy o kształcie (rozmiar bieżącej warstwy, 1)
    activation -- funkcja aktywacji używana w tej warstwie, przechowywana jako tekst: "sigmoid" lub "relu"

    Zwraca:
    A -- wynik funkcji aktywacji, nazywany również wartością po aktywacji
    cache -- słownik python zawierający "linear_cache" i "activation_cache";
             przechowywany do efektywnego obliczania propagacji wstecznej
    """
    return A, cache


def compute_cost(AL, Y):
    """
    Argumenty:
    AL -- wektor prawdopodobieństwa odpowiadający twoim przewidywaniom etykiet, kształt (1, liczba przykładów)
    Y -- prawdziwy wektor etykiet (na przykład zawierający 0, jeśli nie-kot, 1, jeśli kot), kształt (1, liczba przykładów)

    Zwraca:
    cost -- koszt entropii krzyżowej
    """
    return cost

def linear_backward(dZ, cache):
    """
    Argumenty:
    dZ -- Gradient kosztu względem liniowego wyjścia (bieżącej warstwy l)
    cache -- krotka wartości (A_prev, W, b) pochodząca z propagacji wprzód w bieżącej warstwie

    Zwraca:
    dA_prev -- Gradient kosztu względem aktywacji (poprzedniej warstwy l-1), taki sam kształt jak A_prev
    dW -- Gradient kosztu względem W (bieżącej warstwy l), taki sam kształt jak W
    db -- Gradient kosztu względem b (bieżącej warstwy l), taki sam kształt jak b
    """
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Argumenty:
    dA -- gradient po aktywacji dla bieżącej warstwy l
    cache -- krotka wartości (linear_cache, activation_cache) przechowywana dla efektywnego obliczania propagacji wstecznej
    activation -- funkcja aktywacji używana w tej warstwie, "sigmoid" lub "relu"

    Zwraca:
    dA_prev -- Gradient kosztu względem aktywacji (poprzedniej warstwy l-1), taki sam kształt jak A_prev
    dW -- Gradient kosztu względem W (bieżącej warstwy l), taki sam kształt jak W
    db -- Gradient kosztu względem b (bieżącej warstwy l), taki sam kształt jak b
    """
    return dA_prev, dW, db

def update_parameters(parameters, grads, learning_rate):
    """
    Argumenty:
    parameters -- słownik Pythona zawierający twoje parametry
    grads -- słownik Pythona zawierający twoje gradienty, wynik L_model_backward
    learning_rate -- współczynnik uczenia się

    Zwraca:
    parameters -- słownik Pythona zawierający zaktualizowane parametry
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """
    return parameters
