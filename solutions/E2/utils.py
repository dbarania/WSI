import numpy as np
def himmelblau_function(x:float,y:float)->float:
    elem1 = (x**2 + y - 11)**2 
    elem2 = (x + y**2 - 7)**2
    return elem1+elem2

def ackley_function(x:float,y:float)->float:
    elem1 = np.exp(-0.2*np.sqrt(0.5*(x**2+y**2)))
    elem2 = np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))

    return -20*elem1-elem2+20+np.e

