from typing import Callable,Tuple
import numpy as np
import matplotlib.pyplot as plt

def himmelblau_function(x,y):
    elem1 = (x**2 + y - 11)**2 
    elem2 = (x + y**2 - 7)**2
    return elem1+elem2

def himmelblau_funnction_gradient(x,y):
    dx = 2*(2*x*(x**2+y-11)+x+y**2-7)
    dy = 2*(x**2+2*y*(x+y**2-7)+y-11)
    return (dx,dy)

def ackley_function(x,y):
    elem1 = np.exp(-0.2*np.sqrt(0.5*(x**2+y**2)))
    elem2 = np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))
    return -20*elem1-elem2+20+np.e

def ackley_function_gradient(x,y):
    def elem1(a,b):
        return 2**(3/2)*a*np.exp(-0.2/np.sqrt(2)*np.sqrt(a**2+b**2)/np.sqrt(a**2+b**2))
    def elem2(a,b):
        return np.pi*np.sin(np.pi*2*a)*np.exp(0.5*(np.cos(np.pi*2*a)+np.cos(np.pi*2*b)))
    dx = elem1(x,y)+elem2(x,y)
    dy = elem1(y,x)+elem2(y,x)
    return(dx,dy)

def gradient_descent(func:Callable[[float,float],float],grad_func:Callable[[float,float],float],start_pos:Tuple[float,float],steps:int,beta:float)->list[list[tuple,tuple]]:
    position = start_pos
    trace = [position]
    def calculate_new_position():
        grad_value = grad_func(position[0],position[1])
        e1 = position[0]-beta*grad_value[0]
        e2 = position[1]-beta*grad_value[1]
        return (e1,e2)
    for i in range(1,steps+1):
        position=calculate_new_position()
        trace.append(position)
    return trace

def plot_function(func:Callable[[float,float],float], trace:list):
    X = np.arange(-5, 5, 0.1)
    Y = np.arange(-5,5, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = func(X,Y)

X = np.arange(-5, 5, 0.1)
Y = np.arange(-5,5, 0.1)
X, Y = np.meshgrid(X, Y)
Z = himmelblau_function(X,Y)
r = gradient_descent(himmelblau_function,himmelblau_funnction_gradient,(0,0),100,0.01)
a = list(zip(*r))
plt.contourf(X,Y,Z,200,cmap="coolwarm")
plt.plot(a[0],a[1],color="red")
plt.colorbar()
plt.show()