# class GradientDescent:
#     def __init__(self,func,grad_func,beta) -> None:
#         self.func = func
#         self.grad_func = grad_func
#         self.beta = beta

#     def eval_func(self,arg:tuple):
#         return self.func(arg[0],arg[1])
    
#     def eval_grad(self,arg:tuple):
#         return self.grad_func(arg[0],arg[1])
    
#     def new_position(self, position:tuple):
#         return np.array(position)-self.beta*np.array(self.eval_grad(position))

#     def find_minimum(self, start_position, steps):
#         position = start_position
#         for i in range(1,steps+1):
#             position = self.new_position(position)
#             print(f"{i}. {position}, grad {self.eval_grad(position)}, func {self.eval_func(position)

# def grad_g1(x, y):
#     gradient = (2*x*np.exp(-x**2-y**2)+(x-1)*np.exp(-(x-1)**2-(y+2)**2),
#                 2*y*np.exp(-x**2-y**2)+(y+2)*np.exp(-(x-1)**2-(y+2)**2))
#     return gradient

# solver1 = GradientDescent(g1,grad_g1,0.1)

# solver1.find_minimum((1.5,-3),100)
import math
from typing import Callable,Tuple
import numpy as np
import matplotlib.pyplot as plt
f1 = lambda x,y : (x**2 + y - 11)**2 + (x + y**2 - 7)**2
f1_grad = lambda x,y : (2*(-7 + x + y**2 + 2*x*(-11 + x**2 + y)),2*(-11 + x**2 + y + 2*y*(-7 + x + y**2)))

# f2 = lambda x,y: -20*math.exp(-0.2*math.sqrt(0.5*(x**2+y**2)))-math.exp(0.5*(math.cos(2*math.pi*x)+math.cos(2*math.pi*y)))+math.e+20
f2_grad = lambda x,y: ((2**(3/2)*x*math.exp(-0.2*math.sqrt((x**2+y**2)/2)))/math.sqrt(x**2+y**2)+math.pi*math.sin(2*math.pi*x)*math.exp(0.5*(math.cos(2*math.pi*x)+math.cos(2*math.pi*y))),(2**(3/2)*y*math.exp(-0.2*math.sqrt((x**2+y**2)/2)))/math.sqrt(x**2+y**2)+math.pi*math.sin(2*math.pi*y)*math.exp(0.5*(math.cos(2*math.pi*x)+math.cos(2*math.pi*y))))

def f2(x,y):
    return -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2)))-np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))+np.e+20

def gradient_descent(func:Callable[[float,float],float],grad_func:Callable[[float,float],float],start_pos:Tuple[float,float],steps:int,beta:float):
    print(func(0,0))
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
    # print(func(]))
    # print(grad_func(*trace[-1][1]))
    return trace

res = gradient_descent(f2,f2_grad,(2,-2),10000,0.35)
print(res[-1])
# print(res)

x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
x, y = np.meshgrid(x, y)
z = f2(x, y)

plt.contourf(x, y, z, cmap='viridis')
plt.scatter(*res[-1],s=10,color="red")
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contour Plot of f(x, y)')
plt.show()