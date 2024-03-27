import numpy as np
import random
from typing import Callable
from copy import deepcopy
from utils import himmelblau_function,ackley_function
import matplotlib.pyplot as plt

class Individual:
    def __init__(self,position:list[float]) -> None:
        self.position = position
        self.size = len(position)
        self.value = None

    def evaluate(self, func:Callable)->None:
        self.value = func(*self.position)
    
    def __lt__(self,other:'Individual'):
        return self.value<other.value
      
    def __str__(self) -> str:
        return f"Position: [{self.position[0]:.5f}, {self.position[1]:.5f}]\t Value: {self.value:.5f}"
    
    def mutate(self, strength:float)->None:
        temp_position = []
        for gene in self.position:
            temp_position.append(gene + strength*random.gauss(0,1))
        self.position = temp_position
    
    def crossover(self,other:'Individual')->'Individual':
        assert self.size == other.size
        result_position = []
        for i in range(self.size):    
            weight = random.random()
            gene = self.position[i]*weight+other.position[i]*(1-weight)
            result_position.append(gene)
        return Individual(result_position)


def generate_population(size:int,restrictions:list[tuple[float,float]])->list[Individual]:
    population = list()
    for _ in range(size):
        position = [random.uniform(restrictions[i][0],restrictions[i][1]) for i in range(len(restrictions))]
        population.append(Individual(position))
    return population



def rate_individuals(population:list[Individual],func:Callable[[list[float],float],float])->None:
    for individual in population:
        individual.evaluate(func)
    population.sort(key=lambda ind:ind.value)



def tournament_selection(population:list[Individual],competition=2):
    temp_population = []
    while len(temp_population)<len(population):
        group = random.choices(population,k=competition)
        temp_population.append(min(group,key=lambda ind:ind.value))
    return temp_population


def crossover_phase(population:list[Individual], probability):
    temp_population = []
    for individual in population:
        if random.random()<probability:
            other = random.choice(population)
            temp_population.append(individual.crossover(other))
        else:
            temp_population.append(individual)
    return temp_population


def mutate_phase(population:list[Individual],mutation_strentgth):
    temp_population = []
    for individual in population:
        individual.mutate(mutation_strentgth)
        temp_population.append(individual)
    return temp_population

def succession_phase(population:list[Individual],best:Individual):
    population.append(deepcopy(best))
    population.sort(key = lambda ind:ind.value)
    population.pop()


def plot_trace(func:Callable,trace:list[Individual]):
    X = np.arange(-10, 10, 0.1)
    Y = np.arange(-10, 10, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = func(X,Y)
    plt.contourf(X,Y,Z,100,cmap="jet")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.title(func.__name__.replace("_"," "))
    trace_x = [best.position[0] for best in trace]
    trace_y = [best.position[1] for best in trace]
    plt.scatter(trace_x,trace_y,s=5,c="red")
    plt.show()

def plot_values(trace:list[Individual]):
    trace_values = [best.value for best in trace]
    steps = range(len(trace_values))
    print(steps)
    plt.plot(steps,trace_values)
    plt.yscale("log")
    plt.show()

def print_trace(trace:list[Individual]):
    best = trace[0]
    for i, curr in enumerate(trace):
        if curr<best:
            best=curr
            print(f"Step: {i}\t {curr}")
            


def evolutionary_algorithm(func:Callable,start_population:list[Individual],iteration_time:int,mutation_strength:float,crossover_probability:float):
    population = deepcopy(start_population)
    rate_individuals(population,func)
    best_individual = deepcopy(population[0])
    trace = [best_individual]
    for step in range(iteration_time):
        population = tournament_selection(population,2)
        population = crossover_phase(population,crossover_probability)
        population = mutate_phase(population,mutation_strength)
        rate_individuals(population,func)
        succession_phase(population,best_individual)
        if population[0]<best_individual:
            best_individual = deepcopy(population[0])
        trace.append(best_individual)
    return trace
