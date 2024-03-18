from typing import Callable
import random
from utils import himmelblau_function
random.seed(0)

class Individual:
    def __init__(self,position:list[float]) -> None:
        self.position = position 
        self.value = None
        self.size = len(self.position)

    def evaluate_individual(self,func:Callable[[list[float],float],float])->None:
        self.value = func(*self.position)

    def __str__(self) -> str:
        return f"Position: {self.position}, value: {self.value}"
    
    def __lt__(self,other):
        return self.value<other.value
    
    def mutate(self,strength:float):
        pass


    @staticmethod
    def cross(individual1,individual2):
        result_position = []
        assert individual1.size == individual2.size
        for i in range(individual1.size):
            weight = random.random()
            result_position.append(individual1.position[i]*weight+individual1.position[i]*(1-weight))
        return Individual(result_position)


def generate_population(size:int,restrictions:list[tuple[float,float]])->list[Individual]:
    result = list()
    for _ in range(size):
        position = list()
        for limit in restrictions:
            position.append(random.uniform(limit[0],limit[1]))
        result.append(Individual(position))
    return result

def rate_individuals(population:list[Individual],func:Callable[[list[float],float],float])->None:
    for i in population:
        i.evaluate_individual(func)
    population.sort(key=lambda ind:ind.value)

def tournament_selection(population:list[Individual],competition=2)->list[Individual]:
    temp_population = []
    while len(temp_population)<=len(population):
        group = random.choices(population,k=competition)
        temp_population.append(min(group,key=lambda ind:ind.value))
    return temp_population


def crossover_phase(population:list[Individual],cross_probability:float)->None:
    temp_population = []
    for i,individual in enumerate(population):
        while i == parent_index:
            parent_index = random.choice(range(len(population)))
        if random.random()<cross_probability:
            parent2 = population[parent_index]
            temp_population.append(Individual.cross(individual,parent2))
        else:
            temp_population.append(individual)
    return temp_population

def mutate_phase()->None:
    pass

def succession_phase()->None:
    pass

def evolutionary_algorithm(func:Callable,start_population:list[Individual],iteration_time:int,mutation_strength:float,crossover_probability:float):
    population = start_population
    rate_individuals(population,func)
    best_individual = population[0]
    for step in range(iteration_time):
        population = tournament_selection(population,2)
        population = crossover_phase(population,crossover_probability)
        population = mutate_phase()
        population = rate_individuals()
        population = succession_phase()