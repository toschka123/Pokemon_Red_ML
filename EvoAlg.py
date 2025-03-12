import numpy as np 
import random
from pyboy import PyBoy
import math

def initialize_pop(pop, seq_length):
    individuals =[]
    for p in range(pop):
        individual= []
        for i in range(seq_length):
            individual.append(random.randint(0,6))
        individuals.append(individual)
    return individuals
        
def eval_fitness(indiv):
    return 0


def play_indiv(indiv):
    with open("state_file.state", "rb") as f:
        pyboy.load_state(f)
        
    for action in indiv:
        if i == 0:
            pyboy.button('a')
        elif i == 1:
            pyboy.button('b')
        elif i == 2:
            pyboy.button('up')
        elif i == 3:
            pyboy.button('down')
        elif i == 4:
            pyboy.button('left')
        elif i == 5:
            pyboy.button('right')
        elif i == 6:
            pyboy.button('start')
        pyboy.tick()

def play_pop(pop):
    rewards = []
    
    for individual in pop:
        with open("state_file.state", "rb") as f:
            pyboy.load_state(f)
        pyboy.set_emulation_speed(0) # No speed limit
        hist=[]
        hist.append(pyboy.game_area())
        reward = 0
        for i in individual:
            if i == 0:
                pyboy.button('a')
            elif i == 1:
                pyboy.button('b')
            elif i == 2:
                pyboy.button('up')
            elif i == 3:
                pyboy.button('down')
            elif i == 4:
                pyboy.button('left')
            elif i == 5:
                pyboy.button('right')
            elif i == 6:
                pyboy.button('start')
            pyboy.tick(1,True, False)
            current = pyboy.game_area()

            loss = math.inf
            for frame in hist:
                dist = np.linalg.norm(frame-current)
                if dist < loss:
                        loss = dist
            if loss > 2*math.e**23:
                reward += 0.01
                hist.append(current)
        reward += (get_team_levels())
        rewards.append(reward)
    return rewards
        
def survivor_select(pop, fit):
    survivors = []
    for i in range(len(pop)):
        participants = random.sample(range(0,len(pop)), 5) 
        best_fit_idx = participants[0]
        for j in participants:
            if fit[j] > fit[best_fit_idx]:
                best_fit_idx = j
        survivors.append(pop[best_fit_idx])
    return survivors

def uniform_crossover(population, alpha):
    children = []
    for individual in population:
        kid_a = []
        for gen in individual:
            if random.random() > alpha:
                kid_a.append(random.randint(0,6))
                #kid_a.append(gen + 1 if gen < 6 else 0)
            else:
                kid_a.append(gen)
        children.append(kid_a)
        #children.append(individual) 
    return children


                

def get_team_levels():
    team_levels = []
    for i in range(6):  # Assuming a maximum of 6 Pokemon in the team
        #level_address = 0xD163 + i * 0x2C  # Address for each Pokemon's level
        level = pyboy.memory[0xD163 + i * 0x2C]
        team_levels.append(level)
    return sum(team_levels)

pop = 50
gen = 20
seq_length = 100
alpha = 0.6
pyboy = PyBoy("Pokemon Red.gb")  # Replace with your ROM filename

with open("state_file.state", "rb") as f:
    pyboy.load_state(f)





game_state = pyboy.memory[0xFFCF]
team_level_sum = get_team_levels()

population = initialize_pop(pop, seq_length) #Maybe increase sequence as time goes on
generations = 10
#Itterations
for i in range(generations):
    rew = play_pop(population)
    survivors = survivor_select(population, rew)
    new_generation = uniform_crossover(survivors, alpha) #Make parameter smaller over mo itterations
    population = new_generation
    print(f"Generation {i} max fitness {max(rew)}, average is {sum(rew)/len(rew)}")
    alpha = alpha - 0.05
print(population)