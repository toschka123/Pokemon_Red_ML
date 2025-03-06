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


def play_pop(pop):
    rewards = []
    
    for individual in pop:
        with open("state_file.state", "rb") as f:
            pyboy.load_state(f)
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
            pyboy.tick()
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
        
        

def get_team_levels():
    team_levels = []
    for i in range(6):  # Assuming a maximum of 6 Pokemon in the team
        #level_address = 0xD163 + i * 0x2C  # Address for each Pokemon's level
        level = pyboy.memory[0xD163 + i * 0x2C]
        team_levels.append(level)
    return sum(team_levels)

pop = 10
gen = 10
seq_length = 1000

pyboy = PyBoy("Pokemon Red.gb")  # Replace with your ROM filename

with open("state_file.state", "rb") as f:
    pyboy.load_state(f)





game_state = pyboy.memory[0xFFCF]
team_level_sum = get_team_levels()

population = initialize_pop(2, 10000)

rew = play_pop(population)
print(rew)
